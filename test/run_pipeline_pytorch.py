import sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import torch
from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader
import gc
import pickle
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, load_config, create_directories, print_section
from src.gpu_manager import GPUManager
from src.data_loader import DataLoader
from src.data_processor import DataProcessor
from src.model_pytorch import LSTMModel
from src.evaluator import Evaluator

logger = None

def step_0_initialize_device(config):
    print_section("STEP 0: Device Initialization")
    
    cuda_available = torch.cuda.is_available()
    device = torch.device('cuda' if cuda_available else 'cpu')
    
    if cuda_available:
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"cuDNN Version: {torch.backends.cudnn.version()}")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        logger.warning("GPU not available. Using CPU.")
    
    logger.info(f"Using device: {device}")
    return device

def step_1_load_data(config):
    print_section("STEP 1: Loading Data")
    
    loader = DataLoader(
        repo_id=config['data']['repo_id'],
        cache_dir=Path(config['paths']['data_dir'])
    )
    
    df = loader.load_klines(
        symbol=config['data']['symbol'],
        timeframe=config['data']['timeframe']
    )
    
    if not loader.validate_data(df):
        raise ValueError("Data validation failed")
    
    logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"Date range: {df['open_time'].min()} to {df['open_time'].max()}")
    
    return df

def step_2_preprocess_data(df, config):
    print_section("STEP 2: Preprocessing Data (StandardScaler + Huber Loss)")
    
    processor = DataProcessor()
    
    # Store original df for time-based prediction later
    df_original = df.copy()
    
    if config['features']['use_technical_indicators']:
        df = processor.add_technical_indicators(df)
    
    feature_columns = config['features'].get('selected_features', [
        'returns', 'high_low_ratio', 'open_close_ratio', 
        'price_to_sma_10', 'price_to_sma_20', 'volatility_20',
        'volatility_5', 'momentum_5', 'momentum_10', 'ATR',
        'RSI', 'MACD', 'MACD_Signal', 'BB_Position', 'Volume_Ratio',
        'returns_std_5'
    ])
    
    logger.info(f"Using {len(feature_columns)} features for prediction:")
    for i, feat in enumerate(feature_columns, 1):
        logger.info(f"  {i:2d}. {feat}")
    
    # Use StandardScaler instead of MinMaxScaler for better variance preservation
    normalized_data, scaler = processor.normalize_features(df, feature_columns)
    
    logger.info(f"Data normalized using StandardScaler (Z-score).")
    logger.info(f"Normalized shape: {normalized_data.shape}")
    logger.info(f"Returns (idx=0) - mean: {normalized_data[:, 0].mean():.6f}, std: {normalized_data[:, 0].std():.6f}")
    logger.info(f"Volatility_5 (idx=6) - mean: {normalized_data[:, 6].mean():.6f}, std: {normalized_data[:, 6].std():.6f}")
    
    return normalized_data, scaler, feature_columns, processor, df

def step_3_create_sequences(normalized_data, config, df_with_timestamps):
    print_section("STEP 3: Creating Sequences")
    
    processor = DataProcessor()
    seq_length = config['model']['sequence_length']
    pred_length = config['model']['prediction_length']
    
    X, y = processor.create_sequences(normalized_data, seq_length, pred_length)
    
    # Store timestamps for time-based prediction
    # The i-th sequence corresponds to close_time at position (i + seq_length + pred_length - 1)
    sequence_end_indices = np.arange(seq_length + pred_length - 1, len(df_with_timestamps))
    
    logger.info(f"Sequences created: X shape {X.shape}, y shape {y.shape}")
    logger.info(f"Number of sequences: {len(X)}")
    logger.info(f"Sample volatility_5 (idx=6): mean={y[:, :, 6].mean():.6f}, std={y[:, :, 6].std():.6f}")
    logger.info(f"Sample volatility_20 (idx=5): mean={y[:, :, 5].mean():.6f}, std={y[:, :, 5].std():.6f}")
    
    return X, y, sequence_end_indices, df_with_timestamps

def step_4_split_data(X, y, config):
    print_section("STEP 4: Splitting Data")
    
    test_split = config['model']['test_split']
    val_split = config['model']['validation_split']
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_split, shuffle=False
    )
    
    val_size = int(len(X_temp) * val_split / (1 - test_split))
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, shuffle=False
    )
    
    logger.info(f"Data split:")
    logger.info(f"  Train: {X_train.shape[0]} samples")
    logger.info(f"  Val:   {X_val.shape[0]} samples")
    logger.info(f"  Test:  {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def step_5_build_model(config, num_features, device):
    print_section("STEP 5: Building Model (Huber Loss)")
    
    # Handle lstm_units as list or int
    lstm_units = config['model'].get('lstm_units', 128)
    if isinstance(lstm_units, list):
        lstm_units = lstm_units[0]  # Use first value
    
    model = LSTMModel(
        sequence_length=config['model']['sequence_length'],
        num_features=num_features,
        prediction_length=config['model']['prediction_length'],
        lstm_units=lstm_units,
        dropout_rate=config['model']['dropout_rate'],
        learning_rate=config['model']['learning_rate'],
        l2_reg=config['model'].get('l2_regularization', 0.0),
        device=device,
        use_huber_loss=True,
        huber_delta=0.5
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info("Model built successfully")
    logger.info(f"Architecture: 2-layer LSTM with Huber loss")
    logger.info(f"LSTM units: {lstm_units}")
    logger.info(f"Dropout: {config['model']['dropout_rate']}")
    logger.info(f"Loss: Huber (delta=0.5)")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    return model

def step_6_train_model(model, X_train, y_train, X_val, y_val, config, device):
    print_section("STEP 6: Training Model")
    
    # Get training parameters from config
    batch_size = config['model'].get('batch_size', 32)
    epochs = config['model'].get('epochs', 50)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Training parameters:")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {config['model']['learning_rate']}")
    logger.info(f"  Loss function: Huber")
    logger.info(f"  Normalization: StandardScaler")
    
    model.fit(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        verbose=True
    )
    
    model_path = Path(config['paths']['model_dir']) / "btc_15m_model_pytorch.pt"
    model.save(str(model_path))
    logger.info(f"Model weights saved to {model_path}")
    
    return model

def predict_in_batches(model, X, batch_size=4, device='cuda'):
    predictions = []
    total_samples = len(X)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    logger.info(f"Predicting {total_samples} samples with batch_size={batch_size}")
    
    model.eval()
    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            batch_end = min(i + batch_size, total_samples)
            batch = X_tensor[i:batch_end].to(device)
            
            try:
                batch_pred = model(batch)
                predictions.append(batch_pred.cpu().numpy())
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"OOM at batch {i//batch_size}, trying with single samples")
                    for j in range(i, batch_end):
                        single = X_tensor[j:j+1].to(device)
                        pred = model(single)
                        predictions.append(pred.cpu().numpy())
                else:
                    raise
            
            del batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if (i // batch_size + 1) % 100 == 0:
                logger.info(f"Processed {min(i + batch_size, total_samples)}/{total_samples} samples")
    
    logger.info(f"Prediction complete")
    return np.concatenate(predictions, axis=0)

def step_7_evaluate_model(model, X_test, y_test, config, device, feature_columns):
    print_section("STEP 7: Evaluating Model (Test Set)")
    
    evaluator = Evaluator(results_dir=config['paths']['results_dir'])
    eval_batch_size = 4
    
    logger.info("Making predictions on test set...")
    y_test_pred = predict_in_batches(model, X_test, batch_size=eval_batch_size, device=device)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("--- Test Set Metrics ---")
    logger.info(f"Prediction std: {y_test_pred.std():.6f}")
    logger.info(f"Target std: {y_test.std():.6f}")
    logger.info(f"Variance ratio (pred/target): {y_test_pred.std() / (y_test.std() + 1e-8):.4f}")
    
    # Feature-specific analysis
    logger.info("\n--- Feature-Specific Analysis ---")
    volatility_5_idx = feature_columns.index('volatility_5') if 'volatility_5' in feature_columns else -1
    volatility_20_idx = feature_columns.index('volatility_20') if 'volatility_20' in feature_columns else -1
    returns_idx = feature_columns.index('returns') if 'returns' in feature_columns else 0
    
    if volatility_5_idx >= 0:
        vol5_pred = y_test_pred[:, :, volatility_5_idx]
        vol5_true = y_test[:, :, volatility_5_idx]
        logger.info(f"volatility_5 - Pred std: {vol5_pred.std():.6f}, True std: {vol5_true.std():.6f}, Ratio: {vol5_pred.std() / (vol5_true.std() + 1e-8):.4f}")
    
    if volatility_20_idx >= 0:
        vol20_pred = y_test_pred[:, :, volatility_20_idx]
        vol20_true = y_test[:, :, volatility_20_idx]
        logger.info(f"volatility_20 - Pred std: {vol20_pred.std():.6f}, True std: {vol20_true.std():.6f}, Ratio: {vol20_pred.std() / (vol20_true.std() + 1e-8):.4f}")
    
    returns_pred = y_test_pred[:, :, returns_idx]
    returns_true = y_test[:, :, returns_idx]
    logger.info(f"returns - Pred std: {returns_pred.std():.6f}, True std: {returns_true.std():.6f}, Ratio: {returns_pred.std() / (returns_true.std() + 1e-8):.4f}")
    
    evaluator.calculate_metrics(y_test, y_test_pred)
    evaluator.print_metrics_summary()
    evaluator.save_metrics_report()
    
    logger.info("Model evaluation completed")
    
    del y_test_pred
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return evaluator

def step_8_summary(config, device):
    print_section("STEP 8: Execution Summary")
    
    logger.info(f"Project: {config['project']['name']}")
    logger.info(f"Symbol: {config['data']['symbol']}")
    logger.info(f"Timeframe: {config['data']['timeframe']}")
    logger.info(f"Device: {device}")
    logger.info(f"Framework: PyTorch")
    lstm_units = config['model'].get('lstm_units', 128)
    if isinstance(lstm_units, list):
        lstm_units = lstm_units[0]
    logger.info(f"Model: 2-Layer LSTM with {lstm_units} units")
    logger.info(f"Sequence Length: {config['model']['sequence_length']}")
    logger.info(f"Prediction Length: {config['model']['prediction_length']}")
    logger.info(f"Loss Function: Huber (delta=0.5)")
    logger.info(f"Normalization: StandardScaler (Z-score)")
    logger.info(f"L2 Regularization: {config['model'].get('l2_regularization', 0.0)}")
    logger.info(f"Dropout Rate: {config['model']['dropout_rate']}")
    logger.info(f"Results saved to: {config['paths']['results_dir']}")
    logger.info(f"Model saved to: {config['paths']['model_dir']}")
    logger.info(f"Logs saved to: {config['paths']['logs_dir']}")

def main():
    global logger
    
    config = load_config()
    logger = setup_logging(config['paths']['logs_dir'])
    create_directories(config)
    
    print_section("BTC_15m Cryptocurrency Price Prediction System (PyTorch)")
    logger.info(f"Configuration loaded from config/config.yaml")
    logger.info(f"Updates: StandardScaler + Huber Loss + Enhanced Volatility Features")
    
    try:
        device = step_0_initialize_device(config)
        df = step_1_load_data(config)
        normalized_data, scaler, feature_columns, processor, df_processed = step_2_preprocess_data(df, config)
        X, y, seq_indices, df_with_ts = step_3_create_sequences(normalized_data, config, df_processed)
        X_train, X_val, X_test, y_train, y_val, y_test = step_4_split_data(X, y, config)
        model = step_5_build_model(config, len(feature_columns), device)
        model = step_6_train_model(model, X_train, y_train, X_val, y_val, config, device)
        evaluator = step_7_evaluate_model(model, X_test, y_test, config, device, feature_columns)
        step_8_summary(config, device)
        
        # Save scaler for later use in app
        scaler_path = Path(config['paths']['model_dir']) / "btc_15m_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info(f"Scaler saved to {scaler_path}")
        
        # Save feature columns for later use
        features_path = Path(config['paths']['model_dir']) / "feature_columns.pkl"
        with open(features_path, 'wb') as f:
            pickle.dump(feature_columns, f)
        logger.info(f"Feature columns saved to {features_path}")
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
