import sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import torch
from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader
import gc

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, load_config, create_directories, print_section
from src.data_loader import DataLoader
from src.data_processor import DataProcessor
from src.model_pytorch import LSTMModel
from src.evaluator import Evaluator

logger = None

def predict_in_batches(model, X, batch_size=4, device='cuda'):
    """
    Make predictions in very small batches to avoid memory issues
    
    Args:
        model: Trained model
        X: Input data as numpy array
        batch_size: Batch size for prediction
        device: Device to use
        
    Returns:
        Predictions as numpy array
    """
    predictions = []
    total_samples = len(X)
    
    # Convert to tensor on CPU first
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    logger.info(f"Predicting {total_samples} samples with batch_size={batch_size}")
    
    # Process in very small batches
    for i in range(0, total_samples, batch_size):
        batch_end = min(i + batch_size, total_samples)
        batch = X_tensor[i:batch_end].to(device)
        
        try:
            with torch.no_grad():
                batch_pred = model(batch)
            predictions.append(batch_pred.cpu().numpy())
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"OOM at batch {i//batch_size}, trying with single samples")
                # Try again with single sample prediction
                for j in range(i, batch_end):
                    single = X_tensor[j:j+1].to(device)
                    with torch.no_grad():
                        pred = model(single)
                    predictions.append(pred.cpu().numpy())
            else:
                raise
        
        # Free memory after each batch
        del batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Progress indicator
        if (i // batch_size + 1) % 100 == 0:
            logger.info(f"Processed {min(i + batch_size, total_samples)}/{total_samples} samples")
    
    logger.info(f"Prediction complete")
    return np.concatenate(predictions, axis=0)

def main():
    global logger
    
    config = load_config()
    
    logger = setup_logging(config['paths']['logs_dir'])
    
    create_directories(config)
    
    print_section("BTC_15m Model Evaluation Only (PyTorch)")
    logger.info(f"Configuration loaded from config/config.yaml")
    
    try:
        # Initialize device
        cuda_available = torch.cuda.is_available()
        device = torch.device('cuda' if cuda_available else 'cpu')
        
        if cuda_available:
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("GPU not available. Using CPU.")
        
        logger.info(f"Using device: {device}")
        
        # Load data
        print_section("Loading and Preprocessing Data")
        loader = DataLoader(
            repo_id=config['data']['repo_id'],
            cache_dir=Path(config['paths']['data_dir'])
        )
        
        df = loader.load_klines(
            symbol=config['data']['symbol'],
            timeframe=config['data']['timeframe']
        )
        
        logger.info(f"Data loaded: {df.shape[0]} rows")
        
        # Preprocess
        processor = DataProcessor()
        
        if config['features']['use_technical_indicators']:
            df = processor.add_technical_indicators(df)
        
        feature_columns = ['open', 'high', 'low', 'close', 'volume']
        if config['features']['use_technical_indicators']:
            feature_columns.extend(['SMA_10', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal'])
        
        logger.info(f"Using {len(feature_columns)} features")
        
        normalized_data, scaler = processor.normalize_features(df, feature_columns)
        
        # Create sequences
        seq_length = config['model']['sequence_length']
        pred_length = config['model']['prediction_length']
        X, y = processor.create_sequences(normalized_data, seq_length, pred_length)
        
        logger.info(f"Sequences created: {X.shape[0]} samples")
        
        # Split data
        test_split = config['model']['test_split']
        val_split = config['model']['validation_split']
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_split, shuffle=False
        )
        
        val_size = int(len(X_temp) * val_split / (1 - test_split))
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, shuffle=False
        )
        
        logger.info(f"Data split: Train {X_train.shape[0]}, Val {X_val.shape[0]}, Test {X_test.shape[0]}")
        
        # Load trained model
        print_section("Loading Trained Model")
        model = LSTMModel(
            sequence_length=config['model']['sequence_length'],
            num_features=len(feature_columns),
            prediction_length=config['model']['prediction_length'],
            lstm_units=config['model']['lstm_units'],
            dropout_rate=config['model']['dropout_rate'],
            learning_rate=config['model']['learning_rate'],
            l2_reg=config['model'].get('l2_regularization', 0.0),
            device=device
        )
        
        model_path = Path(config['paths']['model_dir']) / "btc_15m_model_pytorch.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        model.load(str(model_path))
        logger.info(f"Model loaded from {model_path}")
        
        # Evaluate
        print_section("Evaluating Model (Test Set Only)")
        
        evaluator = Evaluator(results_dir=config['paths']['results_dir'])
        eval_batch_size = 4
        
        logger.info("Making predictions on test set...")
        y_test_pred = predict_in_batches(model, X_test, batch_size=eval_batch_size, device=device)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Calculate and save metrics
        logger.info("--- Test Set Metrics ---")
        evaluator.calculate_metrics(y_test, y_test_pred)
        evaluator.print_metrics_summary()
        evaluator.save_metrics_report()
        
        # Plot results
        if model.history is not None:
            evaluator.plot_training_history(model.history)
        
        evaluator.plot_predictions(y_test, y_test_pred, sample_idx=0)
        
        logger.info("\nModel evaluation completed successfully!")
        logger.info(f"\nResults saved to: {config['paths']['results_dir']}")
        
        # Clean up memory
        del y_test_pred
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    except Exception as e:
        logger.error(f"Evaluation failed with error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
