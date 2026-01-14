import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import logging

sys.path.insert(0, str(Path(__file__).parent))

from src.utils import load_config, setup_logging
from src.data_loader import DataLoader
from src.data_processor import DataProcessor
from src.model_pytorch import LSTMModel

def train():
    # Setup
    config = load_config()
    logger = setup_logging(config['paths']['logs_dir'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading data...")
    loader = DataLoader(
        repo_id=config['data']['repo_id'],
        cache_dir=Path(config['paths']['data_dir'])
    )
    df = loader.load_klines(
        symbol=config['data']['symbol'],
        timeframe=config['data']['timeframe']
    )
    
    # Process data
    logger.info("Processing data...")
    processor = DataProcessor()
    df_processed = processor.add_technical_indicators(df)
    
    feature_columns = config['features'].get('selected_features', [
        'returns', 'high_low_ratio', 'open_close_ratio',
        'price_to_sma_10', 'price_to_sma_20', 'volatility_20',
        'volatility_5', 'momentum_5', 'momentum_10', 'ATR',
        'RSI', 'MACD', 'MACD_Signal', 'BB_Position',
        'Volume_Ratio', 'returns_std_5'
    ])
    
    normalized_data, scaler = processor.normalize_features(df_processed, feature_columns)
    
    # Create sequences
    logger.info("Creating sequences...")
    X, y = processor.create_sequences(
        normalized_data,
        config['model']['sequence_length'],
        config['model']['prediction_length']
    )
    
    # Split data
    X_train, X_test, y_train, y_test = processor.prepare_train_test_split(X, y)
    
    # Create model
    logger.info("Creating model...")
    model = LSTMModel(
        sequence_length=config['model']['sequence_length'],
        num_features=len(feature_columns),
        prediction_length=config['model']['prediction_length'],
        lstm_units=config['model']['lstm_units'],
        dropout_rate=config['model']['dropout_rate'],
        learning_rate=config['model']['learning_rate'],
        l2_reg=config['model'].get('l2_regularization', 0.0),
        device=str(device),
        use_huber_loss=True,  # Use Huber loss instead of MSE
        huber_delta=0.5
    )
    
    # Train
    logger.info("Starting training...")
    logger.info(f"Training parameters:")
    logger.info(f"  Epochs: {config['training']['epochs']}")
    logger.info(f"  Batch size: {config['training']['batch_size']}")
    logger.info(f"  Learning rate: {config['model']['learning_rate']}")
    logger.info(f"  Loss function: Huber (delta=0.5)")
    logger.info(f"  Normalization: StandardScaler (Z-score)")
    
    model.fit(
        X_train, y_train,
        X_test, y_test,
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        verbose=True
    )
    
    # Evaluate
    logger.info("Evaluating model...")
    from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
    
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    )
    test_loader = TorchDataLoader(test_dataset, batch_size=config['training']['batch_size'])
    
    test_loss, mae, rmse, r2 = model.evaluate(test_loader)
    logger.info(f"\nFinal Test Results:")
    logger.info(f"  Loss: {test_loss:.6f}")
    logger.info(f"  MAE: {mae:.6f}")
    logger.info(f"  RMSE: {rmse:.6f}")
    logger.info(f"  R² Score: {r2:.4f}")
    
    # Save model and scaler
    model_dir = Path(config['paths']['model_dir'])
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model.save(str(model_dir / "btc_15m_model_pytorch.pt"))
    
    # Save scaler
    import pickle
    scaler_path = model_dir / "btc_15m_scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler saved to {scaler_path}")
    
    logger.info("\nTraining completed!")
    return model, scaler

if __name__ == "__main__":
    train()
