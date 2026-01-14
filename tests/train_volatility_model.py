import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
from sklearn.preprocessing import StandardScaler
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config
from src.data_processor import DataProcessor
from src.volatility_model import VolatilityModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_volatility_targets(df: pd.DataFrame, pred_steps: int = 15, lookback: int = 5) -> np.ndarray:
    """
    Prepare volatility targets for training.
    
    For each historical point, calculate the volatility (absolute movement)
    for the next pred_steps bars.
    
    Args:
        df: DataFrame with close prices
        pred_steps: How many bars ahead to measure volatility
        lookback: Lookback period for volatility calculation
        
    Returns:
        (num_samples, pred_steps) array of volatility targets
    """
    volatilities = []
    closes = df['close'].values
    
    # Calculate rolling volatility
    for i in range(len(closes) - pred_steps):
        step_volatilities = []
        for j in range(pred_steps):
            current_idx = i + j
            if current_idx + lookback < len(closes):
                # Get returns for next lookback bars
                future_returns = np.abs(closes[current_idx+1:current_idx+lookback+1] / closes[current_idx] - 1)
                volatility = future_returns.mean()
                step_volatilities.append(volatility)
            else:
                step_volatilities.append(0.0)
        
        if len(step_volatilities) == pred_steps:
            volatilities.append(step_volatilities)
    
    return np.array(volatilities)

def train_volatility_model(
    config_path: str = 'config.yaml',
    data_path: str = None,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001
):
    """
    Train volatility magnitude prediction model.
    
    Args:
        config_path: Path to config file
        data_path: Path to training data CSV
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
    """
    
    # Load configuration
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    if data_path is None:
        data_path = config['paths']['data_dir'] / 'BTC_15m_historical.csv'
    
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Process data
    processor = DataProcessor()
    df_processed = processor.add_technical_indicators(df)
    
    feature_columns = config['features'].get('selected_features', [
        'returns', 'high_low_ratio', 'open_close_ratio',
        'price_to_sma_10', 'price_to_sma_20', 'volatility_20',
        'volatility_5', 'momentum_5', 'momentum_10', 'ATR',
        'RSI', 'MACD', 'MACD_Signal', 'BB_Position',
        'Volume_Ratio', 'returns_std_5'
    ])
    
    # Prepare features
    X_data = df_processed[feature_columns].values
    
    # Normalize features
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X_data)
    
    # Prepare volatility targets
    logger.info("Preparing volatility targets...")
    y_volatility = prepare_volatility_targets(df_processed)
    
    logger.info(f"Features shape: {X_normalized.shape}")
    logger.info(f"Volatility targets shape: {y_volatility.shape}")
    logger.info(f"Volatility range: {y_volatility.min():.6f} to {y_volatility.max():.6f}")
    logger.info(f"Volatility mean: {y_volatility.mean():.6f}, std: {y_volatility.std():.6f}")
    
    # Create sliding windows
    seq_length = config['model']['sequence_length']
    pred_length = config['model']['prediction_length']
    
    X_windows = []
    y_windows = []
    
    for i in range(len(X_normalized) - seq_length - pred_length):
        X_windows.append(X_normalized[i:i+seq_length])
        y_windows.append(y_volatility[i:i+pred_length])
    
    X_windows = np.array(X_windows)
    y_windows = np.array(y_windows)
    
    logger.info(f"Windows shape - X: {X_windows.shape}, y: {y_windows.shape}")
    
    # Create dataset and dataloader
    X_tensor = torch.tensor(X_windows, dtype=torch.float32)
    y_tensor = torch.tensor(y_windows, dtype=torch.float32)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Initialize model
    model = VolatilityModel(
        sequence_length=seq_length,
        num_features=len(feature_columns),
        prediction_length=pred_length,
        lstm_units=config['model'].get('lstm_units', 64),
        dropout_rate=config['model'].get('dropout_rate', 0.2),
        learning_rate=learning_rate,
        device=str(device)
    )
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    logger.info(f"Starting training for {epochs} epochs...")
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        
        val_loss /= len(val_dataset)
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            model_path = Path(config['paths']['model_dir']) / "btc_15m_volatility_model.pt"
            model.save(str(model_path))
            logger.info(f"Best model saved - Val Loss: {val_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Save scaler
    scaler_path = Path(config['paths']['model_dir']) / "volatility_scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler saved to {scaler_path}")
    
    logger.info("Training completed!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train volatility prediction model")
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--data', type=str, default=None, help='Training data path')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    train_volatility_model(
        config_path=args.config,
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
