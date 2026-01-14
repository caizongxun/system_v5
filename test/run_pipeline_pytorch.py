import sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader
import gc
import pickle
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, load_config, create_directories, print_section
from src.gpu_manager import GPUManager
from src.data_loader import DataLoader

logger = None

# ==================== EncoderDecoder LSTM Model ====================

class EncoderDecoderLSTM(nn.Module):
    """Encoder-Decoder LSTM for multi-step ahead forecasting
    
    Reference: Machine Learning Mastery - Multi-step LSTM Time Series Forecasting
    This architecture prevents error accumulation in recursive predictions
    """
    def __init__(self, input_size, hidden_size, num_layers, forecast_horizon, dropout_rate=0.2, l2_reg=0.0, device='cpu'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        self.device = device
        self.l2_reg = l2_reg
        
        # Encoder - compress historical sequence
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Decoder - generate future sequence
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Output layer - predict all features for each time step
        self.fc = nn.Linear(hidden_size, input_size)
        
        # Optimizer and loss
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=l2_reg)
        self.criterion = nn.HuberLoss(delta=0.5)
        
    def forward(self, x):
        """Forward pass for encoder-decoder
        
        Args:
            x: (batch_size, seq_length, input_size)
            
        Returns:
            predictions: (batch_size, forecast_horizon, input_size)
        """
        # Encoder: compress sequence to context vector
        encoder_out, (h_n, c_n) = self.encoder(x)
        
        # Decoder: use last encoder output as initial input
        decoder_input = encoder_out[:, -1:, :]  # (batch, 1, hidden_size)
        
        predictions = []
        
        # Iteratively decode forecast_horizon steps
        for _ in range(self.forecast_horizon):
            decoder_out, (h_n, c_n) = self.decoder(decoder_input, (h_n, c_n))
            output = self.fc(decoder_out)  # (batch, 1, input_size)
            predictions.append(output)
            decoder_input = decoder_out  # Use decoder output as next input
        
        # Concatenate all predictions: (batch, forecast_horizon, input_size)
        return torch.cat(predictions, dim=1)
    
    def fit(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, verbose=True):
        """Train the model"""
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32)
        )
        
        train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience_limit = 20
        
        self.to(self.device)
        
        for epoch in range(epochs):
            # Training
            self.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                y_pred = self(X_batch)
                loss = self.criterion(y_pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    y_pred = self(X_batch)
                    loss = self.criterion(y_pred, y_batch)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0 and verbose:
                logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
    
    def predict(self, X, batch_size=4):
        """Make predictions on data"""
        predictions = []
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        self.eval()
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_end = min(i + batch_size, len(X))
                batch = X_tensor[i:batch_end].to(self.device)
                
                try:
                    batch_pred = self(batch)
                    predictions.append(batch_pred.cpu().numpy())
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.warning(f"OOM at batch {i//batch_size}, trying single samples")
                        for j in range(i, batch_end):
                            single = X_tensor[j:j+1].to(self.device)
                            pred = self(single)
                            predictions.append(pred.cpu().numpy())
                    else:
                        raise
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return np.concatenate(predictions, axis=0)
    
    def save(self, path):
        """Save model weights"""
        torch.save(self.state_dict(), path)
        logger.info(f"Model saved to {path}")

# ==================== Technical Indicators Calculation ====================

def calculate_technical_indicators(df):
    """Calculate 17 recommended technical indicators
    
    Based on research: Fidan (2024), LSTM-GARCH hybrid models
    """
    df = df.copy()
    
    # 1. Returns
    df['returns'] = df['close'].pct_change().fillna(0)
    
    # 2. Log Returns
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
    
    # 3. Volatility (20-day rolling std)
    df['volatility_20'] = df['returns'].rolling(window=20).std().fillna(0)
    
    # 4. Volatility (5-day rolling std)
    df['volatility_5'] = df['returns'].rolling(window=5).std().fillna(0)
    
    # 5. ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean().fillna(0)
    
    # 6. RSI (14-period)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].fillna(50)
    
    # 7-9. MACD (12, 26, 9)
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']
    
    # 10-12. SMAs (7, 14, 21)
    df['sma_7'] = df['close'].rolling(window=7).mean().bfill().ffill()
    df['sma_14'] = df['close'].rolling(window=14).mean().bfill().ffill()
    df['sma_21'] = df['close'].rolling(window=21).mean().bfill().ffill()
    
    # 13. KAMA (Kaufman's Adaptive Moving Average)
    period = 10
    fastsc = 2 / (2 + 1)
    slowsc = 2 / (30 + 1)
    
    change = abs(df['close'].diff(period))
    volatility = df['close'].diff().abs().rolling(period).sum()
    efficiency_ratio = change / (volatility + 1e-8)
    smoothing_constant = efficiency_ratio * (fastsc - slowsc) + slowsc
    
    kama = [0] * len(df)
    for i in range(period, len(df)):
        if i == period:
            kama[i] = df['close'].iloc[i]
        else:
            kama[i] = kama[i-1] + (smoothing_constant.iloc[i] ** 2) * (df['close'].iloc[i] - kama[i-1])
    df['kama'] = kama
    
    # 14. High-Low Ratio
    df['high_low_ratio'] = (df['high'] - df['low']) / (df['close'] + 1e-8)
    
    # 15. Open-Close Ratio
    df['open_close_ratio'] = (df['close'] - df['open']) / (df['open'] + 1e-8)
    
    # 16-17. Price to SMA ratios
    df['price_to_sma_10'] = df['close'] / (df['sma_7'] + 1e-8)
    df['price_to_sma_20'] = df['close'] / (df['sma_14'] + 1e-8)
    
    # Fill NaN values
    df = df.bfill().ffill()
    
    return df

# ==================== Sequence Creation ====================

def create_sequences(data, lookback, forecast_horizon):
    """Create sequences for LSTM input"""
    X, y = [], []
    for i in range(len(data) - lookback - forecast_horizon + 1):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback:i+lookback+forecast_horizon])
    return np.array(X), np.array(y)

# ==================== Pipeline Steps ====================

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
    
    logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"Date range: {df.get('open_time', pd.Series(index=df.index)).min()} to {df.get('open_time', pd.Series(index=df.index)).max()}")
    
    return df

def step_2_calculate_features(df, config):
    print_section("STEP 2: Feature Engineering (17 Recommended Features)")
    
    logger.info("Calculating 17 technical indicators...")
    df = calculate_technical_indicators(df)
    
    # Select key features for LSTM
    feature_columns = [
        'returns', 'log_returns',
        'volatility_20', 'volatility_5',
        'atr', 'rsi', 'macd', 'macd_signal', 'macd_diff',
        'sma_7', 'sma_14', 'sma_21', 'kama',
        'high_low_ratio', 'open_close_ratio',
        'price_to_sma_10', 'price_to_sma_20'
    ]
    
    logger.info(f"Using {len(feature_columns)} features:")
    for i, feat in enumerate(feature_columns, 1):
        logger.info(f"  {i:2d}. {feat}")
    
    return df, feature_columns

def step_3_normalize_data(df, feature_columns):
    print_section("STEP 3: Data Normalization (StandardScaler)")
    
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(df[feature_columns])
    
    logger.info(f"Data normalized using StandardScaler (Z-score)")
    logger.info(f"Normalized shape: {normalized_data.shape}")
    logger.info(f"Mean: {normalized_data.mean():.6f}, Std: {normalized_data.std():.6f}")
    
    return normalized_data, scaler

def step_4_create_sequences(normalized_data, config):
    print_section("STEP 4: Creating Sequences")
    
    lookback = config['model']['sequence_length']
    forecast_horizon = config['model']['prediction_length']
    
    X, y = create_sequences(normalized_data, lookback, forecast_horizon)
    
    logger.info(f"Sequences created: X={X.shape}, y={y.shape}")
    logger.info(f"Number of samples: {len(X)}")
    logger.info(f"Lookback window: {lookback}, Forecast horizon: {forecast_horizon}")
    
    return X, y

def step_5_split_data(X, y, config):
    print_section("STEP 5: Data Split")
    
    test_split = config['model']['test_split']
    val_split = config['model']['validation_split']
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_split, shuffle=False
    )
    
    val_size = int(len(X_temp) * val_split / (1 - test_split))
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, shuffle=False
    )
    
    logger.info(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def step_6_build_model(config, num_features, device):
    print_section("STEP 6: Building EncoderDecoder LSTM Model")
    
    model = EncoderDecoderLSTM(
        input_size=num_features,
        hidden_size=config['model'].get('lstm_units', 256),
        num_layers=2,
        forecast_horizon=config['model']['prediction_length'],
        dropout_rate=config['model']['dropout_rate'],
        l2_reg=config['model'].get('l2_regularization', 0.0),
        device=device
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model: EncoderDecoder LSTM")
    logger.info(f"  Input features: {num_features}")
    logger.info(f"  Hidden size: {config['model'].get('lstm_units', 256)}")
    logger.info(f"  Forecast horizon: {config['model']['prediction_length']}")
    logger.info(f"  Dropout: {config['model']['dropout_rate']}")
    logger.info(f"  L2 Reg: {config['model'].get('l2_regularization', 0.0)}")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    
    return model

def step_7_train_model(model, X_train, y_train, X_val, y_val, config, device):
    print_section("STEP 7: Training Model")
    
    logger.info(f"Training parameters:")
    logger.info(f"  Epochs: {config['model']['epochs']}")
    logger.info(f"  Batch size: {config['model']['batch_size']}")
    logger.info(f"  Learning rate: {config['model']['learning_rate']}")
    logger.info(f"  Loss function: Huber (delta=0.5)")
    logger.info(f"  Normalization: StandardScaler")
    
    model.fit(
        X_train, y_train,
        X_val, y_val,
        epochs=config['model']['epochs'],
        batch_size=config['model']['batch_size'],
        verbose=True
    )
    
    model_path = Path(config['paths']['model_dir']) / "encoder_decoder_lstm.pt"
    model.save(str(model_path))
    
    return model

def step_8_evaluate_model(model, X_test, y_test, config, device):
    print_section("STEP 8: Model Evaluation")
    
    logger.info("Making predictions on test set...")
    y_pred = model.predict(X_test, batch_size=4)
    
    # Calculate metrics
    mse = np.mean((y_pred - y_test) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred - y_test))
    
    logger.info(f"")
    logger.info(f"--- Test Metrics ---")
    logger.info(f"MSE:  {mse:.6f}")
    logger.info(f"RMSE: {rmse:.6f}")
    logger.info(f"MAE:  {mae:.6f}")
    logger.info(f"Prediction std: {y_pred.std():.6f}")
    logger.info(f"Target std: {y_test.std():.6f}")
    
    return y_pred

def step_9_summary(config, device):
    print_section("STEP 9: Execution Summary")
    
    logger.info(f"Project: {config['project']['name']}")
    logger.info(f"Symbol: {config['data']['symbol']}")
    logger.info(f"Timeframe: {config['data']['timeframe']}")
    logger.info(f"Device: {device}")
    logger.info(f"Framework: PyTorch")
    logger.info(f"Model: EncoderDecoder LSTM (2 layers)")
    logger.info(f"Architecture: {config['model']['sequence_length']} to {config['model']['prediction_length']}")
    logger.info(f"Features: 17 Technical Indicators (Research-Backed)")
    logger.info(f"Loss: Huber (delta=0.5)")
    logger.info(f"Normalization: StandardScaler")
    logger.info(f"Results saved to: {config['paths']['results_dir']}")
    logger.info(f"")
    logger.info(f"Pipeline completed successfully!")

def main():
    global logger
    
    config = load_config()
    logger = setup_logging(config['paths']['logs_dir'])
    create_directories(config)
    
    print_section("EncoderDecoder LSTM Multi-Step Forecasting System")
    logger.info("Configuration: 17 Features + Encoder-Decoder Architecture")
    
    try:
        device = step_0_initialize_device(config)
        df = step_1_load_data(config)
        df, feature_columns = step_2_calculate_features(df, config)
        normalized_data, scaler = step_3_normalize_data(df, feature_columns)
        X, y = step_4_create_sequences(normalized_data, config)
        X_train, X_val, X_test, y_train, y_val, y_test = step_5_split_data(X, y, config)
        model = step_6_build_model(config, len(feature_columns), device)
        model = step_7_train_model(model, X_train, y_train, X_val, y_val, config, device)
        y_pred = step_8_evaluate_model(model, X_test, y_test, config, device)
        step_9_summary(config, device)
        
        # Save scaler and features
        scaler_path = Path(config['paths']['model_dir']) / "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info(f"Scaler saved to {scaler_path}")
        
        features_path = Path(config['paths']['model_dir']) / "feature_columns.pkl"
        with open(features_path, 'wb') as f:
            pickle.dump(feature_columns, f)
        logger.info(f"Feature columns saved to {features_path}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
