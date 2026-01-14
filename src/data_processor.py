import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import ta
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Process market data: add technical indicators and normalize features.
    Uses StandardScaler instead of MinMaxScaler to preserve variance in returns.
    """
    
    def __init__(self):
        pass
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to OHLCV data.
        
        Args:
            df: DataFrame with columns [open_time, open, high, low, close, volume]
            
        Returns:
            DataFrame with added technical indicators
        """
        df = df.copy()
        
        # Basic features
        df['returns'] = df['close'].pct_change()
        df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        df['open_close_ratio'] = (df['close'] - df['open']) / df['open']
        
        # Moving averages
        df['price_to_sma_10'] = df['close'] / df['close'].rolling(window=10).mean()
        df['price_to_sma_20'] = df['close'] / df['close'].rolling(window=20).mean()
        
        # Volatility
        df['volatility_5'] = df['returns'].rolling(window=5).std()
        df['volatility_20'] = df['returns'].rolling(window=20).std()
        df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        df['ATR'] = df['ATR'] / df['close']  # Normalize by price
        
        # Momentum
        df['momentum_5'] = df['close'].diff(5) / df['close'].shift(5)
        df['momentum_10'] = df['close'].diff(10) / df['close'].shift(10)
        
        # RSI
        df['RSI'] = ta.momentum.rsi(df['close'], window=14) / 100  # Normalize to [0,1]
        
        # MACD - Handle different return formats
        try:
            macd_result = ta.trend.macd(df['close'])
            # Check if result is DataFrame or Series
            if isinstance(macd_result, pd.DataFrame):
                # macd_result has columns: macd, macd_signal, macd_diff
                df['MACD'] = macd_result.iloc[:, 0] / df['close']
                df['MACD_Signal'] = macd_result.iloc[:, 1] / df['close']
            else:
                # Single series returned, create MACD_Signal from macd_signal function
                df['MACD'] = macd_result / df['close']
                macd_signal = ta.trend.macd_signal(df['close'])
                if isinstance(macd_signal, pd.Series):
                    df['MACD_Signal'] = macd_signal / df['close']
                else:
                    df['MACD_Signal'] = 0.0
        except Exception as e:
            logger.warning(f"Error calculating MACD: {e}. Setting to 0.")
            df['MACD'] = 0.0
            df['MACD_Signal'] = 0.0
        
        # Bollinger Bands
        try:
            bb = ta.volatility.bollinger_channel(df['close'], window=20, scalar=2)
            if isinstance(bb, pd.DataFrame):
                df['BB_Position'] = (df['close'] - bb.iloc[:, 0]) / (bb.iloc[:, 2] - bb.iloc[:, 0])
            else:
                df['BB_Position'] = 0.5
        except Exception as e:
            logger.warning(f"Error calculating Bollinger Bands: {e}. Setting to 0.5.")
            df['BB_Position'] = 0.5
        
        df['BB_Position'] = df['BB_Position'].fillna(0.5)  # Fill NaN with middle
        
        # Volume
        df['Volume_Ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        # Returns std (volatility of returns)
        df['returns_std_5'] = df['returns'].rolling(window=5).std()
        
        # Drop NaN rows
        df = df.dropna()
        
        logger.info(f"Added technical indicators. Shape after dropna: {df.shape}")
        
        return df
    
    def normalize_features(
        self, 
        df: pd.DataFrame,
        feature_columns: list
    ) -> tuple:
        """
        Normalize features using StandardScaler (Z-score normalization).
        This preserves variance better than MinMaxScaler for returns prediction.
        
        Args:
            df: DataFrame with features
            feature_columns: List of column names to normalize
            
        Returns:
            (normalized_data: np.ndarray, scaler: StandardScaler)
        """
        # Check if all feature columns exist
        missing_cols = [col for col in feature_columns if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}. Using available columns.")
            feature_columns = [col for col in feature_columns if col in df.columns]
        
        data = df[feature_columns].values
        
        # Use StandardScaler instead of MinMaxScaler
        # This preserves variance in the data better for regression tasks
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(data)
        
        logger.info(f"Normalized {len(feature_columns)} features")
        logger.info(f"Scaler type: StandardScaler (Z-score)")
        
        if 'returns' in feature_columns:
            returns_idx = feature_columns.index('returns')
            logger.info(f"Returns mean: {normalized_data[:, returns_idx].mean():.6f}")
            logger.info(f"Returns std: {normalized_data[:, returns_idx].std():.6f}")
        
        return normalized_data, scaler
    
    def create_sequences(
        self,
        data: np.ndarray,
        seq_length: int,
        pred_length: int
    ) -> tuple:
        """
        Create sequences for LSTM training.
        Each sequence: X(seq_length bars) -> y(pred_length future bars)
        
        Args:
            data: Normalized feature array (num_samples, num_features)
            seq_length: Input sequence length (e.g., 100)
            pred_length: Output sequence length (e.g., 15)
            
        Returns:
            (X, y) where X shape is (num_sequences, seq_length, num_features)
                       and y shape is (num_sequences, pred_length, num_features)
        """
        X, y = [], []
        
        for i in range(len(data) - seq_length - pred_length + 1):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length:i + seq_length + pred_length])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Created sequences: X shape {X.shape}, y shape {y.shape}")
        
        return X, y
    
    def prepare_train_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.8
    ) -> tuple:
        """
        Split data into train/test sets.
        
        Args:
            X: Input sequences
            y: Target sequences
            train_ratio: Ratio of training data
            
        Returns:
            (X_train, X_test, y_train, y_test)
        """
        split_idx = int(len(X) * train_ratio)
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        logger.info(f"Train set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
