import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging
from typing import Tuple, Dict

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.scaler_dict = {}
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to dataframe.
        Focus on relative changes instead of absolute prices.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added indicators
        """
        df = df.copy()
        
        # Price changes (relative, not absolute)
        df['returns'] = df['close'].pct_change()
        df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        df['open_close_ratio'] = (df['close'] - df['open']) / df['open']
        
        # Moving averages (normalized)
        df['SMA_10'] = df['close'].rolling(window=10).mean()
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        
        # Price relative to moving averages
        df['price_to_sma_10'] = (df['close'] - df['SMA_10']) / df['SMA_10']
        df['price_to_sma_20'] = (df['close'] - df['SMA_20']) / df['SMA_20']
        
        # Volatility (relative)
        df['volatility_20'] = df['returns'].rolling(window=20).std()
        
        df['RSI'] = self._calculate_rsi(df['close'], period=14)
        
        macd_result = self._calculate_macd(df['close'])
        df['MACD'] = macd_result['MACD']
        df['MACD_Signal'] = macd_result['Signal']
        df['MACD_Diff'] = macd_result['Diff']
        
        bb_result = self._calculate_bollinger_bands(df['close'], period=20)
        df['BB_Upper'] = bb_result['Upper']
        df['BB_Middle'] = bb_result['Middle']
        df['BB_Lower'] = bb_result['Lower']
        df['BB_Position'] = (df['close'] - bb_result['Lower']) / (bb_result['Upper'] - bb_result['Lower'])
        
        df['Volume_SMA'] = df['volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['volume'] / df['Volume_SMA']
        
        df = df.dropna()
        logger.info(f"Added technical indicators. Shape after dropna: {df.shape}")
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        
        return {
            'MACD': macd,
            'Signal': signal_line,
            'Diff': histogram
        }
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Dict:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        
        return {
            'Upper': upper,
            'Middle': sma,
            'Lower': lower
        }
    
    def normalize_features(self, df: pd.DataFrame, feature_columns: list) -> Tuple[np.ndarray, MinMaxScaler]:
        """
        Normalize features using MinMaxScaler
        
        Args:
            df: DataFrame with features
            feature_columns: List of columns to normalize
            
        Returns:
            Normalized array and scaler object
        """
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(df[feature_columns])
        logger.info(f"Normalized {len(feature_columns)} features")
        return normalized_data, scaler
    
    def create_sequences(self, data: np.ndarray, seq_length: int, pred_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training
        
        Args:
            data: Normalized data array
            seq_length: Input sequence length (100)
            pred_length: Prediction sequence length (15)
            
        Returns:
            X, y sequences
        """
        X, y = [], []
        
        for i in range(len(data) - seq_length - pred_length + 1):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length:i + seq_length + pred_length])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Created sequences: X shape {X.shape}, y shape {y.shape}")
        return X, y
