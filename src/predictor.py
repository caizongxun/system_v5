import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class RollingPredictor:
    """
    Predictor that maintains consistency across multiple time steps.
    Uses a rolling window approach:
    1. Make predictions for next 15 bars using current 100-bar window
    2. When moving to next time step, use actual data + previous predictions
    3. Ensures consistent baseline for predictions
    """
    
    def __init__(self, model, scaler, feature_columns, device='cuda'):
        """
        Args:
            model: Trained LSTM model
            scaler: StandardScaler fitted on historical data
            feature_columns: List of feature names
            device: 'cuda' or 'cpu'
        """
        self.model = model
        self.scaler = scaler
        self.feature_columns = feature_columns
        self.device = device
        self.prediction_history = []
    
    def predict_single_step(self, X_normalized: np.ndarray) -> np.ndarray:
        """
        Make a single prediction (15 bars into future)
        
        Args:
            X_normalized: (100, num_features) normalized sequence
            
        Returns:
            (15, num_features) predictions
        """
        X_tensor = torch.tensor(X_normalized.reshape(1, -1, len(self.feature_columns)), 
                               dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(X_tensor)
        
        return prediction.cpu().numpy()[0]
    
    def denormalize_prediction(self, pred_normalized: np.ndarray) -> np.ndarray:
        """
        Convert normalized predictions back to original scale
        """
        return self.scaler.inverse_transform(pred_normalized)
    
    def generate_klines_from_prediction(
        self, 
        pred_denorm: np.ndarray,
        feature_columns: List[str],
        volatility_multiplier: float = 2.0
    ) -> List[Dict]:
        """
        Convert predicted features to K-line OHLCV data.
        Uses directly predicted prices (price_open, price_high, price_low, price_close, price_volume)
        instead of reconstructing from indicators.
        
        Args:
            pred_denorm: (15, num_features) denormalized predictions
            feature_columns: List of feature names
            volatility_multiplier: Multiplier to enhance visible volatility (unused when using direct prices)
            
        Returns:
            List of 15 predicted candles
        """
        klines = []
        
        # Get feature indices for OHLCV prices
        try:
            price_open_idx = feature_columns.index('price_open')
            price_high_idx = feature_columns.index('price_high')
            price_low_idx = feature_columns.index('price_low')
            price_close_idx = feature_columns.index('price_close')
            price_volume_idx = feature_columns.index('price_volume')
            has_prices = True
        except ValueError:
            has_prices = False
            logger.warning("Price features not found in predictions. Using indicator-based reconstruction.")
        
        if has_prices:
            # Use directly predicted prices
            for i in range(len(pred_denorm)):
                pred_features = pred_denorm[i]
                
                open_price = max(float(pred_features[price_open_idx]), 1.0)
                high_price = max(float(pred_features[price_high_idx]), open_price)
                low_price = min(float(pred_features[price_low_idx]), open_price)
                close_price = float(pred_features[price_close_idx])
                volume = max(float(pred_features[price_volume_idx]), 1000)
                
                # Ensure realistic OHLC relationships
                high_price = max(high_price, max(open_price, close_price))
                low_price = min(low_price, min(open_price, close_price))
                
                current_time = pd.to_datetime('now') + timedelta(minutes=15 * (i + 1))
                
                klines.append({
                    'time': current_time,
                    'open': float(open_price),
                    'high': float(high_price),
                    'low': float(low_price),
                    'close': float(close_price),
                    'volume': float(volume),
                    'type': 'predicted',
                    'step': i + 1
                })
        else:
            # Fallback: reconstruct from indicators
            logger.info("Using indicator-based K-line reconstruction as fallback.")
            
            try:
                returns_idx = feature_columns.index('returns')
                high_low_ratio_idx = feature_columns.index('high_low_ratio')
                volume_ratio_idx = feature_columns.index('Volume_Ratio')
                volatility_5_idx = feature_columns.index('volatility_5')
            except ValueError:
                logger.error("Required indicator columns not found.")
                return []
            
            current_close = 100.0  # Default reference price
            current_volume = 10000  # Default volume
            
            for i in range(len(pred_denorm)):
                pred_features = pred_denorm[i]
                
                returns = np.clip(pred_features[returns_idx], -0.05, 0.05)
                high_low_ratio = abs(pred_features[high_low_ratio_idx])
                volume_ratio = np.clip(pred_features[volume_ratio_idx], 0.3, 3.0)
                
                open_price = current_close
                close_price = open_price * (1 + returns)
                range_size = high_low_ratio * max(abs(open_price), abs(close_price))
                range_size = max(range_size, 0.01 * open_price)
                
                if close_price > open_price:
                    high_price = max(open_price, close_price) + range_size * 0.3
                    low_price = min(open_price, close_price) - range_size * 0.1
                else:
                    high_price = max(open_price, close_price) + range_size * 0.1
                    low_price = min(open_price, close_price) - range_size * 0.3
                
                high_price = max(high_price, max(open_price, close_price))
                low_price = min(low_price, min(open_price, close_price))
                
                volume = max(current_volume * volume_ratio, 1000)
                
                current_time = pd.to_datetime('now') + timedelta(minutes=15 * (i + 1))
                
                klines.append({
                    'time': current_time,
                    'open': float(open_price),
                    'high': float(high_price),
                    'low': float(low_price),
                    'close': float(close_price),
                    'volume': float(volume),
                    'type': 'predicted',
                    'step': i + 1
                })
                
                current_close = close_price
                current_volume = volume
        
        return klines
    
    def predict_forward(
        self,
        df_normalized_full: np.ndarray,
        df_original: pd.DataFrame,
        seq_length: int = 100,
        pred_steps: int = 15
    ) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
        """
        Make rolling predictions - predicts next 15 bars based on latest 100 bars.
        This ensures the baseline (last 100 bars) remains fixed for prediction.
        
        Args:
            df_normalized_full: Full normalized dataset
            df_original: Original OHLCV dataframe
            seq_length: Historical window (100)
            pred_steps: Prediction horizon (15)
            
        Returns:
            (predicted_klines, predictions_normalized, predictions_denormalized)
        """
        # Use the most recent seq_length bars as input
        X_latest = df_normalized_full[-seq_length:]
        
        logger.info(f"Making prediction from bar index {len(df_original) - seq_length} to {len(df_original)}")
        
        # Predict next 15 bars
        pred_norm = self.predict_single_step(X_latest)
        pred_denorm = self.denormalize_prediction(pred_norm)
        
        # Generate K-lines
        klines = self.generate_klines_from_prediction(
            pred_denorm,
            self.feature_columns
        )
        
        return klines, pred_norm, pred_denorm
    
    def predict_with_fixed_baseline(
        self,
        X_baseline: np.ndarray,
        df_original: pd.DataFrame,
        pred_steps: int = 15,
        volatility_multiplier: float = 2.0
    ) -> List[Dict]:
        """
        Predict using a fixed baseline (last 100 bars).
        This is used for consistent predictions across time.
        
        Args:
            X_baseline: (100, num_features) fixed baseline
            df_original: Original OHLCV dataframe
            pred_steps: Prediction horizon
            volatility_multiplier: Multiplier to enhance visible volatility
            
        Returns:
            List of predicted klines
        """
        pred_norm = self.predict_single_step(X_baseline)
        pred_denorm = self.denormalize_prediction(pred_norm)
        
        klines = self.generate_klines_from_prediction(
            pred_denorm,
            self.feature_columns,
            volatility_multiplier=volatility_multiplier
        )
        
        return klines


class MultiStepPredictor(RollingPredictor):
    """
    Advanced predictor that can generate predictions multiple steps ahead
    by recursively applying the model.
    
    WARNING: This can accumulate errors. Use only for research.
    """
    
    def predict_recursive(
        self,
        X_initial: np.ndarray,
        num_steps: int,
        step_size: int = 15
    ) -> List[np.ndarray]:
        """
        Make predictions multiple steps ahead by recursive application.
        
        Args:
            X_initial: (100, num_features) initial sequence
            num_steps: How many steps of 15-bar predictions to make
            step_size: Size of each prediction (15)
            
        Returns:
            List of predictions, each of shape (15, num_features)
        """
        predictions = []
        X_current = X_initial.copy()
        
        for step in range(num_steps):
            # Predict next 15 bars
            pred_norm = self.predict_single_step(X_current)
            predictions.append(pred_norm)
            
            # Roll the window: remove oldest 15 bars, add new 15 predictions
            X_current = np.vstack([X_current[step_size:], pred_norm])
            
            logger.info(f"Recursive step {step + 1}/{num_steps} completed")
        
        return predictions
