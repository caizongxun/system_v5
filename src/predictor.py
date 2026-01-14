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
        Convert normalized predictions back to original scale using StandardScaler.
        
        StandardScaler formula:
        normalized = (X - mean) / std
        
        Inverse:
        X = normalized * std + mean
        
        Args:
            pred_normalized: (15, num_features) normalized predictions from model
            
        Returns:
            (15, num_features) denormalized predictions in original scale
        """
        if self.scaler is None:
            logger.warning("No scaler available, returning normalized predictions")
            return pred_normalized
        
        # Use scaler.inverse_transform() to properly denormalize
        # This uses the mean and std saved during training
        pred_denorm = self.scaler.inverse_transform(pred_normalized)
        
        logger.debug(f"Denormalized prediction shape: {pred_denorm.shape}")
        logger.debug(f"Denormalized returns range: [{pred_denorm[:, 0].min():.6f}, {pred_denorm[:, 0].max():.6f}]")
        
        return pred_denorm
    
    def generate_klines_from_prediction(
        self, 
        pred_denorm: np.ndarray,
        last_candle: Dict,
        feature_columns: List[str],
        volatility_multiplier: float = 2.0
    ) -> List[Dict]:
        """
        Convert predicted features to K-line OHLCV data with proper OHLC constraints.
        Uses denormalized returns directly for accurate price generation.
        
        Args:
            pred_denorm: (15, num_features) denormalized predictions
            last_candle: Previous candle's OHLCV data
            feature_columns: List of feature names
            volatility_multiplier: Multiplier to enhance visible volatility
            
        Returns:
            List of 15 predicted candles with valid OHLC relationships
        """
        klines = []
        current_time = last_candle['time'] + timedelta(minutes=15)
        current_close = last_candle['close']
        current_volume = last_candle['volume']
        
        # Get feature indices
        returns_idx = feature_columns.index('returns')
        high_low_ratio_idx = feature_columns.index('high_low_ratio')
        open_close_ratio_idx = feature_columns.index('open_close_ratio')
        volume_ratio_idx = feature_columns.index('Volume_Ratio')
        volatility_5_idx = feature_columns.index('volatility_5')
        
        logger.info(f"Generating {len(pred_denorm)} K-lines from denormalized predictions")
        logger.info(f"Last close price: ${current_close:.2f}")
        
        for i in range(len(pred_denorm)):
            pred_features = pred_denorm[i]
            
            # Extract denormalized features
            returns = pred_features[returns_idx]  # Now in original scale! (e.g., 0.001 = 0.1%)
            high_low_ratio = pred_features[high_low_ratio_idx]
            open_close_ratio = pred_features[open_close_ratio_idx]
            volume_ratio = pred_features[volume_ratio_idx]
            volatility = max(abs(pred_features[volatility_5_idx]), 0.001)
            
            # CRITICAL FIX: Returns is now real percentage (e.g., 0.001 = 0.1%), not Z-score
            # Clamp to realistic range: -2% to +2%
            returns_clamped = np.clip(returns, -0.02, 0.02)
            
            # Calculate base OHLC prices
            open_price = current_close
            close_price = open_price * (1 + returns_clamped)
            
            # Calculate intra-bar range
            intra_range = abs(high_low_ratio) * abs(close_price)
            volatility_range = abs(volatility) * abs(open_price) * volatility_multiplier
            total_range = max(intra_range, volatility_range)
            
            # Determine high and low
            if close_price >= open_price:
                high_price = max(open_price, close_price) + total_range * 0.5
                low_price = min(open_price, close_price) - total_range * 0.1
            else:
                high_price = max(open_price, close_price) + total_range * 0.1
                low_price = min(open_price, close_price) - total_range * 0.5
            
            # Enforce OHLC constraints
            high_price = max(high_price, max(open_price, close_price))
            low_price = min(low_price, min(open_price, close_price))
            
            if high_price <= low_price:
                high_price = low_price + abs(total_range)
            
            # Volume
            volume = max(current_volume * np.clip(volume_ratio, 0.3, 3.0), 100)
            
            # Create candle
            kline = {
                'time': current_time,
                'open': float(np.round(open_price, 2)),
                'high': float(np.round(high_price, 2)),
                'low': float(np.round(low_price, 2)),
                'close': float(np.round(close_price, 2)),
                'volume': float(np.round(volume, 0)),
                'type': 'predicted',
                'step': i + 1,
                'returns': float(returns_clamped)
            }
            
            klines.append(kline)
            
            logger.debug(f"Step {i+1}: O={open_price:.2f} H={high_price:.2f} L={low_price:.2f} C={close_price:.2f} Returns={returns_clamped:.4f}")
            
            # Update for next iteration
            current_time += timedelta(minutes=15)
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
        
        # Get last actual candle for reference
        last_candle = {
            'time': pd.to_datetime(df_original.iloc[-1]['open_time']),
            'open': float(df_original.iloc[-1]['open']),
            'high': float(df_original.iloc[-1]['high']),
            'low': float(df_original.iloc[-1]['low']),
            'close': float(df_original.iloc[-1]['close']),
            'volume': float(df_original.iloc[-1]['volume'])
        }
        
        # Generate K-lines
        klines = self.generate_klines_from_prediction(
            pred_denorm,
            last_candle,
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
        
        CRITICAL FIX: Now properly denormalizes all features including returns!
        
        Args:
            X_baseline: (100, num_features) fixed baseline (NORMALIZED)
            df_original: Original OHLCV dataframe
            pred_steps: Prediction horizon
            volatility_multiplier: Multiplier to enhance visible volatility
            
        Returns:
            List of predicted klines
        """
        logger.info(f"\n{'='*60}")
        logger.info("PREDICTION STEP")
        logger.info(f"Input baseline shape: {X_baseline.shape}")
        logger.info(f"Last historical price: ${df_original.iloc[-1]['close']:.2f}")
        
        # Get normalized prediction from model
        pred_norm = self.predict_single_step(X_baseline)
        logger.info(f"Model output (normalized) shape: {pred_norm.shape}")
        logger.info(f"Normalized returns range: [{pred_norm[:, 0].min():.6f}, {pred_norm[:, 0].max():.6f}]")
        
        # CRITICAL: Denormalize to get real feature values
        pred_denorm = self.denormalize_prediction(pred_norm)
        logger.info(f"After denormalization:")
        logger.info(f"  Returns range: [{pred_denorm[:, 0].min():.6f}, {pred_denorm[:, 0].max():.6f}]")
        logger.info(f"  Avg return: {pred_denorm[:, 0].mean():.6f}")
        
        last_candle = {
            'time': pd.to_datetime(df_original.iloc[-1]['open_time']),
            'open': float(df_original.iloc[-1]['open']),
            'high': float(df_original.iloc[-1]['high']),
            'low': float(df_original.iloc[-1]['low']),
            'close': float(df_original.iloc[-1]['close']),
            'volume': float(df_original.iloc[-1]['volume'])
        }
        
        klines = self.generate_klines_from_prediction(
            pred_denorm,
            last_candle,
            self.feature_columns,
            volatility_multiplier=volatility_multiplier
        )
        
        # Log prediction summary
        if klines:
            closes = [k['close'] for k in klines]
            returns_gen = [(closes[i] - closes[i-1]) / closes[i-1] if i > 0 else 0 for i in range(len(closes))]
            logger.info(f"\nGenerated K-lines:")
            logger.info(f"  Price range: ${min(closes):.2f} - ${max(closes):.2f}")
            logger.info(f"  Avg predicted return: {np.mean(returns_gen):.4f}")
            logger.info(f"  Returns std: {np.std(returns_gen):.4f}")
            logger.info(f"{'='*60}\n")
        
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
