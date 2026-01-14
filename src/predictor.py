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
        
        pred_denorm = self.scaler.inverse_transform(pred_normalized)
        
        logger.debug(f"Denormalized prediction shape: {pred_denorm.shape}")
        
        return pred_denorm
    
    def calculate_market_volatility(self, df_original: pd.DataFrame, periods: int = 20) -> float:
        """
        Calculate recent market volatility to determine realistic price movement.
        
        Args:
            df_original: Original OHLCV dataframe
            periods: Number of periods to calculate volatility over
            
        Returns:
            Volatility as a percentage (e.g., 0.015 for 1.5%)
        """
        if len(df_original) < periods:
            periods = len(df_original)
        
        recent_returns = df_original.iloc[-periods:]['close'].pct_change()
        volatility = recent_returns.std()
        
        logger.info(f"Market volatility (last {periods} bars): {volatility:.6f} ({volatility*100:.4f}%)")
        
        return max(volatility, 0.0001)
    
    def generate_klines_from_prediction(
        self, 
        pred_denorm: np.ndarray,
        last_candle: Dict,
        feature_columns: List[str],
        volatility_multiplier: float = 2.0,
        market_volatility: float = None
    ) -> List[Dict]:
        """
        Convert predicted features to K-line OHLCV data.
        Uses model's predicted returns scaled by market volatility to determine realistic prices.
        
        Args:
            pred_denorm: (15, num_features) denormalized predictions
            last_candle: Previous candle's OHLCV data
            feature_columns: List of feature names
            volatility_multiplier: Multiplier to enhance visible volatility
            market_volatility: Recent market volatility to scale predictions
            
        Returns:
            List of 15 predicted candles with valid OHLC relationships
        """
        klines = []
        current_time = last_candle['time'] + timedelta(minutes=15)
        current_close = last_candle['close']
        current_volume = last_candle['volume']
        
        returns_idx = feature_columns.index('returns')
        high_low_ratio_idx = feature_columns.index('high_low_ratio')
        open_close_ratio_idx = feature_columns.index('open_close_ratio')
        volume_ratio_idx = feature_columns.index('Volume_Ratio')
        volatility_5_idx = feature_columns.index('volatility_5')
        
        if market_volatility is None:
            market_volatility = 0.001
        
        logger.info(f"Generating {len(pred_denorm)} K-lines from predictions")
        logger.info(f"Last close price: ${current_close:.2f}")
        logger.info(f"Market volatility baseline: {market_volatility:.6f}")
        
        for i in range(len(pred_denorm)):
            pred_features = pred_denorm[i]
            
            returns = pred_features[returns_idx]
            high_low_ratio = pred_features[high_low_ratio_idx]
            open_close_ratio = pred_features[open_close_ratio_idx]
            volume_ratio = pred_features[volume_ratio_idx]
            volatility = max(abs(pred_features[volatility_5_idx]), 0.001)
            
            # Scale returns by market volatility to get realistic price movements
            scaled_returns = returns / (market_volatility + 1e-8) if market_volatility > 0 else returns
            # Cap the scaling to avoid extreme moves
            scaled_returns = np.clip(scaled_returns * market_volatility * volatility_multiplier, -0.05, 0.05)
            
            open_price = current_close
            close_price = open_price * (1 + scaled_returns)
            
            # Use actual volatility levels from predictions
            intra_range = abs(high_low_ratio) * abs(close_price) * volatility_multiplier
            volatility_range = abs(volatility) * abs(open_price) * volatility_multiplier * 10
            total_range = max(intra_range, volatility_range, abs(close_price - open_price) * 0.5)
            
            if close_price >= open_price:
                high_price = max(open_price, close_price) + total_range * 0.3
                low_price = min(open_price, close_price) - total_range * 0.1
            else:
                high_price = max(open_price, close_price) + total_range * 0.1
                low_price = min(open_price, close_price) - total_range * 0.3
            
            high_price = max(high_price, max(open_price, close_price))
            low_price = min(low_price, min(open_price, close_price))
            
            if high_price <= low_price:
                high_price = low_price + max(1, abs(total_range))
            
            volume = max(current_volume * np.clip(volume_ratio, 0.3, 3.0), 100)
            
            kline = {
                'time': current_time,
                'open': float(np.round(open_price, 2)),
                'high': float(np.round(high_price, 2)),
                'low': float(np.round(low_price, 2)),
                'close': float(np.round(close_price, 2)),
                'volume': float(np.round(volume, 0)),
                'type': 'predicted',
                'step': i + 1,
                'returns': float(returns),
                'scaled_returns': float(scaled_returns)
            }
            
            klines.append(kline)
            
            logger.debug(f"Step {i+1}: Raw returns={returns:.8f}, Scaled={scaled_returns:.6f}, O={open_price:.2f} C={close_price:.2f}")
            
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
        X_latest = df_normalized_full[-seq_length:]
        
        logger.info(f"Making prediction from bar index {len(df_original) - seq_length} to {len(df_original)}")
        
        pred_norm = self.predict_single_step(X_latest)
        pred_denorm = self.denormalize_prediction(pred_norm)
        
        last_candle = {
            'time': pd.to_datetime(df_original.iloc[-1]['open_time']),
            'open': float(df_original.iloc[-1]['open']),
            'high': float(df_original.iloc[-1]['high']),
            'low': float(df_original.iloc[-1]['low']),
            'close': float(df_original.iloc[-1]['close']),
            'volume': float(df_original.iloc[-1]['volume'])
        }
        
        market_vol = self.calculate_market_volatility(df_original)
        
        klines = self.generate_klines_from_prediction(
            pred_denorm,
            last_candle,
            self.feature_columns,
            market_volatility=market_vol
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
        
        Scales model predictions by actual market volatility to generate realistic prices.
        
        Args:
            X_baseline: (100, num_features) fixed baseline (NORMALIZED)
            df_original: Original OHLCV dataframe
            pred_steps: Prediction horizon
            volatility_multiplier: Multiplier to enhance visible volatility
            
        Returns:
            List of predicted klines
        """
        logger.info(f"{'='*60}")
        logger.info("PREDICTION STEP")
        logger.info(f"Input baseline shape: {X_baseline.shape}")
        logger.info(f"Last historical price: ${df_original.iloc[-1]['close']:.2f}")
        
        pred_norm = self.predict_single_step(X_baseline)
        logger.info(f"Model output (normalized) shape: {pred_norm.shape}")
        logger.info(f"Normalized returns - min: {pred_norm[:, 0].min():.6f}, max: {pred_norm[:, 0].max():.6f}, mean: {pred_norm[:, 0].mean():.6f}")
        
        pred_denorm = self.denormalize_prediction(pred_norm)
        logger.info(f"After denormalization (real scale):")
        logger.info(f"  Returns - min: {pred_denorm[:, 0].min():.6f}, max: {pred_denorm[:, 0].max():.6f}, mean: {pred_denorm[:, 0].mean():.6f}")
        logger.info(f"  high_low_ratio - min: {pred_denorm[:, 1].min():.6f}, max: {pred_denorm[:, 1].max():.6f}")
        logger.info(f"  Volume_Ratio - min: {pred_denorm[:, 13].min():.6f}, max: {pred_denorm[:, 13].max():.6f}")
        
        # Calculate market volatility to scale predictions
        market_vol = self.calculate_market_volatility(df_original)
        
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
            volatility_multiplier=volatility_multiplier,
            market_volatility=market_vol
        )
        
        if klines:
            closes = [k['close'] for k in klines]
            highs = [k['high'] for k in klines]
            lows = [k['low'] for k in klines]
            logger.info(f"Generated K-lines summary:")
            logger.info(f"  Price range: ${min(lows):.2f} - ${max(highs):.2f}")
            logger.info(f"  Close range: ${min(closes):.2f} - ${max(closes):.2f}")
            logger.info(f"  First: O={klines[0]['open']:.2f} C={klines[0]['close']:.2f}")
            logger.info(f"  Last: O={klines[-1]['open']:.2f} C={klines[-1]['close']:.2f}")
            logger.info(f"  Price change: ${klines[0]['close']:.2f} -> ${klines[-1]['close']:.2f} ({(klines[-1]['close']/klines[0]['close']-1)*100:.3f}%)")
            logger.info(f"{'='*60}")
        
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
            pred_norm = self.predict_single_step(X_current)
            predictions.append(pred_norm)
            
            X_current = np.vstack([X_current[step_size:], pred_norm])
            
            logger.info(f"Recursive step {step + 1}/{num_steps} completed")
        
        return predictions
