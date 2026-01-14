import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class VolatilityModel(nn.Module):
    """
    Specialized model for predicting volatility magnitude (absolute price movement).
    Separate from direction prediction (returns).
    
    This model learns to predict how much the price will move,
    independent of direction.
    """
    
    def __init__(
        self,
        sequence_length: int = 100,
        num_features: int = 16,
        prediction_length: int = 15,
        lstm_units: int = 64,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        l2_reg: float = 0.0,
        device: str = 'cuda'
    ):
        """
        Args:
            sequence_length: Historical window size (100 bars)
            num_features: Number of input features
            prediction_length: How many bars ahead to predict (15)
            lstm_units: LSTM hidden units
            dropout_rate: Dropout rate
            learning_rate: Learning rate
            l2_reg: L2 regularization coefficient
            device: 'cuda' or 'cpu'
        """
        super(VolatilityModel, self).__init__()
        
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.prediction_length = prediction_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.device = device
        
        # LSTM layer to process historical sequence
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=lstm_units,
            num_layers=2,
            dropout=dropout_rate if dropout_rate > 0 else 0,
            batch_first=True
        )
        
        # Dense layers for volatility magnitude prediction
        self.fc1 = nn.Linear(lstm_units, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Output layer: predict volatility (1 value per bar, 15 bars)
        # Output is always positive (magnitude)
        self.output = nn.Linear(64, prediction_length)
        self.relu_out = nn.ReLU()  # Ensure output is positive
        
        self.to(device)
        logger.info(f"VolatilityModel initialized on {device}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch_size, sequence_length, num_features)
            
        Returns:
            (batch_size, prediction_length) - predicted volatility magnitudes
        """
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use last hidden state
        lstm_last = h_n[-1]  # (batch_size, lstm_units)
        
        # Dense layers
        x = self.fc1(lstm_last)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        # Output volatility (positive)
        volatility = self.output(x)
        volatility = self.relu_out(volatility)  # Ensure positive
        
        return volatility
    
    def save(self, path: str):
        """Save model to file."""
        torch.save(self.state_dict(), path)
        logger.info(f"VolatilityModel saved to {path}")
    
    def load(self, path: str):
        """Load model from file."""
        self.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"VolatilityModel loaded from {path}")


class DualModelPredictor:
    """
    Combines direction prediction (LSTM for returns) with
    magnitude prediction (VolatilityModel).
    
    This separation improves prediction quality:
    - Direction model learns directional patterns
    - Volatility model learns magnitude patterns
    - Combined result respects both signal and magnitude
    """
    
    def __init__(self, direction_model, volatility_model, scaler, feature_columns, device='cuda'):
        """
        Args:
            direction_model: Main LSTM model predicting returns direction
            volatility_model: VolatilityModel predicting magnitude
            scaler: StandardScaler for feature normalization
            feature_columns: List of feature names
            device: 'cuda' or 'cpu'
        """
        self.direction_model = direction_model
        self.volatility_model = volatility_model
        self.scaler = scaler
        self.feature_columns = feature_columns
        self.device = device
    
    def predict_single_step(self, X_normalized: np.ndarray) -> tuple:
        """
        Make predictions for both direction and volatility.
        
        Args:
            X_normalized: (100, num_features) normalized sequence
            
        Returns:
            (direction_pred, volatility_pred) where:
            - direction_pred: (15, num_features) from direction model
            - volatility_pred: (15,) volatility magnitudes
        """
        X_tensor = torch.tensor(
            X_normalized.reshape(1, -1, len(self.feature_columns)),
            dtype=torch.float32
        ).to(self.device)
        
        with torch.no_grad():
            direction_pred = self.direction_model(X_tensor).cpu().numpy()[0]
            volatility_pred = self.volatility_model(X_tensor).cpu().numpy()[0]
        
        logger.info(f"Direction prediction shape: {direction_pred.shape}")
        logger.info(f"Volatility prediction shape: {volatility_pred.shape}")
        logger.info(f"Volatility range: {volatility_pred.min():.6f} to {volatility_pred.max():.6f}")
        
        return direction_pred, volatility_pred
    
    def combine_predictions(self, direction_denorm: np.ndarray, volatility_pred: np.ndarray) -> np.ndarray:
        """
        Combine direction and volatility predictions.
        
        Strategy:
        1. Get return sign from direction model (positive/negative)
        2. Get magnitude from volatility model
        3. Combine: final_return = sign(direction_return) * volatility_magnitude
        
        Args:
            direction_denorm: (15, num_features) denormalized direction predictions
            volatility_pred: (15,) volatility magnitudes
            
        Returns:
            (15, num_features) combined predictions
        """
        combined = direction_denorm.copy()
        returns_idx = self.feature_columns.index('returns')
        
        for i in range(len(direction_denorm)):
            # Get return sign from direction model
            direction_return = direction_denorm[i, returns_idx]
            sign = np.sign(direction_return) if direction_return != 0 else 1
            
            # Get magnitude from volatility model
            magnitude = volatility_pred[i]
            
            # Combine: apply direction sign to volatility magnitude
            combined[i, returns_idx] = sign * magnitude
            
            logger.debug(f"Step {i}: direction={direction_return:.6f}, volatility={magnitude:.6f}, combined={sign * magnitude:.6f}")
        
        return combined

