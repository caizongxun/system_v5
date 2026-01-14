import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class LSTMModel:
    def __init__(self, sequence_length: int, num_features: int, prediction_length: int, 
                 lstm_units: list, dropout_rate: float = 0.2, learning_rate: float = 0.001):
        """
        Initialize LSTM model
        
        Args:
            sequence_length: Input sequence length (100)
            num_features: Number of features
            prediction_length: Prediction sequence length (15)
            lstm_units: List of LSTM units [128, 64]
            dropout_rate: Dropout rate
            learning_rate: Learning rate
        """
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.prediction_length = prediction_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
    
    def build(self):
        """
        Build LSTM model architecture
        """
        self.model = Sequential([
            LSTM(self.lstm_units[0], return_sequences=True, 
                 input_shape=(self.sequence_length, self.num_features)),
            BatchNormalization(),
            Dropout(self.dropout_rate),
            
            LSTM(self.lstm_units[1], return_sequences=False),
            BatchNormalization(),
            Dropout(self.dropout_rate),
            
            Dense(64, activation='relu'),
            Dropout(self.dropout_rate),
            
            Dense(self.prediction_length * self.num_features)
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        logger.info("Model built successfully")
        self.model.summary()
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 50, batch_size: int = 32):
        """
        Train the model
        
        Args:
            X_train: Training input sequences
            y_train: Training target sequences
            X_val: Validation input sequences
            y_val: Validation target sequences
            epochs: Number of epochs
            batch_size: Batch size
        """
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        logger.info(f"Starting training for {epochs} epochs...")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        logger.info("Training completed")
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Input sequences
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build() first.")
        
        predictions = self.model.predict(X)
        return predictions.reshape(predictions.shape[0], self.prediction_length, self.num_features)
    
    def save(self, model_path: str):
        """
        Save model to disk
        
        Args:
            model_path: Path to save model
        """
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load(self, model_path: str):
        """
        Load model from disk
        
        Args:
            model_path: Path to load model
        """
        self.model = keras.models.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
    
    def get_summary(self):
        """
        Get model summary
        """
        if self.model is None:
            return "Model not built yet"
        return self.model.summary()
