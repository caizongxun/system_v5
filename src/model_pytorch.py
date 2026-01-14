import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class LSTMModel(nn.Module):
    """
    LSTM model for multi-step time series prediction.
    Uses Huber loss to reduce over-smoothing in regression tasks.
    """
    
    def __init__(
        self,
        sequence_length: int,
        num_features: int,
        prediction_length: int,
        lstm_units = 128,
        dropout_rate: float = 0.0,
        learning_rate: float = 0.001,
        l2_reg: float = 0.0,
        device: str = 'cuda',
        use_huber_loss: bool = True,
        huber_delta: float = 0.5
    ):
        super(LSTMModel, self).__init__()
        
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.prediction_length = prediction_length
        self.device = device
        self.use_huber_loss = use_huber_loss
        self.huber_delta = huber_delta
        
        # Handle lstm_units as either int or list
        if isinstance(lstm_units, list):
            lstm_units_list = lstm_units
        else:
            lstm_units_list = [lstm_units, lstm_units // 2]
        
        lstm_units_1 = lstm_units_list[0] if len(lstm_units_list) > 0 else 128
        lstm_units_2 = lstm_units_list[1] if len(lstm_units_list) > 1 else lstm_units_1 // 2
        
        # LSTM layers with stacking for better feature learning
        self.lstm1 = nn.LSTM(
            input_size=num_features,
            hidden_size=lstm_units_1,
            batch_first=True,
            dropout=dropout_rate if dropout_rate > 0 else 0
        )
        
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.lstm2 = nn.LSTM(
            input_size=lstm_units_1,
            hidden_size=lstm_units_2,
            batch_first=True,
            dropout=dropout_rate if dropout_rate > 0 else 0
        )
        
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Dense layers for output prediction
        self.fc1 = nn.Linear(lstm_units_2, lstm_units_2)
        self.fc2 = nn.Linear(lstm_units_2, num_features * prediction_length)
        
        self.relu = nn.ReLU()
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # Loss function: Huber instead of MSE to preserve variance
        if use_huber_loss:
            self.criterion = nn.HuberLoss(delta=huber_delta, reduction='mean')
            logger.info(f"Using Huber loss with delta={huber_delta}")
        else:
            self.criterion = nn.MSELoss()
            logger.info("Using MSE loss")
        
        self.to(device)
        
        logger.info(
            f"LSTM Model initialized: seq_length={sequence_length}, "
            f"num_features={num_features}, pred_length={prediction_length}, "
            f"lstm_units={lstm_units_1},{lstm_units_2}"
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (batch_size, sequence_length, num_features)
            
        Returns:
            (batch_size, prediction_length, num_features)
        """
        # LSTM layer 1
        lstm_out1, _ = self.lstm1(x)  # (batch, seq_len, lstm_units)
        lstm_out1 = self.dropout1(lstm_out1)
        
        # LSTM layer 2
        lstm_out2, _ = self.lstm2(lstm_out1)  # (batch, seq_len, lstm_units//2)
        lstm_out2 = self.dropout2(lstm_out2)
        
        # Use last output for prediction
        last_output = lstm_out2[:, -1, :]  # (batch, lstm_units//2)
        
        # Dense layers
        fc_out = self.relu(self.fc1(last_output))
        output = self.fc2(fc_out)  # (batch, num_features * pred_length)
        
        # Reshape to (batch, pred_length, num_features)
        output = output.reshape(-1, self.prediction_length, self.num_features)
        
        return output
    
    def calculate_loss(self, y_pred, y_true):
        """
        Calculate loss with optional L2 regularization.
        
        Args:
            y_pred: Model predictions
            y_true: Ground truth targets
            
        Returns:
            Total loss (loss + L2 penalty)
        """
        # Primary loss (MSE or Huber)
        loss = self.criterion(y_pred, y_true)
        
        # L2 regularization
        if self.l2_reg > 0:
            l2_penalty = 0
            for param in self.parameters():
                l2_penalty += torch.norm(param)
            loss += self.l2_reg * l2_penalty
        
        return loss
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Average loss for the epoch
        """
        self.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            y_pred = self(X_batch)
            
            # Calculate loss
            loss = self.calculate_loss(y_pred, y_batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(self, test_loader):
        """
        Evaluate model on test data.
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            (average_loss, mae, rmse, r2_score)
        """
        self.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                y_pred = self(X_batch)
                loss = self.criterion(y_pred, y_batch)
                
                total_loss += loss.item()
                all_preds.append(y_pred.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
        
        # Concatenate all predictions and targets
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Calculate metrics
        avg_loss = total_loss / len(test_loader)
        
        # Flatten for metric calculation
        preds_flat = all_preds.flatten()
        targets_flat = all_targets.flatten()
        
        mae = np.mean(np.abs(preds_flat - targets_flat))
        rmse = np.sqrt(np.mean((preds_flat - targets_flat) ** 2))
        
        # R2 score
        ss_res = np.sum((targets_flat - preds_flat) ** 2)
        ss_tot = np.sum((targets_flat - np.mean(targets_flat)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return avg_loss, mae, rmse, r2
    
    def fit(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: bool = True
    ):
        """
        Train the model.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Print training progress
        """
        # Create data loaders
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        test_dataset = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32)
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_mae, val_rmse, val_r2 = self.evaluate(test_loader)
            
            if verbose and (epoch + 1) % 5 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_loss:.6f}, "
                    f"Val Loss: {val_loss:.6f}, "
                    f"MAE: {val_mae:.6f}, "
                    f"RMSE: {val_rmse:.6f}, "
                    f"R2_Score: {val_r2:.4f}"
                )
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        logger.info("Training complete")
    
    def save(self, path: str):
        """
        Save model weights.
        
        Args:
            path: Path to save model
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """
        Load model weights.
        
        Args:
            path: Path to load model from
        """
        self.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"Model loaded from {path}")
