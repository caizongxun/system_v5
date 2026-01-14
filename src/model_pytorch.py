import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class LSTMModel(nn.Module):
    def __init__(self, sequence_length: int, num_features: int, prediction_length: int,
                 lstm_units: list, dropout_rate: float = 0.2, learning_rate: float = 0.001,
                 l2_reg: float = 0.0, device=None):
        """
        Initialize LSTM model for PyTorch
        
        Args:
            sequence_length: Input sequence length (100)
            num_features: Number of features
            prediction_length: Prediction sequence length (15)
            lstm_units: List of LSTM units [256, 128, 64]
            dropout_rate: Dropout rate
            learning_rate: Learning rate
            l2_reg: L2 regularization coefficient (weight decay)
            device: torch device (cpu or cuda)
        """
        super(LSTMModel, self).__init__()
        
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.prediction_length = prediction_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.device = device if device is not None else torch.device('cpu')
        self.history = None
        
        # Build LSTM layers
        self.lstm_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # First LSTM layer
        self.lstm_layers.append(
            nn.LSTM(
                input_size=num_features,
                hidden_size=lstm_units[0],
                batch_first=True,
                dropout=dropout_rate if len(lstm_units) > 1 else 0
            )
        )
        self.batch_norms.append(nn.BatchNorm1d(lstm_units[0]))
        self.dropouts.append(nn.Dropout(dropout_rate))
        
        # Middle LSTM layers
        for i in range(1, len(lstm_units)):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=lstm_units[i-1],
                    hidden_size=lstm_units[i],
                    batch_first=True,
                    dropout=dropout_rate if i < len(lstm_units) - 1 else 0
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(lstm_units[i]))
            self.dropouts.append(nn.Dropout(dropout_rate))
        
        # Dense layers
        self.fc1 = nn.Linear(lstm_units[-1], 64)
        self.dropout_fc1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, prediction_length * num_features)
        
        # Move to device
        self.to(self.device)
        
        # Setup optimizer and loss
        self.optimizer = optim.Adam(
            self.parameters(),
            lr=learning_rate,
            weight_decay=l2_reg
        )
        self.loss_fn = nn.MSELoss()
        
        logger.info("PyTorch LSTM Model initialized")
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, num_features)
            
        Returns:
            Output tensor of shape (batch_size, prediction_length, num_features)
        """
        # Pass through LSTM layers
        for i, lstm_layer in enumerate(self.lstm_layers):
            x, _ = lstm_layer(x)
            # Apply batch norm only if sequence length > 1
            if x.size(1) > 1:
                # Reshape for batch norm: (batch, seq_len, hidden) -> (batch*seq_len, hidden)
                batch_size, seq_len, hidden_size = x.size()
                x_reshaped = x.contiguous().view(-1, hidden_size)
                x_reshaped = self.batch_norms[i](x_reshaped)
                x = x_reshaped.view(batch_size, seq_len, hidden_size)
            x = self.dropouts[i](x)
        
        # Take last output from LSTM sequence
        x = x[:, -1, :]  # (batch_size, lstm_units[-1])
        
        # Dense layers
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout_fc1(x)
        x = self.fc2(x)
        
        # Reshape output
        x = x.view(-1, self.prediction_length, self.num_features)
        
        return x
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch
        
        Args:
            train_loader: PyTorch DataLoader for training
            
        Returns:
            Average loss for the epoch
        """
        self.train()
        total_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self(batch_x)
            loss = self.loss_fn(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """
        Validate the model
        
        Args:
            val_loader: PyTorch DataLoader for validation
            
        Returns:
            Average loss for validation set
        """
        self.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self(batch_x)
                loss = self.loss_fn(outputs, batch_y)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, epochs: int = 50, device=None):
        """
        Train the model
        
        Args:
            train_loader: PyTorch DataLoader for training
            val_loader: PyTorch DataLoader for validation
            epochs: Number of epochs
            device: Device to use (cpu or cuda)
        """
        if device is not None:
            self.device = device
            self.to(self.device)
        
        self.history = {
            'train_loss': [],
            'val_loss': []
        }
        
        patience = 15
        patience_counter = 0
        best_val_loss = float('inf')
        
        logger.info(f"Starting training for {epochs} epochs...")
        logger.info(f"Training samples: {len(train_loader.dataset)}, Validation samples: {len(val_loader.dataset)}")
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            
            # Learning rate scheduling (manual)
            if patience_counter > 5 and patience_counter % 5 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.5
                    logger.info(f"Reduced learning rate to {param_group['lr']:.2e}")
        
        logger.info("Training completed")
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Input tensor of shape (batch_size, sequence_length, num_features)
               or numpy array
            
        Returns:
            Predictions as numpy array
        """
        self.eval()
        
        # Convert to tensor if needed
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        
        X = X.to(self.device)
        
        with torch.no_grad():
            predictions = self(X)
        
        return predictions.cpu().numpy()
    
    def save(self, model_path: str):
        """
        Save model to disk
        
        Args:
            model_path: Path to save model
        """
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load(self, model_path: str):
        """
        Load model from disk
        
        Args:
            model_path: Path to load model
        """
        self.load_state_dict(torch.load(model_path, map_location=self.device))
        self.to(self.device)
        logger.info(f"Model loaded from {model_path}")
    
    def get_summary(self):
        """
        Get model summary
        """
        summary_str = f"""
=== PyTorch LSTM Model Summary ===
Model Architecture:
  - Input Shape: ({self.sequence_length}, {self.num_features})
  - LSTM Units: {self.lstm_units}
  - Output Shape: ({self.prediction_length}, {self.num_features})
  - Dropout Rate: {self.dropout_rate}
  - Learning Rate: {self.learning_rate}
  - L2 Regularization: {self.l2_reg}
  - Device: {self.device}

Total Parameters: {self.count_parameters():,}
==================================
        """
        return summary_str
    
    def count_parameters(self):
        """
        Count total number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
