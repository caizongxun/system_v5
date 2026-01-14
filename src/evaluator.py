import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Dict, Tuple
import torch

logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, results_dir: str = "test/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = {}
    
    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
        """
        Calculate MAPE with epsilon to avoid division by zero
        
        Args:
            y_true: True values
            y_pred: Predicted values
            epsilon: Small value to prevent division by zero
            
        Returns:
            MAPE value
        """
        y_true_flat = np.abs(y_true.reshape(-1))
        y_pred_flat = y_pred.reshape(-1)
        
        denominator = np.maximum(y_true_flat, epsilon)
        mape = np.mean(np.abs((y_true_flat - y_pred_flat) / denominator)) * 100
        
        return mape
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calculate evaluation metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with metrics
        """
        y_true_flat = y_true.reshape(-1)
        y_pred_flat = y_pred.reshape(-1)
        
        mse = mean_squared_error(y_true_flat, y_pred_flat)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_flat, y_pred_flat)
        mape = self._calculate_mape(y_true_flat, y_pred_flat)
        
        r_squared = 1 - (np.sum((y_true_flat - y_pred_flat) ** 2) / 
                        np.sum((y_true_flat - np.mean(y_true_flat)) ** 2))
        
        self.metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R_Squared': r_squared
        }
        
        logger.info(f"Metrics - RMSE: {rmse:.6f}, MAE: {mae:.6f}, MAPE: {mape:.2f}%, R2: {r_squared:.4f}")
        return self.metrics
    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        sample_idx: int = 0, save_path: str = None):
        """
        Plot true vs predicted values for a sample
        
        Args:
            y_true: True values
            y_pred: Predicted values
            sample_idx: Sample index to plot
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Predictions vs True Values - Sample {sample_idx}', fontsize=16)
        
        time_steps = range(y_true.shape[1])
        
        sample_true = y_true[sample_idx]
        sample_pred = y_pred[sample_idx]
        
        axes[0, 0].plot(time_steps, sample_true[:, 0], label='True Close', marker='o')
        axes[0, 0].plot(time_steps, sample_pred[:, 0], label='Pred Close', marker='s')
        axes[0, 0].set_title('Close Price')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(time_steps, sample_true[:, 3], label='True Volume', marker='o')
        axes[0, 1].plot(time_steps, sample_pred[:, 3], label='Pred Volume', marker='s')
        axes[0, 1].set_title('Volume')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        axes[1, 0].plot(time_steps, sample_true[:, 1], label='True High', marker='o')
        axes[1, 0].plot(time_steps, sample_pred[:, 1], label='Pred High', marker='s')
        axes[1, 0].set_title('High Price')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(time_steps, sample_true[:, 2], label='True Low', marker='o')
        axes[1, 1].plot(time_steps, sample_pred[:, 2], label='Pred Low', marker='s')
        axes[1, 1].set_title('Low Price')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.results_dir / "predictions_sample.png"
        
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
        plt.close()
    
    def plot_training_history(self, history, save_path: str = None):
        """
        Plot training history (supports both PyTorch dict and Keras history)
        
        Args:
            history: Training history dict or Keras history object
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(1, 1, figsize=(14, 4))
        fig.suptitle('Training History', fontsize=14)
        
        # Support both PyTorch dict format and Keras history format
        if isinstance(history, dict):
            # PyTorch format
            train_loss = history.get('train_loss', [])
            val_loss = history.get('val_loss', [])
            
            if train_loss and val_loss:
                epochs = range(1, len(train_loss) + 1)
                axes.plot(epochs, train_loss, label='Train Loss', marker='o')
                axes.plot(epochs, val_loss, label='Val Loss', marker='s')
                axes.set_title('Loss (MSE)')
                axes.set_xlabel('Epoch')
                axes.set_ylabel('Loss')
                axes.legend()
                axes.grid(True)
        else:
            # Keras format (fallback)
            if hasattr(history, 'history'):
                axes.plot(history.history.get('loss', []), label='Train Loss')
                axes.plot(history.history.get('val_loss', []), label='Val Loss')
                axes.set_title('Loss (MSE)')
                axes.set_xlabel('Epoch')
                axes.set_ylabel('Loss')
                axes.legend()
                axes.grid(True)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.results_dir / "training_history.png"
        
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")
        plt.close()
    
    def save_metrics_report(self, save_path: str = None):
        """
        Save metrics report to file
        
        Args:
            save_path: Path to save report
        """
        if not self.metrics:
            logger.warning("No metrics available")
            return
        
        if save_path is None:
            save_path = self.results_dir / "metrics_report.txt"
        
        with open(save_path, 'w') as f:
            f.write("Model Evaluation Metrics\n")
            f.write("=" * 40 + "\n\n")
            f.write("Primary Metrics (MSE-based):\n")
            f.write(f"  MSE:  {self.metrics['MSE']:.8f}\n")
            f.write(f"  RMSE: {self.metrics['RMSE']:.8f}\n")
            f.write(f"  MAE:  {self.metrics['MAE']:.8f}\n")
            f.write(f"  R2:   {self.metrics['R_Squared']:.6f}\n")
            f.write(f"\nAdditional Metric:\n")
            f.write(f"  MAPE: {self.metrics['MAPE']:.2f}%\n")
            f.write(f"\nNote: MAPE is for reference only.\n")
            f.write(f"Focus on RMSE/MAE for model evaluation.\n")
        
        logger.info(f"Metrics report saved to {save_path}")
    
    def print_metrics_summary(self):
        """
        Print metrics summary to console
        """
        if not self.metrics:
            logger.warning("No metrics available")
            return
        
        print("\n" + "=" * 60)
        print("Model Evaluation Metrics Summary")
        print("=" * 60)
        print(f"{'MSE':<20} : {self.metrics['MSE']:>12.8f}")
        print(f"{'RMSE':<20} : {self.metrics['RMSE']:>12.8f}")
        print(f"{'MAE':<20} : {self.metrics['MAE']:>12.8f}")
        print(f"{'R-Squared':<20} : {self.metrics['R_Squared']:>12.6f}")
        print(f"{'MAPE (reference)':<20} : {self.metrics['MAPE']:>12.2f}%")
        print("\nNote: Use RMSE/MAE for model evaluation.")
        print("MAPE can be inflated due to small normalized values.")
        print("=" * 60 + "\n")
