import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, results_dir: str = "test/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = {}
    
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
        mape = mean_absolute_percentage_error(y_true_flat, y_pred_flat)
        
        self.metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }
        
        logger.info(f"Metrics calculated - RMSE: {rmse:.6f}, MAE: {mae:.6f}, MAPE: {mape:.6f}")
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
        Plot training history
        
        Args:
            history: Training history from model.fit()
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        fig.suptitle('Training History', fontsize=14)
        
        axes[0].plot(history.history['loss'], label='Train Loss')
        axes[0].plot(history.history['val_loss'], label='Val Loss')
        axes[0].set_title('Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(history.history['mae'], label='Train MAE')
        axes[1].plot(history.history['val_mae'], label='Val MAE')
        axes[1].set_title('MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        axes[1].grid(True)
        
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
            for metric_name, metric_value in self.metrics.items():
                f.write(f"{metric_name}: {metric_value:.6f}\n")
        
        logger.info(f"Metrics report saved to {save_path}")
    
    def print_metrics_summary(self):
        """
        Print metrics summary to console
        """
        if not self.metrics:
            logger.warning("No metrics available")
            return
        
        print("\n" + "=" * 50)
        print("Model Evaluation Metrics Summary")
        print("=" * 50)
        for metric_name, metric_value in self.metrics.items():
            print(f"{metric_name:15} : {metric_value:12.6f}")
        print("=" * 50 + "\n")
