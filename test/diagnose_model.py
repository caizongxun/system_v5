import sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import torch
from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader
import gc
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, load_config, create_directories, print_section
from src.data_loader import DataLoader
from src.data_processor import DataProcessor
from src.model_pytorch import LSTMModel

logger = None

def predict_in_batches(model, X, batch_size=4, device='cuda'):
    predictions = []
    total_samples = len(X)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    for i in range(0, total_samples, batch_size):
        batch_end = min(i + batch_size, total_samples)
        batch = X_tensor[i:batch_end].to(device)
        
        with torch.no_grad():
            batch_pred = model(batch)
        predictions.append(batch_pred.cpu().numpy())
        
        del batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return np.concatenate(predictions, axis=0)

def main():
    global logger
    
    config = load_config()
    logger = setup_logging(config['paths']['logs_dir'])
    create_directories(config)
    
    print_section("Model Diagnosis: Overfitting Check")
    
    try:
        # Initialize device
        cuda_available = torch.cuda.is_available()
        device = torch.device('cuda' if cuda_available else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Load and preprocess data
        print_section("Data Loading")
        loader = DataLoader(
            repo_id=config['data']['repo_id'],
            cache_dir=Path(config['paths']['data_dir'])
        )
        
        df = loader.load_klines(
            symbol=config['data']['symbol'],
            timeframe=config['data']['timeframe']
        )
        
        processor = DataProcessor()
        if config['features']['use_technical_indicators']:
            df = processor.add_technical_indicators(df)
        
        feature_columns = ['open', 'high', 'low', 'close', 'volume']
        if config['features']['use_technical_indicators']:
            feature_columns.extend(['SMA_10', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal'])
        
        normalized_data, scaler = processor.normalize_features(df, feature_columns)
        logger.info(f"Normalized data shape: {normalized_data.shape}")
        logger.info(f"Normalized data statistics:")
        logger.info(f"  Min: {normalized_data.min(axis=0).min():.4f}")
        logger.info(f"  Max: {normalized_data.max(axis=0).max():.4f}")
        logger.info(f"  Mean: {normalized_data.mean(axis=0).mean():.4f}")
        logger.info(f"  Std: {normalized_data.std(axis=0).mean():.4f}")
        
        # Create sequences
        seq_length = config['model']['sequence_length']
        pred_length = config['model']['prediction_length']
        X, y = processor.create_sequences(normalized_data, seq_length, pred_length)
        
        logger.info(f"\nSequence data statistics:")
        logger.info(f"  X shape: {X.shape}")
        logger.info(f"  y shape: {y.shape}")
        logger.info(f"  X min: {X.min():.4f}, max: {X.max():.4f}")
        logger.info(f"  y min: {y.min():.4f}, max: {y.max():.4f}")
        
        # Split data
        test_split = config['model']['test_split']
        val_split = config['model']['validation_split']
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_split, shuffle=False
        )
        
        val_size = int(len(X_temp) * val_split / (1 - test_split))
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, shuffle=False
        )
        
        logger.info(f"\nData distribution:")
        logger.info(f"  Train - y min: {y_train.min():.4f}, max: {y_train.max():.4f}, mean: {y_train.mean():.4f}")
        logger.info(f"  Val   - y min: {y_val.min():.4f}, max: {y_val.max():.4f}, mean: {y_val.mean():.4f}")
        logger.info(f"  Test  - y min: {y_test.min():.4f}, max: {y_test.max():.4f}, mean: {y_test.mean():.4f}")
        
        # Load model
        print_section("Model Predictions")
        model = LSTMModel(
            sequence_length=config['model']['sequence_length'],
            num_features=len(feature_columns),
            prediction_length=config['model']['prediction_length'],
            lstm_units=config['model']['lstm_units'],
            dropout_rate=config['model']['dropout_rate'],
            learning_rate=config['model']['learning_rate'],
            l2_reg=config['model'].get('l2_regularization', 0.0),
            device=device
        )
        
        model_path = Path(config['paths']['model_dir']) / "btc_15m_model_pytorch.pt"
        model.load(str(model_path))
        
        # Make predictions on all three sets
        logger.info("Making predictions on training set...")
        y_train_pred = predict_in_batches(model, X_train, batch_size=4, device=device)
        
        logger.info("Making predictions on validation set...")
        y_val_pred = predict_in_batches(model, X_val, batch_size=4, device=device)
        
        logger.info("Making predictions on test set...")
        y_test_pred = predict_in_batches(model, X_test, batch_size=4, device=device)
        
        # Calculate metrics for each set
        print_section("Overfitting Analysis")
        
        def calc_metrics(y_true, y_pred, set_name):
            mse = np.mean((y_true - y_pred) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_true - y_pred))
            
            # R² score
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - y_true.mean()) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            logger.info(f"\n{set_name} Set:")
            logger.info(f"  RMSE: {rmse:.6f}")
            logger.info(f"  MAE:  {mae:.6f}")
            logger.info(f"  R²:   {r2:.6f}")
            logger.info(f"  Pred min: {y_pred.min():.4f}, max: {y_pred.max():.4f}, mean: {y_pred.mean():.4f}")
            
            return rmse, mae, r2
        
        train_rmse, train_mae, train_r2 = calc_metrics(y_train, y_train_pred, "Training")
        val_rmse, val_mae, val_r2 = calc_metrics(y_val, y_val_pred, "Validation")
        test_rmse, test_mae, test_r2 = calc_metrics(y_test, y_test_pred, "Test")
        
        # Overfitting diagnosis
        print_section("Diagnosis")
        logger.info(f"\nOverfitting Gap (Test - Train RMSE): {test_rmse - train_rmse:.6f}")
        logger.info(f"Overfitting Gap (Test - Train R²): {test_r2 - train_r2:.6f}")
        
        if train_r2 > 0.5 and test_r2 < 0:
            logger.warning("\n⚠️ SEVERE OVERFITTING DETECTED!")
            logger.warning("   Model learns training data perfectly but fails on test data")
            logger.warning("   Possible causes:")
            logger.warning("   1. Data leakage or sequence overlap")
            logger.warning("   2. Non-stationary data (distribution shift)")
            logger.warning("   3. Insufficient regularization")
        elif test_r2 < 0:
            logger.warning("\n❌ Model performs worse than predicting average")
            logger.warning("   Recommendations:")
            logger.warning("   1. Check data preprocessing (normalization issues)")
            logger.warning("   2. Use different features or feature engineering")
            logger.warning("   3. Try simpler model or different architecture")
        
        # Plot comparison
        print_section("Saving Diagnostic Plots")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: RMSE comparison
        sets = ['Train', 'Val', 'Test']
        rmses = [train_rmse, val_rmse, test_rmse]
        axes[0, 0].bar(sets, rmses, color=['green', 'orange', 'red'])
        axes[0, 0].set_title('RMSE by Data Set')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].grid(axis='y')
        
        # Plot 2: R² comparison
        r2s = [train_r2, val_r2, test_r2]
        axes[0, 1].bar(sets, r2s, color=['green', 'orange', 'red'])
        axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[0, 1].set_title('R² by Data Set')
        axes[0, 1].set_ylabel('R²')
        axes[0, 1].grid(axis='y')
        
        # Plot 3: Test set prediction vs actual (first 500 samples)
        sample_size = min(500, len(y_test))
        x_axis = np.arange(sample_size)
        axes[1, 0].plot(x_axis, y_test[:sample_size, 0, 0], label='Actual', alpha=0.7)
        axes[1, 0].plot(x_axis, y_test_pred[:sample_size, 0, 0], label='Predicted', alpha=0.7)
        axes[1, 0].set_title('Test Set: Actual vs Predicted (First 500 Samples)')
        axes[1, 0].set_ylabel('Close Price (Normalized)')
        axes[1, 0].legend()
        axes[1, 0].grid()
        
        # Plot 4: Residuals
        residuals = y_test[:sample_size, 0, 0] - y_test_pred[:sample_size, 0, 0]
        axes[1, 1].scatter(y_test_pred[:sample_size, 0, 0], residuals, alpha=0.5)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_title('Residual Plot (Test Set)')
        axes[1, 1].set_xlabel('Predicted Value')
        axes[1, 1].set_ylabel('Residual')
        axes[1, 1].grid()
        
        plt.tight_layout()
        plot_path = Path(config['paths']['results_dir']) / 'diagnosis_plot.png'
        plt.savefig(plot_path, dpi=150)
        logger.info(f"Diagnostic plot saved to {plot_path}")
        
        logger.info("\n✅ Diagnosis complete!")
        
    except Exception as e:
        logger.error(f"Diagnosis failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
