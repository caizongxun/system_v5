import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, setup_logging
from src.data_loader import DataLoader
from src.data_processor import DataProcessor
from src.model_pytorch import LSTMModel

def analyze_training_data():
    config = load_config()
    logger = setup_logging(config['paths']['logs_dir'])
    
    # Load data
    loader = DataLoader(
        repo_id=config['data']['repo_id'],
        cache_dir=Path(config['paths']['data_dir'])
    )
    df = loader.load_klines(
        symbol=config['data']['symbol'],
        timeframe=config['data']['timeframe']
    )
    
    processor = DataProcessor()
    df_processed = processor.add_technical_indicators(df)
    
    feature_columns = config['features'].get('selected_features', [])
    normalized_data, scaler = processor.normalize_features(df_processed, feature_columns)
    
    # Check returns distribution
    returns_idx = feature_columns.index('returns')
    all_returns = normalized_data[:, returns_idx]
    
    print("\n" + "="*60)
    print("NORMALIZED RETURNS STATISTICS")
    print("="*60)
    print(f"Min: {all_returns.min():.6f}")
    print(f"Max: {all_returns.max():.6f}")
    print(f"Mean: {all_returns.mean():.6f}")
    print(f"Std: {all_returns.std():.6f}")
    print(f"Median: {np.median(all_returns):.6f}")
    
    # Analyze sequences
    seq_length = config['model']['sequence_length']
    pred_length = config['model']['prediction_length']
    X, y = processor.create_sequences(normalized_data, seq_length, pred_length)
    
    print("\n" + "="*60)
    print("SEQUENCE ANALYSIS")
    print("="*60)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # Check target variability
    y_returns = y[:, :, returns_idx]
    print(f"\ny target (returns) statistics:")
    print(f"  Shape: {y_returns.shape}")
    print(f"  Min: {y_returns.min():.6f}")
    print(f"  Max: {y_returns.max():.6f}")
    print(f"  Mean: {y_returns.mean():.6f}")
    print(f"  Std: {y_returns.std():.6f}")
    
    # Check variability across time steps
    print(f"\nVariability by prediction step:")
    for i in range(pred_length):
        step_returns = y_returns[:, i]
        print(f"  Step {i+1}: mean={step_returns.mean():.6f}, std={step_returns.std():.6f}, range=[{step_returns.min():.6f}, {step_returns.max():.6f}]")
    
    # Check raw (denormalized) returns
    print("\n" + "="*60)
    print("DENORMALIZED ANALYSIS")
    print("="*60)
    
    y_sample = y[:1000]  # Sample
    y_denorm = scaler.inverse_transform(y_sample.reshape(-1, len(feature_columns)))
    y_denorm = y_denorm.reshape(y_sample.shape)
    y_denorm_returns = y_denorm[:, :, returns_idx]
    
    print(f"y denormalized (returns) statistics:")
    print(f"  Min: {y_denorm_returns.min():.6f}")
    print(f"  Max: {y_denorm_returns.max():.6f}")
    print(f"  Mean: {y_denorm_returns.mean():.6f}")
    print(f"  Std: {y_denorm_returns.std():.6f}")
    
    # Check model predictions
    print("\n" + "="*60)
    print("MODEL PREDICTION ANALYSIS")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    if model_path.exists():
        model.load(str(model_path))
        model.eval()
        
        # Get predictions on sample
        X_sample = torch.tensor(X[:100], dtype=torch.float32).to(device)
        with torch.no_grad():
            pred_norm = model(X_sample)
        pred_norm_np = pred_norm.cpu().numpy()
        pred_denorm = scaler.inverse_transform(pred_norm_np.reshape(-1, len(feature_columns)))
        pred_denorm = pred_denorm.reshape(pred_norm_np.shape)
        pred_returns = pred_denorm[:, :, returns_idx]
        
        print(f"Normalized predictions (returns):")
        print(f"  Min: {pred_norm_np[:, :, returns_idx].min():.6f}")
        print(f"  Max: {pred_norm_np[:, :, returns_idx].max():.6f}")
        print(f"  Mean: {pred_norm_np[:, :, returns_idx].mean():.6f}")
        print(f"  Std: {pred_norm_np[:, :, returns_idx].std():.6f}")
        
        print(f"\nDenormalized predictions (returns):")
        print(f"  Min: {pred_returns.min():.6f}")
        print(f"  Max: {pred_returns.max():.6f}")
        print(f"  Mean: {pred_returns.mean():.6f}")
        print(f"  Std: {pred_returns.std():.6f}")
        
        # Comparison
        print(f"\nComparison (Target vs Prediction):")
        print(f"  Target std: {y_denorm_returns[:100].std():.6f}")
        print(f"  Prediction std: {pred_returns.std():.6f}")
        print(f"  Ratio: {pred_returns.std() / (y_denorm_returns[:100].std() + 1e-8):.4f}")
        
        # Check per-step variability
        print(f"\nPrediction variability by step:")
        for i in range(pred_length):
            step_pred = pred_returns[:, i]
            print(f"  Step {i+1}: mean={step_pred.mean():.6f}, std={step_pred.std():.6f}, range=[{step_pred.min():.6f}, {step_pred.max():.6f}]")
    else:
        print("Model file not found!")
    
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    print("""
    If prediction std is much lower than target std:
    - Model is over-smoothing (learning to predict average)
    - This is a classic LSTM regression issue
    
    Solutions:
    1. Use different loss function (e.g., Huber loss instead of MSE)
    2. Add noise to training data for regularization
    3. Use Bayesian layers for uncertainty quantification
    4. Try different network architecture (residual connections)
    5. Reduce dropout to allow model to capture more variation
    """)

if __name__ == "__main__":
    analyze_training_data()
