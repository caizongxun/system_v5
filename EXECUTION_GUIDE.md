# BTC_15m Prediction System - PyTorch Training Guide

## Overview

This system trains an LSTM model to predict the next 15 K-bars (15-minute candles) based on the previous 100 K-bars for BTC/USDT trading pair. The model uses 16 technical features including volatility indicators to capture market dynamics.

## Key Features

### 1. Enhanced Volatility Capture
- **volatility_5**: 5-period rolling standard deviation of returns
- **volatility_20**: 20-period rolling standard deviation of returns
- **momentum_5**: 5-period price momentum
- **momentum_10**: 10-period price momentum
- **ATR**: Average True Range (normalized by price)

### 2. Model Architecture
- Framework: PyTorch
- Architecture: 2-layer LSTM (256 -> 128 units)
- Loss Function: Huber Loss (delta=0.5) - preserves variance better than MSE
- Normalization: StandardScaler (Z-score) - preserves variance information
- Dropout: 0.3 for regularization
- L2 Regularization: 0.0001

### 3. Training Configuration
- Input: 100 K-bars x 16 features
- Output: 15 K-bars x 16 features (including volatility predictions)
- Batch Size: 32
- Epochs: 100 (with early stopping)
- Learning Rate: 0.001
- Device: Auto-detect (GPU if available, CPU otherwise)

## Execution

### Prerequisites

```bash
pip install -r requirements.txt
```

Required packages:
- torch
- tensorflow (for GPU initialization)
- scikit-learn
- pandas
- numpy
- ta-lib (technical analysis)
- huggingface-hub

### Running the Training Pipeline

```bash
python test/run_pipeline_pytorch.py
```

This will execute 8 sequential steps:

1. **STEP 0: Device Initialization**
   - Auto-detect GPU (CUDA)
   - Display device information
   - Falls back to CPU if GPU unavailable

2. **STEP 1: Loading Data**
   - Downloads BTC_15m data from HuggingFace
   - Validates data integrity
   - Caches locally for future runs

3. **STEP 2: Preprocessing Data**
   - Adds 16 technical indicators
   - Normalizes features using StandardScaler
   - Preserves variance information for volatility prediction

4. **STEP 3: Creating Sequences**
   - Creates sliding windows: 100 bars -> 15 bars
   - Total sequences depend on data length (~220k sequences for BTC_15m)

5. **STEP 4: Splitting Data**
   - Train: 70% of data
   - Validation: 20% of data
   - Test: 10% of data
   - No shuffle (time-series aware)

6. **STEP 5: Building Model**
   - Initializes 2-layer LSTM with Huber loss
   - Displays model parameters and architecture

7. **STEP 6: Training Model**
   - Trains for up to 100 epochs
   - Early stopping if validation loss doesn't improve for 10 epochs
   - Saves best model weights
   - GPU acceleration available

8. **STEP 7: Evaluating Model**
   - Makes predictions on test set
   - Calculates RMSE, MAE, MAPE, R2 score
   - Feature-specific analysis for volatility indicators
   - Saves detailed metrics report

9. **STEP 8: Summary**
   - Displays final configuration and results
   - Saves scaler and feature columns for later inference

## Output Files

All outputs are saved to `test/` directory:

```
test/
├── data/
│   └── (cached K-line data from HuggingFace)
├── models/
│   ├── btc_15m_model_pytorch.pt      # Model weights
│   ├── btc_15m_scaler.pkl           # Feature scaler (StandardScaler)
│   └── feature_columns.pkl          # Feature column names
├── results/
│   ├── metrics_report.txt           # Detailed metrics
│   ├── predictions.npy              # Predictions on test set
│   ├── targets.npy                  # Ground truth targets
│   └── visualizations/              # Optional plots
└── logs/
    └── training_YYYYMMDD_HHMMSS.log # Training logs
```

## Interpreting Results

### Key Metrics

1. **R2 Score**: Coefficient of determination
   - Range: 0-1 (higher is better)
   - Target: 0.80+
   - Measures how well model explains variance

2. **RMSE**: Root Mean Squared Error
   - Lower is better
   - Penalizes large errors more heavily

3. **MAE**: Mean Absolute Error
   - Robust to outliers
   - Easier to interpret (same units as predictions)

4. **MAPE**: Mean Absolute Percentage Error
   - Percentage-based error metric
   - Useful for comparing across different price levels

### Volatility-Specific Metrics

The model outputs predictions for all 16 features, including volatility:

```
volume_ratio (pred/target): X.XXX
  - 1.0 = perfect calibration
  - < 1.0 = model under-predicts volatility (too smooth)
  - > 1.0 = model over-predicts volatility
```

Target: Volatility ratio should be 0.85-1.05 for realistic predictions.

## Troubleshooting

### GPU Memory Issues

If you encounter CUDA out-of-memory errors:

1. Reduce batch size in config.yaml:
   ```yaml
   batch_size: 16  # was 32
   ```

2. Reduce LSTM units:
   ```yaml
   lstm_units: 128  # was 256
   ```

3. Use CPU:
   ```yaml
   gpu:
     enabled: false
   ```

### Slow Training

If training is slow on GPU:

1. Verify CUDA is properly installed
2. Check for cuDNN installation
3. Monitor GPU usage: `nvidia-smi` (NVIDIA cards)
4. Consider reducing epochs for quick tests

### Data Loading Issues

If HuggingFace download fails:

1. Check internet connection
2. Verify HuggingFace token (if private dataset)
3. Manually download: https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data
4. Place in `test/data/` folder

## Next Steps

After successful training:

1. **Evaluate stability** (recommended):
   - Run inference with the same model multiple times
   - Verify identical inputs produce identical predictions
   - Check that volatility_5 and volatility_20 predictions are stable

2. **Build production model** (after validation):
   - Copy trained model to production directory
   - Implement real-time data pipeline
   - Create REST API for predictions

3. **Scale to other coins** (after testing):
   - Change `symbol` in config.yaml
   - Run same pipeline for other coins
   - Compare models across different coins

## Configuration Reference

All settings are in `config/config.yaml`:

```yaml
data:
  symbol: "BTCUSDT"      # Trading pair (38 options available)
  timeframe: "15m"       # K-bar timeframe (15m, 1h, 1d)

model:
  sequence_length: 100   # Input length (100 K-bars)
  prediction_length: 15  # Output length (15 K-bars)
  lstm_units: 256        # LSTM hidden units (first layer)
  dropout_rate: 0.3      # Dropout for regularization
  batch_size: 32         # Training batch size
  epochs: 100            # Maximum epochs
  learning_rate: 0.001   # Adam optimizer learning rate
```

## Advanced Usage

### Custom Data Path

To use custom K-line data instead of HuggingFace:

1. Prepare CSV with columns: `open_time, open, high, low, close, volume, close_time`
2. Modify `src/data_loader.py` to load from CSV
3. Run pipeline as normal

### Training on Different Timeframes

Change `timeframe` in config.yaml:
```yaml
timeframe: "1h"   # 1-hour K-bars
# or
timeframe: "1d"   # Daily K-bars
```

### Adjusting Feature Set

Edit `selected_features` in config.yaml to add/remove indicators:

```yaml
selected_features:
  - returns          # Keep this
  - volatility_5     # New
  - custom_indicator # Add your own
```

## Performance Notes

### Expected Training Time

- **GPU (Tesla T4)**: ~60-90 minutes for 100 epochs
- **GPU (RTX 3090)**: ~5-10 minutes for 100 epochs
- **CPU (Intel i7)**: ~2-4 hours for 100 epochs

### Memory Requirements

- **GPU memory**: ~10GB (with current config)
- **RAM**: ~20GB (for data loading and processing)
- **Disk**: ~50GB (including cache and model)

## References

- HuggingFace Dataset: https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data
- PyTorch Documentation: https://pytorch.org/docs/stable/index.html
- LSTM Papers: https://colah.github.io/posts/2015-08-Understanding-LSTMs/

## Support

For issues or questions:
1. Check logs in `test/logs/`
2. Review error messages in console output
3. Verify config.yaml matches your setup
4. Test data loading separately with `test/diagnose_training_data.py`
