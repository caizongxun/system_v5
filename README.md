# System V5: Cryptocurrency Price Prediction System

A modular LSTM-based system for predicting cryptocurrency price movements using historical K-line data.

## Project Structure

```
system_v5/
├── config/
│   └── config.yaml              # Configuration file for all parameters
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Load data from HuggingFace
│   ├── data_processor.py        # Preprocessing and feature engineering
│   ├── model.py                 # LSTM model architecture
│   ├── evaluator.py             # Model evaluation and metrics
│   └── utils.py                 # Utility functions
├── test/
│   ├── run_pipeline.py          # Main pipeline runner
│   ├── data/                    # Data cache directory
│   ├── models/                  # Trained models
│   ├── results/                 # Evaluation results
│   └── logs/                    # Execution logs
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Parameters

Edit `config/config.yaml` to adjust:
- Model hyperparameters (LSTM units, dropout, epochs)
- Data parameters (symbol, timeframe)
- Feature engineering options
- Data split ratios

### 3. Run Pipeline

```bash
python test/run_pipeline.py
```

## Pipeline Overview

The main pipeline (`test/run_pipeline.py`) executes 8 modular steps:

1. **Load Data** - Fetch K-line data from HuggingFace dataset
2. **Preprocess** - Add technical indicators and normalize features
3. **Create Sequences** - Generate training sequences (100 bars input, 15 bars output)
4. **Split Data** - Divide into train/validation/test sets
5. **Build Model** - Initialize LSTM architecture
6. **Train Model** - Train with early stopping and learning rate decay
7. **Evaluate** - Calculate metrics and generate visualizations
8. **Summary** - Print execution summary

## Configuration

Key parameters in `config/config.yaml`:

- `sequence_length`: 100 (input K-bars)
- `prediction_length`: 15 (output K-bars to predict)
- `lstm_units`: [128, 64] (hidden layer sizes)
- `epochs`: 50
- `batch_size`: 32
- `learning_rate`: 0.001

## Data Source

Data is loaded from HuggingFace dataset: `zongowo111/v2-crypto-ohlcv-data`

- Contains 38 cryptocurrency trading pairs
- Available timeframes: 15m, 1h, 1d
- OHLCV format with technical metadata

## Output

- `test/models/` - Trained model files
- `test/results/` - Metrics report, prediction plots, training history
- `test/logs/` - Execution logs with timestamps
- `test/data/` - Cached data from HuggingFace

## Technical Details

### Model Architecture

- Input: (batch_size, 100, num_features)
- LSTM Layer 1: 128 units + BatchNorm + Dropout
- LSTM Layer 2: 64 units + BatchNorm + Dropout
- Dense Layer: 64 units with ReLU
- Output: (batch_size, 15, num_features)

### Features Used

Base OHLCV:
- Open, High, Low, Close, Volume

Technical Indicators (if enabled):
- SMA_10, SMA_20, SMA_50
- RSI (14)
- MACD, MACD Signal, MACD Diff
- Bollinger Bands (20, 2)

### Evaluation Metrics

- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)

## Future Enhancements

- [ ] Multi-symbol training
- [ ] Ensemble models
- [ ] Attention mechanisms
- [ ] Real-time prediction API
- [ ] Backtesting framework
- [ ] Risk management integration

## License

MIT
