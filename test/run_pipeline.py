import sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, load_config, create_directories, print_section
from src.data_loader import DataLoader
from src.data_processor import DataProcessor
from src.model import LSTMModel
from src.evaluator import Evaluator

logger = None

def step_1_load_data(config):
    """
    Step 1: Load data from HuggingFace
    """
    print_section("STEP 1: Loading Data")
    
    loader = DataLoader(
        repo_id=config['data']['repo_id'],
        cache_dir=Path(config['paths']['data_dir'])
    )
    
    df = loader.load_klines(
        symbol=config['data']['symbol'],
        timeframe=config['data']['timeframe']
    )
    
    if not loader.validate_data(df):
        raise ValueError("Data validation failed")
    
    logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"Date range: {df['open_time'].min()} to {df['open_time'].max()}")
    
    return df

def step_2_preprocess_data(df, config):
    """
    Step 2: Preprocess and engineer features
    """
    print_section("STEP 2: Preprocessing Data")
    
    processor = DataProcessor()
    
    if config['features']['use_technical_indicators']:
        df = processor.add_technical_indicators(df)
    
    feature_columns = ['open', 'high', 'low', 'close', 'volume']
    if config['features']['use_technical_indicators']:
        feature_columns.extend(['SMA_10', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal'])
    
    logger.info(f"Using {len(feature_columns)} features: {feature_columns}")
    
    normalized_data, scaler = processor.normalize_features(df, feature_columns)
    
    logger.info(f"Data normalized. Shape: {normalized_data.shape}")
    
    return normalized_data, scaler, feature_columns, processor

def step_3_create_sequences(normalized_data, config):
    """
    Step 3: Create training sequences
    """
    print_section("STEP 3: Creating Sequences")
    
    processor = DataProcessor()
    seq_length = config['model']['sequence_length']
    pred_length = config['model']['prediction_length']
    
    X, y = processor.create_sequences(normalized_data, seq_length, pred_length)
    
    logger.info(f"Sequences created: X shape {X.shape}, y shape {y.shape}")
    
    return X, y

def step_4_split_data(X, y, config):
    """
    Step 4: Split data into train, validation, test sets
    """
    print_section("STEP 4: Splitting Data")
    
    test_split = config['model']['test_split']
    val_split = config['model']['validation_split']
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_split, shuffle=False
    )
    
    val_size = int(len(X_temp) * val_split / (1 - test_split))
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, shuffle=False
    )
    
    logger.info(f"Data split:")
    logger.info(f"  Train: {X_train.shape[0]} samples")
    logger.info(f"  Val:   {X_val.shape[0]} samples")
    logger.info(f"  Test:  {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def step_5_build_model(config, num_features):
    """
    Step 5: Build LSTM model
    """
    print_section("STEP 5: Building Model")
    
    model = LSTMModel(
        sequence_length=config['model']['sequence_length'],
        num_features=num_features,
        prediction_length=config['model']['prediction_length'],
        lstm_units=config['model']['lstm_units'],
        dropout_rate=config['model']['dropout_rate'],
        learning_rate=config['model']['learning_rate']
    )
    
    model.build()
    logger.info("Model built successfully")
    
    return model

def step_6_train_model(model, X_train, y_train, X_val, y_val, config):
    """
    Step 6: Train the model
    """
    print_section("STEP 6: Training Model")
    
    model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=config['model']['epochs'],
        batch_size=config['model']['batch_size']
    )
    
    model_path = Path(config['paths']['model_dir']) / "btc_15m_model.h5"
    model.save(str(model_path))
    logger.info(f"Model training completed and saved")
    
    return model

def step_7_evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, config):
    """
    Step 7: Evaluate model on all sets
    """
    print_section("STEP 7: Evaluating Model")
    
    evaluator = Evaluator(results_dir=config['paths']['results_dir'])
    
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    logger.info("--- Training Set Metrics ---")
    train_metrics = evaluator.calculate_metrics(y_train, y_train_pred)
    
    logger.info("--- Validation Set Metrics ---")
    evaluator.calculate_metrics(y_val, y_val_pred)
    
    logger.info("--- Test Set Metrics ---")
    evaluator.calculate_metrics(y_test, y_test_pred)
    
    evaluator.print_metrics_summary()
    evaluator.save_metrics_report()
    
    if model.history is not None:
        evaluator.plot_training_history(model.history)
    
    evaluator.plot_predictions(y_test, y_test_pred, sample_idx=0)
    
    logger.info("Model evaluation completed")
    
    return evaluator

def step_8_summary(config):
    """
    Step 8: Print summary
    """
    print_section("STEP 8: Execution Summary")
    
    logger.info(f"Project: {config['project']['name']}")
    logger.info(f"Symbol: {config['data']['symbol']}")
    logger.info(f"Timeframe: {config['data']['timeframe']}")
    logger.info(f"Model: LSTM with {config['model']['lstm_units']} units")
    logger.info(f"Sequence Length: {config['model']['sequence_length']}")
    logger.info(f"Prediction Length: {config['model']['prediction_length']}")
    logger.info(f"\nResults saved to: {config['paths']['results_dir']}")
    logger.info(f"Model saved to: {config['paths']['model_dir']}")
    logger.info(f"Logs saved to: {config['paths']['logs_dir']}")

def main():
    global logger
    
    config = load_config()
    
    logger = setup_logging(config['paths']['logs_dir'])
    
    create_directories(config)
    
    print_section("BTC_15m Cryptocurrency Price Prediction System")
    logger.info(f"Configuration loaded from config/config.yaml")
    
    try:
        df = step_1_load_data(config)
        
        normalized_data, scaler, feature_columns, processor = step_2_preprocess_data(df, config)
        
        X, y = step_3_create_sequences(normalized_data, config)
        
        X_train, X_val, X_test, y_train, y_val, y_test = step_4_split_data(X, y, config)
        
        model = step_5_build_model(config, len(feature_columns))
        
        model = step_6_train_model(model, X_train, y_train, X_val, y_val, config)
        
        evaluator = step_7_evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, config)
        
        step_8_summary(config)
        
        logger.info("\nPipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
