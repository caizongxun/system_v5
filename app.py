import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import torch
import logging
import pickle
import time

sys.path.insert(0, str(Path(__file__).parent))

from src.utils import load_config
from src.data_processor import DataProcessor
from src.model_pytorch import LSTMModel
from src.predictor import RollingPredictor
from src.realtime_data_loader import BinanceRealtimeLoader, YFinanceRealtimeLoader

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource
def load_model_and_config():
    """Load model and configuration."""
    try:
        config = load_config()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        feature_columns = config['features'].get('selected_features', [
            'returns', 'high_low_ratio', 'open_close_ratio',
            'price_to_sma_10', 'price_to_sma_20', 'volatility_20',
            'volatility_5', 'momentum_5', 'momentum_10', 'ATR',
            'RSI', 'MACD', 'MACD_Signal', 'BB_Position',
            'Volume_Ratio', 'returns_std_5'
        ])
        
        model = LSTMModel(
            sequence_length=config['model']['sequence_length'],
            num_features=len(feature_columns),
            prediction_length=config['model']['prediction_length'],
            lstm_units=config['model']['lstm_units'],
            dropout_rate=config['model']['dropout_rate'],
            learning_rate=config['model']['learning_rate'],
            l2_reg=config['model'].get('l2_regularization', 0.0),
            device=device,
            use_huber_loss=True
        )
        
        model_path = Path(config['paths']['model_dir']) / "btc_15m_model_pytorch.pt"
        if model_path.exists():
            model.load(str(model_path))
            model.eval()
            logger.info("Model loaded successfully")
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        scaler_path = Path(config['paths']['model_dir']) / "btc_15m_scaler.pkl"
        scaler = None
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            logger.info("Scaler loaded")
        else:
            logger.warning("Scaler not found")
        
        return model, config, feature_columns, device, scaler
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def load_data_from_binance(limit: int = 500):
    """Load data from Binance US or yfinance."""
    try:
        logger.info("Loading from Binance US...")
        binance_loader = BinanceRealtimeLoader(symbol="BTCUSDT", interval="15m")
        df = binance_loader.get_latest_klines(limit=limit)
        
        if not df.empty:
            logger.info(f"Loaded {len(df)} klines from Binance")
            return df, "Binance US"
    
    except Exception as e:
        logger.warning(f"Binance failed: {str(e)}")
    
    # Fallback to yfinance
    try:
        logger.info("Trying yfinance...")
        yfinance_loader = YFinanceRealtimeLoader(symbol="BTC-USD", interval="15m")
        df = yfinance_loader.fetch_klines(days_back=30)
        if not df.empty:
            logger.info(f"Loaded {len(df)} from yfinance")
            return df, "yFinance"
    except Exception as e:
        logger.error(f"yfinance failed: {str(e)}")
    
    return pd.DataFrame(), "Failed"

def preprocess_data(df, processor, feature_columns, scaler=None):
    """Preprocess data with technical indicators."""
    df = df.copy()
    
    if 'returns' not in df.columns:
        logger.info("Adding technical indicators...")
        df_processed = processor.add_technical_indicators(df)
    else:
        df_processed = df.copy()
    
    if scaler is not None:
        normalized_data = scaler.transform(df_processed[feature_columns].values)
    else:
        normalized_data, scaler = processor.normalize_features(df_processed, feature_columns)
    
    return normalized_data, scaler, df_processed

def plot_klines_chart(df_historical, predicted_klines):
    """Create candlestick chart with predictions."""
    try:
        df_plot = df_historical.tail(100).copy()
        
        if isinstance(df_plot['open_time'].iloc[0], str):
            df_plot['open_time'] = pd.to_datetime(df_plot['open_time'])
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
            subplot_titles=("BTC/USDT 15M - Real-Time", "Volume")
        )
        
        # Historical candlesticks
        fig.add_trace(
            go.Candlestick(
                x=df_plot['open_time'],
                open=df_plot['open'],
                high=df_plot['high'],
                low=df_plot['low'],
                close=df_plot['close'],
                name='Market',
                increasing=dict(fillcolor='green', line=dict(color='green')),
                decreasing=dict(fillcolor='red', line=dict(color='red'))
            ),
            row=1, col=1
        )
        
        # Predicted candlesticks
        if predicted_klines and len(predicted_klines) > 0:
            logger.info(f"Adding {len(predicted_klines)} predicted candles")
            
            pred_times = [k['time'] for k in predicted_klines]
            pred_opens = [float(k['open']) for k in predicted_klines]
            pred_highs = [float(k['high']) for k in predicted_klines]
            pred_lows = [float(k['low']) for k in predicted_klines]
            pred_closes = [float(k['close']) for k in predicted_klines]
            
            fig.add_trace(
                go.Candlestick(
                    x=pred_times,
                    open=pred_opens,
                    high=pred_highs,
                    low=pred_lows,
                    close=pred_closes,
                    name='Predictions',
                    increasing=dict(fillcolor='lime', line=dict(color='lime')),
                    decreasing=dict(fillcolor='#FF6B6B', line=dict(color='#FF6B6B'))
                ),
                row=1, col=1
            )
            
            pred_volumes = [float(k['volume']) for k in predicted_klines]
            fig.add_trace(
                go.Bar(
                    x=pred_times,
                    y=pred_volumes,
                    name='Pred Vol',
                    marker=dict(color='rgba(255, 165, 0, 0.5)'),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Historical volume
        fig.add_trace(
            go.Bar(
                x=df_plot['open_time'],
                y=df_plot['volume'],
                name='Volume',
                marker=dict(color='rgba(128, 128, 128, 0.3)'),
                showlegend=False
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title="BTC/USDT 15M Real-Time Chart (Binance)",
            yaxis_title='Price (USDT)',
            xaxis_rangeslider_visible=False,
            height=700,
            hovermode='x unified',
            template='plotly_dark',
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating chart: {str(e)}")
        return None

def display_prediction_table(predicted_klines):
    """Create prediction table."""
    data = []
    for i, k in enumerate(predicted_klines, 1):
        data.append({
            'Bar': i,
            'Time': k['time'].strftime('%H:%M'),
            'Open': f"${k['open']:.2f}",
            'High': f"${k['high']:.2f}",
            'Low': f"${k['low']:.2f}",
            'Close': f"${k['close']:.2f}",
            'Chg%': f"{((k['close'] - k['open']) / k['open'] * 100):.2f}%",
        })
    return pd.DataFrame(data)

def display_model_diagnostics(pred_denorm, feature_columns):
    """Display raw model predictions for debugging."""
    if pred_denorm is None or len(pred_denorm) == 0:
        st.warning("No diagnostic data available")
        return
    
    st.markdown("---")
    st.subheader("Model Predictions Diagnostic")
    
    try:
        # Get feature indices
        returns_idx = feature_columns.index('returns')
        high_low_ratio_idx = feature_columns.index('high_low_ratio')
        open_close_ratio_idx = feature_columns.index('open_close_ratio')
        volume_ratio_idx = feature_columns.index('Volume_Ratio')
        volatility_5_idx = feature_columns.index('volatility_5')
        volatility_20_idx = feature_columns.index('volatility_20')
        
        # Create detailed table
        diag_data = []
        for i, pred_features in enumerate(pred_denorm):
            returns = float(pred_features[returns_idx])
            high_low_ratio = float(pred_features[high_low_ratio_idx])
            open_close_ratio = float(pred_features[open_close_ratio_idx])
            volume_ratio = float(pred_features[volume_ratio_idx])
            volatility_5 = float(pred_features[volatility_5_idx])
            volatility_20 = float(pred_features[volatility_20_idx])
            
            diag_data.append({
                'Step': i + 1,
                'Returns': f"{returns:.6f}",
                'High/Low': f"{high_low_ratio:.6f}",
                'Open/Close': f"{open_close_ratio:.6f}",
                'Vol Ratio': f"{volume_ratio:.6f}",
                'Vol(5)': f"{volatility_5:.6f}",
                'Vol(20)': f"{volatility_20:.6f}"
            })
        
        diag_df = pd.DataFrame(diag_data)
        st.dataframe(diag_df, width='stretch', hide_index=True, use_container_width=True)
        
        # Summary statistics
        st.subheader("Feature Ranges")
        col1, col2, col3 = st.columns(3)
        
        returns_vals = pred_denorm[:, returns_idx]
        high_low_vals = pred_denorm[:, high_low_ratio_idx]
        volatility_5_vals = pred_denorm[:, volatility_5_idx]
        
        with col1:
            st.metric(
                "Returns",
                f"{returns_vals.min():.6f} to {returns_vals.max():.6f}",
                f"Std: {returns_vals.std():.6f}"
            )
        
        with col2:
            st.metric(
                "High/Low Ratio",
                f"{high_low_vals.min():.6f} to {high_low_vals.max():.6f}",
                f"Std: {high_low_vals.std():.6f}"
            )
        
        with col3:
            st.metric(
                "Volatility(5)",
                f"{volatility_5_vals.min():.6f} to {volatility_5_vals.max():.6f}",
                f"Std: {volatility_5_vals.std():.6f}"
            )
        
        # Log ranges to console
        logger.info(f"Returns range: {returns_vals.min():.6f} to {returns_vals.max():.6f}")
        logger.info(f"High/Low ratio range: {high_low_vals.min():.6f} to {high_low_vals.max():.6f}")
        logger.info(f"Volatility(5) range: {volatility_5_vals.min():.6f} to {volatility_5_vals.max():.6f}")
    
    except Exception as e:
        st.error(f"Error displaying diagnostics: {str(e)}")
        logger.exception(f"Diagnostic error: {str(e)}")

def main():
    st.set_page_config(
        page_title="BTC Real-Time Predictor",
        page_icon="Chart",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("BTC 15M Real-Time Price Predictor")
    st.markdown("Powered by Binance US API")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        refresh_interval = st.slider(
            "Refresh Interval (seconds)",
            min_value=30,
            max_value=300,
            value=60,
            step=30
        )
        
        volatility_mult = st.slider(
            "Volatility Multiplier",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.5
        )
        
        show_table = st.checkbox("Show Predictions", value=True)
        show_analysis = st.checkbox("Show Analysis", value=True)
        show_diagnostics = st.checkbox("Show Model Diagnostics", value=True)
        
        st.markdown("---")
        st.info("""
        Live Mode
        Data: Binance US API
        Candles: 15 min
        Predictions: Next 15 bars
        """)
    
    # Initialize session state with proper defaults
    if 'last_update' not in st.session_state:
        st.session_state.last_update = 0
        st.session_state.predicted_klines = []
        st.session_state.df_current = None
        st.session_state.data_source = "Init"
        st.session_state.model_loaded = False
        st.session_state.pred_denorm = None
    
    # Load model once
    if not st.session_state.model_loaded:
        try:
            with st.spinner("Loading model..."):
                model, config, feature_columns, device, scaler = load_model_and_config()
                st.session_state.model = model
                st.session_state.config = config
                st.session_state.feature_columns = feature_columns
                st.session_state.device = device
                st.session_state.scaler = scaler
                st.session_state.processor = DataProcessor()
                st.session_state.model_loaded = True
                st.success("Model ready!")
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            return
    
    # Status metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        time_placeholder = st.empty()
    with col2:
        price_placeholder = st.empty()
    with col3:
        source_placeholder = st.empty()
    with col4:
        preds_placeholder = st.empty()
    
    # Chart and data containers
    chart_placeholder = st.empty()
    analysis_placeholder = st.empty()
    table_placeholder = st.empty()
    diagnostics_placeholder = st.empty()
    
    # Main update loop
    current_time = time.time()
    should_update = (current_time - st.session_state.last_update) >= refresh_interval or st.session_state.df_current is None
    
    if should_update:
        logger.info("Fetching fresh data...")
        
        try:
            # Fetch data
            df_raw, data_source = load_data_from_binance(limit=300)
            
            if df_raw.empty:
                st.error("No data available from Binance")
                st.stop()
            
            logger.info(f"Loaded {len(df_raw)} rows")
            st.session_state.data_source = data_source
            
            # Preprocess
            normalized_data, scaler, df_processed = preprocess_data(
                df_raw, 
                st.session_state.processor,
                st.session_state.feature_columns,
                st.session_state.scaler
            )
            
            # Update current data
            st.session_state.df_current = df_raw.copy()
            st.session_state.scaler = scaler
            
            # Generate predictions
            if len(normalized_data) >= 100:
                logger.info("Generating predictions...")
                predictor = RollingPredictor(
                    st.session_state.model,
                    scaler,
                    st.session_state.feature_columns,
                    st.session_state.device
                )
                
                X_pred = normalized_data[-100:]
                
                # Get predictions (returns both normalized and denormalized)
                st.session_state.predicted_klines = predictor.predict_with_fixed_baseline(
                    X_pred, 
                    df_raw,
                    pred_steps=st.session_state.config['model']['prediction_length'],
                    volatility_multiplier=volatility_mult
                )
                
                # Also get denormalized features for diagnostics
                pred_norm = predictor.predict_single_step(X_pred)
                st.session_state.pred_denorm = scaler.inverse_transform(pred_norm)
                
                logger.info(f"Generated {len(st.session_state.predicted_klines)} predictions")
            
            st.session_state.last_update = current_time
        
        except Exception as e:
            logger.exception(f"Error during update: {str(e)}")
            st.error(f"Update failed: {str(e)}")
    
    # Display status metrics
    if st.session_state.df_current is not None:
        price = st.session_state.df_current.iloc[-1]['close']
        prev_price = st.session_state.df_current.iloc[-2]['close']
        change = ((price - prev_price) / prev_price * 100)
        
        with time_placeholder:
            st.metric("Time", datetime.now().strftime('%H:%M'))
        with price_placeholder:
            st.metric("Price", f"${price:.2f}", f"{change:.3f}%")
        with source_placeholder:
            st.metric("Source", st.session_state.data_source)
        with preds_placeholder:
            st.metric("Preds", len(st.session_state.predicted_klines))
    
    st.markdown("---")
    
    # Display chart
    with chart_placeholder.container():
        st.subheader("Chart with Predictions")
        
        if st.session_state.df_current is not None:
            fig = plot_klines_chart(
                st.session_state.df_current,
                st.session_state.predicted_klines
            )
            
            if fig is not None:
                st.plotly_chart(fig, width='stretch')
            else:
                st.error("Chart generation failed")
        else:
            st.warning("Waiting for data...")
    
    # Display analysis
    if show_analysis and st.session_state.df_current is not None:
        with analysis_placeholder.container():
            st.markdown("---")
            st.subheader("Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            price = st.session_state.df_current.iloc[-1]['close']
            current_vol = st.session_state.df_current['close'].pct_change().tail(20).std() * 100
            
            if st.session_state.predicted_klines:
                pred_closes = [k['close'] for k in st.session_state.predicted_klines]
                pred_range = max([k['high'] for k in st.session_state.predicted_klines]) - min([k['low'] for k in st.session_state.predicted_klines])
                avg_pred = np.mean(pred_closes)
                change = ((avg_pred - price) / price * 100)
            else:
                pred_range = 0
                avg_pred = price
                change = 0
            
            with col1:
                st.metric("Vol (20)", f"{current_vol:.3f}%")
            with col2:
                st.metric("Range", f"${pred_range:.2f}")
            with col3:
                st.metric("Avg Pred", f"${avg_pred:.2f}", f"{change:.2f}%")
    
    # Display table
    if show_table and st.session_state.predicted_klines:
        with table_placeholder.container():
            st.markdown("---")
            st.subheader("Next 15 Predictions")
            
            pred_df = display_prediction_table(st.session_state.predicted_klines)
            st.dataframe(pred_df, width='stretch', hide_index=True)
    
    # Display diagnostics
    if show_diagnostics and st.session_state.pred_denorm is not None:
        with diagnostics_placeholder.container():
            display_model_diagnostics(
                st.session_state.pred_denorm,
                st.session_state.feature_columns
            )
    
    # Auto-refresh
    time.sleep(1)
    st.rerun()

if __name__ == "__main__":
    main()
