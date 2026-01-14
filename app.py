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
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
    else:
        st.error("Model file not found! Please train the model first.")
    
    # Load scaler
    scaler_path = Path(config['paths']['model_dir']) / "btc_15m_scaler.pkl"
    scaler = None
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    else:
        st.warning("Scaler file not found. Using new scaler.")
    
    return model, config, feature_columns, device, scaler

def load_data_from_binance(limit: int = 500) -> pd.DataFrame:
    """
    Load real-time data from Binance US API.
    Falls back to yfinance if Binance is unavailable.
    """
    try:
        # Try Binance US first
        binance_loader = BinanceRealtimeLoader(symbol="BTCUSDT", interval="15m")
        df = binance_loader.get_latest_klines(limit=limit)
        
        if not df.empty:
            logger.info(f"Successfully loaded {len(df)} klines from Binance US")
            return df
        else:
            logger.warning("Binance returned empty data, trying yfinance...")
            raise Exception("Binance returned empty data")
    
    except Exception as e:
        logger.warning(f"Binance loading failed: {str(e)}, falling back to yfinance...")
        try:
            yfinance_loader = YFinanceRealtimeLoader(symbol="BTC-USD", interval="15m")
            df = yfinance_loader.fetch_klines(days_back=30)
            if not df.empty:
                logger.info(f"Successfully loaded {len(df)} klines from yfinance")
                return df
        except Exception as yf_error:
            logger.error(f"yfinance also failed: {str(yf_error)}")
    
    return pd.DataFrame()

def preprocess_data(df, processor, feature_columns, scaler=None):
    df = df.copy()
    
    if 'returns' not in df.columns:
        df_processed = processor.add_technical_indicators(df)
    else:
        df_processed = df
    
    if scaler is not None:
        normalized_data = scaler.transform(df_processed[feature_columns].values)
    else:
        normalized_data, scaler = processor.normalize_features(df_processed, feature_columns)
    
    return normalized_data, scaler, df_processed

def get_current_time_and_next_prediction_time():
    """
    Get current time, last closed K-bar time, and next prediction start time.
    """
    now = datetime.now()
    minutes = now.minute
    
    # 15-minute K-bar boundaries: 00, 15, 30, 45
    current_boundary = (minutes // 15) * 15
    
    # Time of last formed/closed K-bar
    formed_candle_time = now.replace(minute=current_boundary, second=0, microsecond=0)
    
    # Next K-bar prediction start time
    if current_boundary == 45:
        next_prediction_time = formed_candle_time.replace(minute=0) + timedelta(hours=1)
    else:
        next_prediction_time = formed_candle_time + timedelta(minutes=15)
    
    return formed_candle_time, next_prediction_time

def plot_klines(df_historical, predicted_klines, current_time_info):
    """
    Plot K-bars with real-time data and predictions.
    """
    df_plot = df_historical.tail(100).copy()
    
    if isinstance(df_plot['open_time'].iloc[0], str):
        df_plot['open_time'] = pd.to_datetime(df_plot['open_time'])
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=("BTC 15m K-Line with Real-Time Predictions (Binance)", "Volume")
    )
    
    # Historical K-bars
    fig.add_trace(
        go.Candlestick(
            x=df_plot['open_time'],
            open=df_plot['open'],
            high=df_plot['high'],
            low=df_plot['low'],
            close=df_plot['close'],
            name='Historical (Binance)',
            visible=True
        ),
        row=1, col=1
    )
    
    # Predicted K-bars
    if predicted_klines:
        pred_times = [k['time'] for k in predicted_klines]
        pred_opens = [k['open'] for k in predicted_klines]
        pred_highs = [k['high'] for k in predicted_klines]
        pred_lows = [k['low'] for k in predicted_klines]
        pred_closes = [k['close'] for k in predicted_klines]
        
        fig.add_trace(
            go.Candlestick(
                x=pred_times,
                open=pred_opens,
                high=pred_highs,
                low=pred_lows,
                close=pred_closes,
                name='Predicted (Next 15 Bars)',
                visible=True,
                increasing=dict(line=dict(color='lime')),
                decreasing=dict(line=dict(color='red'))
            ),
            row=1, col=1
        )
    
    # Historical volume
    fig.add_trace(
        go.Bar(
            x=df_plot['open_time'],
            y=df_plot['volume'],
            name='Historical Volume',
            showlegend=False,
            marker=dict(color='rgba(128, 128, 128, 0.5)')
        ),
        row=2, col=1
    )
    
    # Predicted volume
    if predicted_klines:
        pred_volumes = [k['volume'] for k in predicted_klines]
        fig.add_trace(
            go.Bar(
                x=pred_times,
                y=pred_volumes,
                name='Predicted Volume',
                showlegend=False,
                marker=dict(color='rgba(255, 165, 0, 0.5)')
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        title="BTC/USDT 15M with Real-Time Price Predictions (Binance Data)",
        yaxis_title='Price (USDT)',
        xaxis_rangeslider_visible=False,
        height=700,
        hovermode='x unified',
        template='plotly_dark'
    )
    
    fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

def display_prediction_table(predicted_klines):
    data = []
    for i, k in enumerate(predicted_klines, 1):
        data.append({
            'K Bar': i,
            'Time': k['time'].strftime('%Y-%m-%d %H:%M'),
            'Open': f"${k['open']:.2f}",
            'High': f"${k['high']:.2f}",
            'Low': f"${k['low']:.2f}",
            'Close': f"${k['close']:.2f}",
            'Change %': f"{((k['close'] - k['open']) / k['open'] * 100):.3f}%",
            'Range': f"${k['high'] - k['low']:.2f}"
        })
    
    return pd.DataFrame(data)

def main():
    st.set_page_config(
        page_title="BTC Price Predictor",
        page_icon="btc",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("BTC 15M Price Prediction Model (Real-Time - Binance US)")
    st.markdown("---")
    
    with st.sidebar:
        st.header("Settings")
        
        # Auto-refresh interval
        refresh_interval = st.slider(
            "Auto-Refresh Interval (seconds)",
            min_value=30,
            max_value=300,
            value=60,
            step=30,
            help="How often to fetch new market data from Binance"
        )
        
        volatility_multiplier = st.slider(
            "Volatility Multiplier",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.5,
            help="Controls visualization of price fluctuation magnitude"
        )
        
        show_table = st.checkbox("Show Prediction Table", value=True)
        show_analysis = st.checkbox("Show Volatility Analysis", value=True)
        show_metrics = st.checkbox("Show Model Metrics", value=True)
        show_time_info = st.checkbox("Show Time Information", value=True)
        
        st.markdown("---")
        st.info("""
        Real-Time Mode (Binance US):
        
        - Data: Live from Binance US API
        - Fallback: yfinance if Binance unavailable
        - Predictions: Based on latest closed K-bar
        - Update: Auto-refresh at set interval
        
        Do NOT use for actual trading without
        additional risk management and validation.
        """)
    
    st.info("Loading model and real-time market data from Binance US...")
    
    try:
        model, config, feature_columns, device, scaler = load_model_and_config()
        processor = DataProcessor()
        
        st.success("Model loaded successfully!")
        
        # Create placeholders for real-time updates
        time_info_placeholder = st.empty()
        data_source_placeholder = st.empty()
        chart_placeholder = st.empty()
        metrics_row = st.empty()
        volatility_placeholder = st.empty()
        table_placeholder = st.empty()
        model_info_placeholder = st.empty()
        update_time_placeholder = st.empty()
        
        # Real-time data update loop
        last_update = 0
        data_source = "Initializing..."
        
        while True:
            current_time = time.time()
            
            # Check if should update
            if current_time - last_update >= refresh_interval:
                with st.spinner(f"Fetching real-time data from Binance US..."):
                    # Load real-time data
                    df = load_data_from_binance(limit=500)
                    
                    if df.empty:
                        st.error("Failed to load data from both Binance and yfinance!")
                        time.sleep(5)
                        continue
                    
                    # Determine data source
                    if len(df) > 100:
                        data_source = "Binance US API"
                    else:
                        data_source = "yfinance (Binance unavailable)"
                    
                    # Preprocess
                    normalized_data, scaler, df_processed = preprocess_data(
                        df, processor, feature_columns, scaler
                    )
                    
                    # Get time info
                    formed_candle_time, next_pred_time = get_current_time_and_next_prediction_time()
                    
                    # Make predictions based on last 100 bars
                    predictor = RollingPredictor(model, scaler, feature_columns, device)
                    X_baseline = normalized_data[-100:]
                    predicted_klines = predictor.predict_with_fixed_baseline(
                        X_baseline, df, pred_steps=config['model']['prediction_length'],
                        volatility_multiplier=volatility_multiplier
                    )
                    
                    last_update = current_time
                
                # Display time info
                if show_time_info:
                    with time_info_placeholder.container():
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Current Time", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                        with col2:
                            st.metric("Last Closed K-Bar", formed_candle_time.strftime('%Y-%m-%d %H:%M'))
                        with col3:
                            st.metric("Next Prediction", next_pred_time.strftime('%Y-%m-%d %H:%M'))
                        with col4:
                            st.metric("Data Source", data_source)
                        st.markdown("---")
                
                # Display chart
                with chart_placeholder.container():
                    st.subheader("Price Chart with Real-Time Predictions")
                    current_time_info = {
                        'formed_candle': formed_candle_time,
                        'next_pred_time': next_pred_time
                    }
                    fig = plot_klines(df, predicted_klines, current_time_info)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display metrics
                with metrics_row.container():
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Current Price (Binance)",
                            f"${df.iloc[-1]['close']:.2f}",
                            f"{((df.iloc[-1]['close'] - df.iloc[-2]['close']) / df.iloc[-2]['close'] * 100):.3f}%"
                        )
                    
                    with col2:
                        avg_pred = np.mean([k['close'] for k in predicted_klines])
                        change = ((avg_pred - df.iloc[-1]['close']) / df.iloc[-1]['close'] * 100)
                        st.metric(
                            "Avg Predicted Price (15 bars)",
                            f"${avg_pred:.2f}",
                            f"{change:.3f}%"
                        )
                    
                    with col3:
                        high_pred = max([k['high'] for k in predicted_klines])
                        st.metric(
                            "Predicted High",
                            f"${high_pred:.2f}"
                        )
                    
                    with col4:
                        low_pred = min([k['low'] for k in predicted_klines])
                        st.metric(
                            "Predicted Low",
                            f"${low_pred:.2f}"
                        )
                
                # Volatility analysis
                if show_analysis:
                    with volatility_placeholder.container():
                        st.markdown("---")
                        st.subheader("Volatility Analysis")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        current_vol = df['close'].pct_change().tail(20).std() * 100
                        
                        pred_returns = [
                            (k['close'] - k['open']) / k['open']
                            for k in predicted_klines
                        ]
                        pred_vol = np.std(pred_returns) * 100 if pred_returns else 0
                        
                        pred_range = max([k['high'] for k in predicted_klines]) - min([k['low'] for k in predicted_klines])
                        
                        with col1:
                            st.metric(
                                "Current Volatility (20-bar)",
                                f"{current_vol:.3f}%"
                            )
                        
                        with col2:
                            st.metric(
                                "Predicted Volatility (15-bar)",
                                f"{pred_vol:.3f}%",
                                f"{pred_vol - current_vol:.3f}%"
                            )
                        
                        with col3:
                            st.metric(
                                "Predicted Price Range",
                                f"${pred_range:.2f}"
                            )
                
                # Prediction table
                if show_table:
                    with table_placeholder.container():
                        st.markdown("---")
                        st.subheader("Detailed Predictions (Next 15 K-Bars)")
                        pred_df = display_prediction_table(predicted_klines)
                        st.dataframe(pred_df, use_container_width=True)
                
                # Model metrics
                if show_metrics:
                    with model_info_placeholder.container():
                        st.markdown("---")
                        st.subheader("Model Information")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Model Type", "LSTM")
                        with col2:
                            st.metric("LSTM Units", str(config['model']['lstm_units']))
                        with col3:
                            st.metric("Features", len(feature_columns))
                        with col4:
                            st.metric("Sequence Length", config['model']['sequence_length'])
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("R2 Score (Test)", "0.2447")
                        with col2:
                            st.metric("RMSE", "0.5811")
                        with col3:
                            st.metric("MAE", "0.3210")
                
                # Update timestamp
                with update_time_placeholder.container():
                    st.markdown("---")
                    st.markdown(
                        f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                        f"Next update in: {refresh_interval}s | "
                        f"Data Source: {data_source}"
                    )
            
            # Avoid excessive resource usage
            time.sleep(1)
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.write("Troubleshooting:")
        st.write("1. Ensure model file exists: test/models/btc_15m_model_pytorch.pt")
        st.write("2. Ensure scaler file exists: test/models/btc_15m_scaler.pkl")
        st.write("3. Ensure config exists: config/config.yaml")
        st.write("4. Check internet connection for Binance API access")
        st.write("5. Install requests library: pip install requests")
        logger.exception("App error:")

if __name__ == "__main__":
    main()
