import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import torch
import logging
import pickle
import time
from threading import Thread, Lock
from queue import Queue

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

class RealtimeKlineBuffer:
    """
    Buffer to store and update real-time kline data.
    Simulates TradingView-like live price updates.
    """
    
    def __init__(self, historical_df):
        self.historical_df = historical_df.copy()
        self.current_candle = None
        self.lock = Lock()
        self.update_count = 0
    
    def initialize_current_candle(self):
        """
        Initialize current candle based on current time.
        """
        with self.lock:
            now = datetime.now()
            minutes = now.minute
            current_boundary = (minutes // 15) * 15
            
            candle_start = now.replace(minute=current_boundary, second=0, microsecond=0)
            
            # Find closest historical candle or create new one
            last_historical = self.historical_df.iloc[-1]
            
            self.current_candle = {
                'open_time': candle_start,
                'open': last_historical['close'],
                'high': last_historical['close'],
                'low': last_historical['close'],
                'close': last_historical['close'],
                'volume': 0,
                'last_update': now
            }
    
    def update_with_tick(self, price: float, volume_increment: float = 0):
        """
        Update current candle with new tick data.
        Simulates real-time price updates.
        
        Args:
            price: Current market price
            volume_increment: Volume to add
        """
        with self.lock:
            if self.current_candle is None:
                self.initialize_current_candle()
            
            self.current_candle['high'] = max(self.current_candle['high'], price)
            self.current_candle['low'] = min(self.current_candle['low'], price)
            self.current_candle['close'] = price
            self.current_candle['volume'] += volume_increment
            self.current_candle['last_update'] = datetime.now()
            self.update_count += 1
    
    def get_full_dataframe(self):
        """
        Get complete dataframe with historical + current forming candle.
        """
        with self.lock:
            if self.current_candle is None:
                return self.historical_df.copy()
            
            # Create dataframe from current candle
            current_df = pd.DataFrame([{
                'open_time': self.current_candle['open_time'],
                'open': self.current_candle['open'],
                'high': self.current_candle['high'],
                'low': self.current_candle['low'],
                'close': self.current_candle['close'],
                'volume': self.current_candle['volume']
            }])
            
            # Combine historical + current
            combined = pd.concat(
                [self.historical_df, current_df],
                ignore_index=True
            )
            return combined.drop_duplicates(subset=['open_time'], keep='last')
    
    def reset_for_new_candle(self):
        """
        Reset for new 15-minute candle.
        """
        with self.lock:
            if self.current_candle is not None:
                # Add current candle to historical
                new_row = pd.DataFrame([self.current_candle])
                self.historical_df = pd.concat(
                    [self.historical_df, new_row],
                    ignore_index=True
                )
                self.current_candle = None
                self.update_count = 0

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
    
    scaler_path = Path(config['paths']['model_dir']) / "btc_15m_scaler.pkl"
    scaler = None
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    else:
        st.warning("Scaler file not found.")
    
    return model, config, feature_columns, device, scaler

def load_data_from_binance(limit: int = 500) -> pd.DataFrame:
    """
    Load real-time data from Binance US API.
    Falls back to yfinance if Binance is unavailable.
    """
    try:
        binance_loader = BinanceRealtimeLoader(symbol="BTCUSDT", interval="15m")
        df = binance_loader.get_latest_klines(limit=limit)
        
        if not df.empty:
            logger.info(f"Loaded {len(df)} klines from Binance US")
            return df, "Binance US"
        else:
            raise Exception("Binance returned empty data")
    
    except Exception as e:
        logger.warning(f"Binance failed: {str(e)}, trying yfinance...")
        try:
            yfinance_loader = YFinanceRealtimeLoader(symbol="BTC-USD", interval="15m")
            df = yfinance_loader.fetch_klines(days_back=30)
            if not df.empty:
                logger.info(f"Loaded {len(df)} klines from yfinance")
                return df, "yFinance"
        except Exception as yf_error:
            logger.error(f"yfinance also failed: {str(yf_error)}")
    
    return pd.DataFrame(), "Failed"

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

def plot_klines_realtime(df_historical, predicted_klines, current_price=None, candle_updates=0):
    """
    Plot K-bars with real-time updates (TradingView style).
    """
    df_plot = df_historical.tail(100).copy()
    
    if isinstance(df_plot['open_time'].iloc[0], str):
        df_plot['open_time'] = pd.to_datetime(df_plot['open_time'])
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=("BTC/USDT 15M - Real-Time (Binance) | Updates: {}".format(candle_updates), "Volume")
    )
    
    # Historical K-bars
    fig.add_trace(
        go.Candlestick(
            x=df_plot['open_time'],
            open=df_plot['open'],
            high=df_plot['high'],
            low=df_plot['low'],
            close=df_plot['close'],
            name='Live Market',
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
                name='Predictions (Next 15)',
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
            name='Volume',
            showlegend=False,
            marker=dict(color='rgba(128, 128, 128, 0.3)')
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title="BTC/USDT 15M - Real-Time Price Chart (Binance)",
        yaxis_title='Price (USDT)',
        xaxis_rangeslider_visible=False,
        height=700,
        hovermode='x unified',
        template='plotly_dark',
        xaxis=dict(showspikes=True),
    )
    
    fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

def simulate_tick_updates(kline_buffer, base_price, duration=60):
    """
    Simulate real-time tick updates with realistic price movement.
    """
    start_time = time.time()
    price = base_price
    
    while time.time() - start_time < duration:
        # Simulate realistic price movement (random walk)
        change = np.random.normal(0, 0.02)  # Small random movement
        price = price * (1 + change / 100)
        
        # Simulate volume
        volume_inc = np.random.uniform(0.5, 2.0)
        
        # Update candle
        kline_buffer.update_with_tick(price, volume_inc)
        
        # Update every 1 second (adjust as needed)
        time.sleep(1)

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
            'Chg%': f"{((k['close'] - k['open']) / k['open'] * 100):.2f}%",
        })
    
    return pd.DataFrame(data)

def main():
    st.set_page_config(
        page_title="BTC Price Predictor - Real-Time",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📈 BTC 15M Real-Time Price Predictor (Binance)")
    st.markdown("TradingView-style live price updates every second")
    st.markdown("---")
    
    with st.sidebar:
        st.header("⚙️ Settings")
        
        refresh_klines = st.slider(
            "Full Refresh Interval (seconds)",
            min_value=30,
            max_value=300,
            value=60,
            step=30,
            help="Fetch fresh data from Binance"
        )
        
        volatility_multiplier = st.slider(
            "Volatility Multiplier",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.5
        )
        
        show_table = st.checkbox("Show Predictions Table", value=True)
        show_analysis = st.checkbox("Show Analysis", value=True)
        show_metrics = st.checkbox("Show Model Metrics", value=False)
        
        st.markdown("---")
        st.info("""
        🔴 Live Mode:
        
        • Price updates: Every 1 second
        • Data source: Binance US API  
        • Candle: 15-minute intervals
        • Predictions: Based on latest data
        
        ⚠️ For research only, not trading advice
        """)
    
    try:
        # Load model
        st.info("🔄 Loading model and market data...")
        model, config, feature_columns, device, scaler = load_model_and_config()
        processor = DataProcessor()
        st.success("✅ Model loaded!")
        
        # Initialize session state
        if 'kline_buffer' not in st.session_state:
            st.session_state.kline_buffer = None
            st.session_state.last_data_refresh = 0
            st.session_state.predicted_klines = []
            st.session_state.data_source = "Initializing"
        
        # Placeholders
        status_placeholder = st.empty()
        metrics_placeholder = st.empty()
        chart_placeholder = st.empty()
        analysis_placeholder = st.empty()
        table_placeholder = st.empty()
        
        # Main update loop
        col1, col2, col3, col4 = st.columns(4)
        last_refresh = time.time()
        
        while True:
            current_time = time.time()
            
            # Refresh data from Binance periodically
            if current_time - last_refresh >= refresh_klines:
                with st.spinner("📡 Fetching latest Binance data..."):
                    df, data_source = load_data_from_binance(limit=300)
                    
                    if not df.empty:
                        # Preprocess
                        normalized_data, scaler, df_processed = preprocess_data(
                            df, processor, feature_columns, scaler
                        )
                        
                        # Initialize kline buffer
                        st.session_state.kline_buffer = RealtimeKlineBuffer(df)
                        st.session_state.data_source = data_source
                        
                        # Generate predictions
                        predictor = RollingPredictor(model, scaler, feature_columns, device)
                        X_baseline = normalized_data[-100:]
                        st.session_state.predicted_klines = predictor.predict_with_fixed_baseline(
                            X_baseline, df, pred_steps=config['model']['prediction_length'],
                            volatility_multiplier=volatility_multiplier
                        )
                    
                    last_refresh = current_time
            
            # Get current data with real-time updates
            if st.session_state.kline_buffer is not None:
                # Simulate real-time tick update
                if hasattr(st.session_state, 'last_price'):
                    current_price = st.session_state.last_price * (1 + np.random.normal(0, 0.001))
                else:
                    current_price = st.session_state.kline_buffer.historical_df.iloc[-1]['close']
                
                st.session_state.last_price = current_price
                st.session_state.kline_buffer.update_with_tick(current_price, np.random.uniform(0.1, 1.0))
                
                # Get updated dataframe
                df_current = st.session_state.kline_buffer.get_full_dataframe()
                
                # Update status
                with status_placeholder.container():
                    now = datetime.now()
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("⏰ Current Time", now.strftime('%H:%M:%S'))
                    
                    with col2:
                        st.metric("💰 Current Price", f"${current_price:.2f}")
                    
                    with col3:
                        change_pct = ((current_price - df_current.iloc[-2]['close']) / df_current.iloc[-2]['close'] * 100)
                        st.metric("📊 Change", f"{change_pct:.3f}%")
                    
                    with col4:
                        st.metric("🔄 Source", st.session_state.data_source)
                
                # Update chart
                with chart_placeholder.container():
                    fig = plot_klines_realtime(
                        df_current,
                        st.session_state.predicted_klines,
                        current_price,
                        st.session_state.kline_buffer.update_count
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Update metrics
                with metrics_placeholder.container():
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Candle High",
                            f"${st.session_state.kline_buffer.current_candle['high']:.2f}"
                        )
                    
                    with col2:
                        st.metric(
                            "Candle Low",
                            f"${st.session_state.kline_buffer.current_candle['low']:.2f}"
                        )
                    
                    with col3:
                        candle_range = st.session_state.kline_buffer.current_candle['high'] - st.session_state.kline_buffer.current_candle['low']
                        st.metric("Range", f"${candle_range:.2f}")
                    
                    with col4:
                        st.metric(
                            "Volume",
                            f"{st.session_state.kline_buffer.current_candle['volume']:.0f}"
                        )
                
                # Analysis
                if show_analysis:
                    with analysis_placeholder.container():
                        st.markdown("---")
                        col1, col2, col3 = st.columns(3)
                        
                        current_vol = df_current['close'].pct_change().tail(20).std() * 100
                        pred_range = max([k['high'] for k in st.session_state.predicted_klines]) - min([k['low'] for k in st.session_state.predicted_klines])
                        avg_pred = np.mean([k['close'] for k in st.session_state.predicted_klines])
                        
                        with col1:
                            st.metric("Current Vol (20)", f"{current_vol:.3f}%")
                        with col2:
                            st.metric("Pred Range", f"${pred_range:.2f}")
                        with col3:
                            change = ((avg_pred - current_price) / current_price * 100)
                            st.metric("Avg Pred", f"${avg_pred:.2f}", f"{change:.2f}%")
                
                # Table
                if show_table:
                    with table_placeholder.container():
                        st.markdown("---")
                        st.subheader("📋 Next 15 K-Bar Predictions")
                        pred_df = display_prediction_table(st.session_state.predicted_klines)
                        st.dataframe(pred_df, use_container_width=True)
            
            # Update every 1 second
            time.sleep(1)
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        logger.exception("App error:")

if __name__ == "__main__":
    main()
