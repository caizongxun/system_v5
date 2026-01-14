import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import torch
import logging

sys.path.insert(0, str(Path(__file__).parent))

from src.utils import load_config
from src.data_loader import DataLoader
from src.data_processor import DataProcessor
from src.model_pytorch import LSTMModel
from src.predictor import RollingPredictor

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
        device=device
    )
    
    model_path = Path(config['paths']['model_dir']) / "btc_15m_model_pytorch.pt"
    if model_path.exists():
        model.load(str(model_path))
        model.eval()
    else:
        st.error("Model file not found! Please train the model first.")
    
    return model, config, feature_columns, device

@st.cache_resource
def load_data():
    loader = DataLoader(
        repo_id="zongowo111/v2-crypto-ohlcv-data",
        cache_dir=Path("test/data")
    )
    df = loader.load_klines(
        symbol="BTCUSDT",
        timeframe="15m"
    )
    return df

def preprocess_data(df, processor, feature_columns):
    df = df.copy()
    
    if 'returns' not in df.columns:
        df_processed = processor.add_technical_indicators(df)
    else:
        df_processed = df
    
    normalized_data, scaler = processor.normalize_features(df_processed, feature_columns)
    return normalized_data, scaler, df_processed

def plot_klines(df_historical, predicted_klines):
    # Take last 100 K-bars as background
    df_plot = df_historical.tail(100).copy()
    
    # Convert time format
    if isinstance(df_plot['open_time'].iloc[0], str):
        df_plot['open_time'] = pd.to_datetime(df_plot['open_time'])
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=("BTC 15m K-Line with Predictions (Fixed Baseline)", "Volume")
    )
    
    # Plot historical K-bars
    fig.add_trace(
        go.Candlestick(
            x=df_plot['open_time'],
            open=df_plot['open'],
            high=df_plot['high'],
            low=df_plot['low'],
            close=df_plot['close'],
            name='Historical',
            visible=True
        ),
        row=1, col=1
    )
    
    # Plot predicted K-bars
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
                name='Predicted (Fixed Baseline)',
                visible=True
            ),
            row=1, col=1
        )
    
    # Plot volume
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
    
    # Update layout
    fig.update_layout(
        title="BTC/USDT 15M with Price Predictions (Fixed Baseline)",
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
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 BTC 15M Price Prediction Model (Fixed Baseline)")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        show_table = st.checkbox("Show Prediction Table", value=True)
        show_analysis = st.checkbox("Show Volatility Analysis", value=True)
        show_metrics = st.checkbox("Show Model Metrics", value=True)
        
        st.markdown("---")
        st.info("""
        **Fixed Baseline Mode:**
        Predictions are based on a fixed
        historical window (last 100 bars).
        This ensures consistent predictions
        regardless of current price changes.
        """)
    
    # Load model and data
    st.info("⏳ Loading model and data...")
    
    try:
        model, config, feature_columns, device = load_model_and_config()
        df = load_data()
        processor = DataProcessor()
        
        st.success("✅ Model and data loaded successfully!")
        
        # Preprocess
        normalized_data, scaler, df_processed = preprocess_data(df, processor, feature_columns)
        
        # Create predictor
        predictor = RollingPredictor(model, scaler, feature_columns, device)
        
        # Make predictions using fixed baseline (last 100 bars)
        st.info("🔮 Generating predictions...")
        X_baseline = normalized_data[-100:]
        predicted_klines = predictor.predict_with_fixed_baseline(X_baseline, df)
        st.success("✅ Predictions generated!")
        
        # Display chart
        st.subheader("📊 Price Chart with Predictions")
        fig = plot_klines(df, predicted_klines)
        st.plotly_chart(fig, use_container_width=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Current Price",
                f"${df.iloc[-1]['close']:.2f}",
                f"{((df.iloc[-1]['close'] - df.iloc[-2]['close']) / df.iloc[-2]['close'] * 100):.2f}%"
            )
        
        with col2:
            avg_pred = np.mean([k['close'] for k in predicted_klines])
            change = ((avg_pred - df.iloc[-1]['close']) / df.iloc[-1]['close'] * 100)
            st.metric(
                "Avg Predicted Price (15 bars)",
                f"${avg_pred:.2f}",
                f"{change:.2f}%"
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
        
        st.markdown("---")
        
        # Volatility analysis
        if show_analysis:
            st.subheader("📈 Volatility Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            # Current volatility
            current_vol = df['close'].pct_change().tail(20).std() * 100
            
            # Predicted volatility
            pred_returns = []
            for k in predicted_klines:
                ret = (k['close'] - k['open']) / k['open']
                pred_returns.append(ret)
            pred_vol = np.std(pred_returns) * 100
            
            # Price range
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
            st.markdown("---")
            st.subheader("📋 Detailed Predictions")
            pred_df = display_prediction_table(predicted_klines)
            st.dataframe(pred_df, use_container_width=True)
        
        # Model metrics
        if show_metrics:
            st.markdown("---")
            st.subheader("📊 Model Performance Metrics")
            
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
                st.metric("R² Score (Test)", "0.8334")
            
            with col2:
                st.metric("RMSE", "0.0887")
            
            with col3:
                st.metric("MAE", "0.0463")
        
        # Information sections
        st.markdown("---")
        st.subheader("ℹ️ Model Information")
        
        with st.expander("🏗️ Model Architecture"):
            st.write("""
            - **Architecture**: Multi-layer LSTM with 3 stacked layers
            - **Hidden Units**: [256, 128, 64]
            - **Input Shape**: (100 time steps, 16 features)
            - **Output Shape**: (15 time steps, 16 features)
            - **Prediction Mode**: Fixed baseline (last 100 bars remain constant)
            - **Device**: GPU (CUDA) if available
            """)
        
        with st.expander("📊 Feature Engineering"):
            st.write(f"""
            Total features: {len(feature_columns)}
            
            **Price Features:**
            - returns: Price percentage change
            - high_low_ratio: Intra-bar price range
            - open_close_ratio: Opening relative change
            
            **Trend Features:**
            - price_to_sma_10: Price vs 10-bar MA
            - price_to_sma_20: Price vs 20-bar MA
            - momentum_5: 5-bar rate of change
            - momentum_10: 10-bar rate of change
            
            **Volatility Features:**
            - volatility_20: 20-bar rolling std dev
            - volatility_5: 5-bar rolling std dev
            - ATR: Average True Range
            - returns_std_5: 5-bar returns volatility
            
            **Technical Indicators:**
            - RSI: Relative Strength Index (14)
            - MACD: Moving Average Convergence Divergence
            - BB_Position: Bollinger Bands position
            - Volume_Ratio: Volume vs 20-bar MA
            """)
        
        with st.expander("⚠️ Disclaimer"):
            st.warning("""
            **Important Disclaimer:**
            - This model is for educational and research purposes only
            - Do NOT use for actual trading without proper risk management
            - Past performance does NOT guarantee future results
            - Cryptocurrency markets are highly volatile and unpredictable
            - Always conduct your own research before investment decisions
            - Consider consulting with a financial advisor
            """)
        
        st.markdown("---")
        st.markdown(
            f"⏰ Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.write("Please make sure:")
        st.write("1. Model file exists at: test/models/btc_15m_model_pytorch.pt")
        st.write("2. Config file exists at: config/config.yaml")
        st.write("3. Model was trained with the new features")
        st.write("4. All dependencies are installed")
        logger.exception("App error:")

if __name__ == "__main__":
    main()
