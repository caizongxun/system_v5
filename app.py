import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import pickle
import time

sys.path.insert(0, str(Path(__file__).parent))

from src.utils import load_config
from src.data_loader import DataLoader

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== Model Architecture ====================

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0
        
        self.head_dim = hidden_size // num_heads
        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)
        
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.hidden_size)
        
        output = self.fc_out(context)
        return output, attn_weights

class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, forecast_horizon, 
                 num_heads=4, dropout_rate=0.3, l2_reg=0.0, device='cpu'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        self.device = device
        self.l2_reg = l2_reg
        self.num_heads = num_heads
        
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=False
        )
        
        self.encoder_ln = nn.LayerNorm(hidden_size)
        self.attention = MultiHeadAttention(hidden_size, num_heads=num_heads, dropout=dropout_rate)
        self.attention_ln = nn.LayerNorm(hidden_size)
        
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=False
        )
        
        self.decoder_ln = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, input_size)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        encoder_out, (h_n, c_n) = self.encoder(x)
        encoder_out = self.encoder_ln(encoder_out)
        
        attn_out, attn_weights = self.attention(
            query=encoder_out,
            key=encoder_out,
            value=encoder_out
        )
        attn_out = self.attention_ln(attn_out)
        encoder_out = encoder_out + attn_out * 0.5
        
        decoder_input = encoder_out[:, -1:, :]
        predictions = []
        
        for _ in range(self.forecast_horizon):
            decoder_out, (h_n, c_n) = self.decoder(decoder_input, (h_n, c_n))
            decoder_out = self.decoder_ln(decoder_out)
            decoder_out = self.dropout(decoder_out)
            output = self.fc(decoder_out)
            predictions.append(output)
            decoder_input = decoder_out
        
        return torch.cat(predictions, dim=1), attn_weights
    
    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.eval()
    
    def predict(self, X):
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(self.device)
            y_pred, _ = self(X_tensor)
            return y_pred.cpu().numpy()[0]

# ==================== Technical Indicators ====================

def calculate_technical_indicators(df):
    df = df.copy()
    
    df['returns'] = df['close'].pct_change().fillna(0)
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
    df['volatility_20'] = df['returns'].rolling(window=20).std().fillna(0)
    df['volatility_5'] = df['returns'].rolling(window=5).std().fillna(0)
    
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean().fillna(0)
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].fillna(50)
    
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']
    
    df['sma_7'] = df['close'].rolling(window=7).mean().bfill().ffill()
    df['sma_14'] = df['close'].rolling(window=14).mean().bfill().ffill()
    df['sma_21'] = df['close'].rolling(window=21).mean().bfill().ffill()
    
    period = 10
    fastsc = 2 / (2 + 1)
    slowsc = 2 / (30 + 1)
    change = abs(df['close'].diff(period))
    volatility = df['close'].diff().abs().rolling(period).sum()
    efficiency_ratio = change / (volatility + 1e-8)
    smoothing_constant = efficiency_ratio * (fastsc - slowsc) + slowsc
    
    kama = [0] * len(df)
    for i in range(period, len(df)):
        if i == period:
            kama[i] = df['close'].iloc[i]
        else:
            kama[i] = kama[i-1] + (smoothing_constant.iloc[i] ** 2) * (df['close'].iloc[i] - kama[i-1])
    df['kama'] = kama
    
    df['high_low_ratio'] = (df['high'] - df['low']) / (df['close'] + 1e-8)
    df['open_close_ratio'] = (df['close'] - df['open']) / (df['open'] + 1e-8)
    df['price_to_sma_10'] = df['close'] / (df['sma_7'] + 1e-8)
    df['price_to_sma_20'] = df['close'] / (df['sma_14'] + 1e-8)
    
    df = df.bfill().ffill()
    return df

# ==================== Cache and Loading ====================

@st.cache_resource
def load_model_and_scaler():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        feature_columns = [
            'returns', 'log_returns',
            'volatility_20', 'volatility_5',
            'atr', 'rsi', 'macd', 'macd_signal', 'macd_diff',
            'sma_7', 'sma_14', 'sma_21', 'kama',
            'high_low_ratio', 'open_close_ratio',
            'price_to_sma_10', 'price_to_sma_20'
        ]
        
        model = AttentionLSTM(
            input_size=17,
            hidden_size=256,
            num_layers=2,
            forecast_horizon=6,
            num_heads=4,
            dropout_rate=0.3,
            l2_reg=0.0001,
            device=device
        )
        
        model_path = Path('test/models/attention_lstm.pt')
        if model_path.exists():
            model.load(str(model_path))
            logger.info(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        scaler_path = Path('test/models/scaler.pkl')
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            logger.info(f"Scaler loaded from {scaler_path}")
        else:
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        
        features_path = Path('test/models/feature_columns.pkl')
        if features_path.exists():
            with open(features_path, 'rb') as f:
                loaded_features = pickle.load(f)
            feature_columns = loaded_features
            logger.info(f"Loaded {len(feature_columns)} feature columns")
        
        return model, scaler, feature_columns, device
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def load_realtime_data(lookback=200):
    try:
        logger.info("Loading BTCUSDT data from HuggingFace...")
        loader = DataLoader(
            repo_id='zongowo111/v2-crypto-ohlcv-data',
            cache_dir='test/data'
        )
        
        df = loader.load_klines(
            symbol='BTCUSDT',
            timeframe='15m'
        )
        
        if len(df) < lookback:
            logger.warning(f"Not enough data: {len(df)} < {lookback}")
            return df.tail(len(df))
        
        return df.tail(lookback)
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def denormalize_features(pred_norm, scaler):
    return scaler.inverse_transform(pred_norm)

def features_to_klines(df_last, pred_denorm, lookback=100):
    pred_klines = []
    current_open = df_last['close'].iloc[-1]
    
    for i, features in enumerate(pred_denorm):
        returns = float(features[0])
        high_low_ratio = float(features[13])
        open_close_ratio = float(features[14])
        volume_ratio = 0.5
        
        close_price = current_open * (1 + returns)
        high_price = close_price * (1 + high_low_ratio / 2)
        low_price = close_price * (1 - high_low_ratio / 2)
        
        volume = int(df_last['volume'].iloc[-1] * volume_ratio)
        
        pred_klines.append({
            'index': i + 1,
            'time': datetime.now() + timedelta(minutes=15 * (i + 1)),
            'open': max(min(current_open, high_price), low_price),
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
        
        current_open = close_price
    
    return pred_klines

def plot_klines_chart(df_historical, predicted_klines):
    try:
        df_plot = df_historical.tail(100).copy()
        
        if 'open_time' in df_plot.columns:
            if isinstance(df_plot['open_time'].iloc[0], str):
                df_plot['open_time'] = pd.to_datetime(df_plot['open_time'])
        else:
            df_plot['open_time'] = pd.date_range(
                end=datetime.now(),
                periods=len(df_plot),
                freq='15min'
            )
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
            subplot_titles=("BTC/USDT 15M - Realtime Predictions", "Volume")
        )
        
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
        
        if predicted_klines and len(predicted_klines) > 0:
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
                    name='Predictions (6 bars)',
                    increasing=dict(fillcolor='cyan', line=dict(color='cyan')),
                    decreasing=dict(fillcolor='orange', line=dict(color='orange'))
                ),
                row=1, col=1
            )
            
            pred_volumes = [float(k['volume']) for k in predicted_klines]
            fig.add_trace(
                go.Bar(
                    x=pred_times,
                    y=pred_volumes,
                    name='Pred Volume',
                    marker=dict(color='rgba(0, 255, 255, 0.5)'),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        fig.add_trace(
            go.Bar(
                x=df_plot['open_time'],
                y=df_plot['volume'],
                name='Historical Volume',
                marker=dict(color='rgba(128, 128, 128, 0.3)'),
                showlegend=False
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title="BTC/USDT 15M Real-Time Chart with Attention-LSTM Predictions",
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
    data = []
    for k in predicted_klines:
        data.append({
            'Step': k['index'],
            'Time': k['time'].strftime('%H:%M'),
            'Open': f"${k['open']:.2f}",
            'High': f"${k['high']:.2f}",
            'Low': f"${k['low']:.2f}",
            'Close': f"${k['close']:.2f}",
            'Change%': f"{((k['close'] - k['open']) / k['open'] * 100):.2f}%",
            'Volume': f"{k['volume']/1e6:.2f}M"
        })
    return pd.DataFrame(data)

def main():
    st.set_page_config(
        page_title="BTC 15M Attention-LSTM Predictor",
        page_icon="Chart",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("BTC 15M Real-Time Price Predictor")
    st.markdown("Attention-LSTM Model - Predicting next 6 bars from last 100 completed candles")
    st.markdown("---")
    
    with st.sidebar:
        st.header("Settings")
        
        refresh_interval = st.slider(
            "Refresh Interval (seconds)",
            min_value=30,
            max_value=300,
            value=60,
            step=30
        )
        
        show_table = st.checkbox("Show Prediction Table", value=True)
        show_metrics = st.checkbox("Show Metrics", value=True)
        
        st.markdown("---")
        st.info("""
        Model: Attention-LSTM
        Input: Last 100 completed candles
        Output: Next 6 candles
        Data: HuggingFace (BTCUSDT 15m)
        Updated: 2026-01-15
        """)
    
    if 'last_update' not in st.session_state:
        st.session_state.last_update = 0
        st.session_state.predicted_klines = []
        st.session_state.df_current = None
        st.session_state.model_loaded = False
    
    if not st.session_state.model_loaded:
        try:
            with st.spinner("Loading Attention-LSTM model..."):
                model, scaler, feature_columns, device = load_model_and_scaler()
                st.session_state.model = model
                st.session_state.scaler = scaler
                st.session_state.feature_columns = feature_columns
                st.session_state.device = device
                st.session_state.model_loaded = True
                st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            logger.error(f"Model loading error: {str(e)}")
            return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        time_placeholder = st.empty()
    with col2:
        price_placeholder = st.empty()
    with col3:
        updated_placeholder = st.empty()
    with col4:
        pred_count_placeholder = st.empty()
    
    chart_placeholder = st.empty()
    table_placeholder = st.empty()
    metrics_placeholder = st.empty()
    
    current_time = time.time()
    should_update = (current_time - st.session_state.last_update) >= refresh_interval or st.session_state.df_current is None
    
    if should_update:
        try:
            logger.info("Fetching data and generating predictions...")
            
            df_raw = load_realtime_data(lookback=200)
            
            if len(df_raw) < 100:
                st.error(f"Not enough data: {len(df_raw)} < 100")
                st.stop()
            
            logger.info(f"Loaded {len(df_raw)} candles")
            
            df_processed = calculate_technical_indicators(df_raw)
            
            normalized_data = st.session_state.scaler.transform(
                df_processed[st.session_state.feature_columns].values
            )
            
            st.session_state.df_current = df_raw.copy()
            
            logger.info("Generating predictions using last 100 completed candles...")
            X_pred = normalized_data[-100:]
            
            pred_norm = st.session_state.model.predict(X_pred)
            pred_denorm = denormalize_features(pred_norm, st.session_state.scaler)
            
            st.session_state.predicted_klines = features_to_klines(
                df_raw,
                pred_denorm
            )
            
            logger.info(f"Generated {len(st.session_state.predicted_klines)} predictions")
            
            st.session_state.last_update = current_time
        
        except Exception as e:
            logger.exception(f"Error during update: {str(e)}")
            st.error(f"Update failed: {str(e)}")
    
    if st.session_state.df_current is not None:
        price = st.session_state.df_current['close'].iloc[-1]
        prev_price = st.session_state.df_current['close'].iloc[-2]
        change = ((price - prev_price) / prev_price * 100)
        
        with time_placeholder:
            st.metric("Current Time", datetime.now().strftime('%H:%M:%S'))
        with price_placeholder:
            st.metric("BTC Price", f"${price:,.2f}", f"{change:+.3f}%")
        with updated_placeholder:
            seconds_ago = int(time.time() - st.session_state.last_update)
            st.metric("Last Update", f"{seconds_ago}s ago")
        with pred_count_placeholder:
            st.metric("Predictions", len(st.session_state.predicted_klines))
    
    st.markdown("---")
    
    with chart_placeholder.container():
        st.subheader("Chart with Predictions")
        
        if st.session_state.df_current is not None:
            fig = plot_klines_chart(
                st.session_state.df_current,
                st.session_state.predicted_klines
            )
            
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Chart generation failed")
        else:
            st.warning("Waiting for data...")
    
    if show_table and st.session_state.predicted_klines:
        with table_placeholder.container():
            st.markdown("---")
            st.subheader("Next 6 Candle Predictions")
            
            pred_df = display_prediction_table(st.session_state.predicted_klines)
            st.dataframe(pred_df, use_container_width=True, hide_index=True)
    
    if show_metrics and st.session_state.df_current is not None and st.session_state.predicted_klines:
        with metrics_placeholder.container():
            st.markdown("---")
            st.subheader("Prediction Analysis")
            
            col1, col2, col3, col4 = st.columns(4)
            
            current_price = st.session_state.df_current['close'].iloc[-1]
            pred_closes = [k['close'] for k in st.session_state.predicted_klines]
            pred_opens = [k['open'] for k in st.session_state.predicted_klines]
            pred_highs = [k['high'] for k in st.session_state.predicted_klines]
            pred_lows = [k['low'] for k in st.session_state.predicted_klines]
            
            final_close = pred_closes[-1]
            highest = max(pred_highs)
            lowest = min(pred_lows)
            
            with col1:
                price_change = ((final_close - current_price) / current_price * 100)
                st.metric(
                    "Expected Final Close",
                    f"${final_close:.2f}",
                    f"{price_change:+.2f}%"
                )
            
            with col2:
                st.metric(
                    "Highest Predicted",
                    f"${highest:.2f}",
                    f"{((highest - current_price) / current_price * 100):+.2f}%"
                )
            
            with col3:
                st.metric(
                    "Lowest Predicted",
                    f"${lowest:.2f}",
                    f"{((lowest - current_price) / current_price * 100):+.2f}%"
                )
            
            with col4:
                pred_range = highest - lowest
                st.metric(
                    "Predicted Range",
                    f"${pred_range:.2f}",
                    f"{(pred_range / current_price * 100):.2f}%"
                )
    
    time.sleep(1)
    st.rerun()

if __name__ == "__main__":
    main()
