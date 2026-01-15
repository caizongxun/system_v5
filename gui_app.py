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
import warnings
import traceback

warnings.filterwarnings('ignore', category=UserWarning, message='.*feature names.*')

sys.path.insert(0, str(Path(__file__).parent))

from src.utils import load_config
from src.data_loader import DataLoader

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QSpinBox, QCheckBox, QTableWidget, QTableWidgetItem,
    QGridLayout, QGroupBox, QScrollArea
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QUrl
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtWebEngineWidgets import QWebEngineView
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint)
        self.to(self.device)
        self.eval()
    
    def predict(self, X):
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(self.device)
            y_pred, _ = self(X_tensor)
            return y_pred.cpu().numpy()[0]

# ==================== Technical Indicators ====================

def calculate_technical_indicators(df):
    """直接使用價格和技術指標，不用 returns"""
    df = df.copy()
    
    df['volatility_20'] = df['close'].pct_change().rolling(window=20).std().fillna(0)
    df['volatility_5'] = df['close'].pct_change().rolling(window=5).std().fillna(0)
    
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

def prepare_features_with_price_context(df_processed, feature_columns):
    """加入價格上下文特徵，幫助模型理解絕對價格水平"""
    df = df_processed.copy()
    
    # 價格水平特徵
    df['price_level_high'] = df['high'] / df['high'].max()
    df['price_level_low'] = df['low'] / df['low'].min()
    df['price_range_pct'] = (df['high'] - df['low']) / df['close'] * 100
    
    # 動量特徵（替代 returns）
    df['price_momentum_5'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5) * 100
    df['price_momentum_10'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10) * 100
    
    # 趨勢強度
    df['sma_distance'] = (df['close'] - df['sma_14']) / df['sma_14'] * 100
    
    enhanced_cols = feature_columns + [
        'price_level_high', 'price_level_low', 'price_range_pct',
        'price_momentum_5', 'price_momentum_10', 'sma_distance'
    ]
    return df[enhanced_cols].values, enhanced_cols

def denormalize_prices(pred_norm, scaler, price_min, price_max):
    """反標準化並約束在合理範圍內"""
    pred_denorm = scaler.inverse_transform(pred_norm)
    
    # 約束價格在歷史範圍內
    pred_denorm = np.clip(pred_denorm, price_min * 0.95, price_max * 1.05)
    
    return pred_denorm

def features_to_klines_with_price_guidance(df_last, pred_denorm, lookback_df):
    """使用反標準化的特徵重建 K 線，帶有價格指導"""
    pred_klines = []
    current_open = df_last['close'].iloc[-1]
    price_mean = lookback_df['close'].mean()
    price_std = lookback_df['close'].std()
    
    logger.info(f"Price guidance: mean={price_mean:.2f}, std={price_std:.2f}")
    
    for i, features in enumerate(pred_denorm):
        momentum_5 = float(features[-3]) if len(features) > 15 else 0
        momentum_10 = float(features[-2]) if len(features) > 16 else 0
        sma_distance = float(features[-1]) if len(features) > 17 else 0
        
        # 使用動量計算預期價格變化
        expected_change_pct = (momentum_5 + momentum_10) / 2
        close_price = current_open * (1 + expected_change_pct / 100)
        
        # 約束在合理範圍內
        close_price = np.clip(close_price, price_mean * 0.98, price_mean * 1.02)
        
        # 隨機高低點
        volatility = float(features[1]) * 100 if len(features) > 1 else 0.5
        volatility = np.clip(volatility, 0.01, 2.0)
        
        range_pct = volatility / 100
        high_price = close_price * (1 + range_pct / 2)
        low_price = close_price * (1 - range_pct / 2)
        
        volume = int(df_last['volume'].iloc[-1] * 0.8)
        
        kline = {
            'index': i + 1,
            'open': max(min(current_open, high_price), low_price),
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        }
        
        logger.info(f"K線 {i+1}: O={kline['open']:.2f}, H={kline['high']:.2f}, "
                   f"L={kline['low']:.2f}, C={kline['close']:.2f}, "
                   f"Change={expected_change_pct:+.3f}%")
        
        pred_klines.append(kline)
        current_open = close_price
    
    return pred_klines

def create_candlestick_chart(df_historical, predicted_klines):
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
            row_heights=[0.7, 0.3]
        )
        
        fig.add_trace(
            go.Candlestick(
                x=df_plot['open_time'],
                open=df_plot['open'],
                high=df_plot['high'],
                low=df_plot['low'],
                close=df_plot['close'],
                name='Historical',
                increasing=dict(fillcolor='green', line=dict(color='green')),
                decreasing=dict(fillcolor='red', line=dict(color='red'))
            ),
            row=1, col=1
        )
        
        if predicted_klines and len(predicted_klines) > 0:
            pred_times = [datetime.now() + timedelta(minutes=15*(i+1)) for i in range(len(predicted_klines))]
            pred_opens = [float(k['open']) for k in predicted_klines]
            pred_highs = [float(k['high']) for k in predicted_klines]
            pred_lows = [float(k['low']) for k in predicted_klines]
            pred_closes = [float(k['close']) for k in predicted_klines]
            pred_volumes = [float(k['volume']) for k in predicted_klines]
            
            fig.add_trace(
                go.Candlestick(
                    x=pred_times,
                    open=pred_opens,
                    high=pred_highs,
                    low=pred_lows,
                    close=pred_closes,
                    name='Predictions',
                    increasing=dict(fillcolor='cyan', line=dict(color='cyan')),
                    decreasing=dict(fillcolor='orange', line=dict(color='orange'))
                ),
                row=1, col=1
            )
            
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
            height=600,
            hovermode='x unified',
            template='plotly_dark',
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig.to_html(include_plotlyjs='cdn')
    
    except Exception as e:
        logger.error(f"Error creating chart: {str(e)}")
        logger.error(traceback.format_exc())
        return None

# ==================== Main Window ====================

class PredictorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BTC 15M Attention-LSTM Predictor (Direct Price Forecasting)")
        self.setGeometry(100, 100, 1600, 1000)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.loader = None
        self.last_data = None
        self.predicted_klines = []
        
        logger.info(f"Initializing GUI with device: {self.device}")
        
        self.init_ui()
        self.load_model()
        self.setup_timer()
        self.initial_update()
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
        
        top_layout = QHBoxLayout()
        self.time_label = QLabel("Time: --:--:--")
        self.price_label = QLabel("Price: $0.00")
        self.change_label = QLabel("Change: 0.00%")
        self.status_label = QLabel("Status: Loading...")
        
        font = QFont()
        font.setPointSize(11)
        font.setBold(True)
        
        for label in [self.time_label, self.price_label, self.change_label, self.status_label]:
            label.setFont(font)
            top_layout.addWidget(label)
        
        main_layout.addLayout(top_layout)
        main_layout.addSpacing(10)
        
        settings_layout = QHBoxLayout()
        settings_layout.addWidget(QLabel("Refresh (s):"))
        self.refresh_spin = QSpinBox()
        self.refresh_spin.setRange(30, 300)
        self.refresh_spin.setValue(60)
        self.refresh_spin.setSingleStep(30)
        settings_layout.addWidget(self.refresh_spin)
        settings_layout.addSpacing(20)
        
        self.show_table_check = QCheckBox("Show Predictions")
        self.show_table_check.setChecked(True)
        settings_layout.addWidget(self.show_table_check)
        
        settings_layout.addStretch()
        main_layout.addLayout(settings_layout)
        
        content_layout = QHBoxLayout()
        
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Recent 6 Predictions:"))
        
        self.table = QTableWidget(6, 7)
        self.table.setHorizontalHeaderLabels(
            ["Step", "Open", "High", "Low", "Close", "Change%", "Volume"]
        )
        self.table.setMaximumWidth(550)
        left_layout.addWidget(self.table)
        
        analysis_group = QGroupBox("Analysis")
        analysis_layout = QGridLayout(analysis_group)
        self.final_close_label = QLabel("Final: $0.00")
        self.highest_label = QLabel("High: $0.00")
        self.lowest_label = QLabel("Low: $0.00")
        self.range_label = QLabel("Range: $0.00")
        
        analysis_layout.addWidget(self.final_close_label, 0, 0)
        analysis_layout.addWidget(self.highest_label, 0, 1)
        analysis_layout.addWidget(self.lowest_label, 1, 0)
        analysis_layout.addWidget(self.range_label, 1, 1)
        
        left_layout.addWidget(analysis_group)
        left_layout.addStretch()
        
        content_layout.addLayout(left_layout, 1)
        
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Candlestick Chart (Last 100 + 6 Predictions):"))
        
        self.chart_view = QWebEngineView()
        right_layout.addWidget(self.chart_view)
        
        self.status_bar = QLabel("Waiting for data...")
        right_layout.addWidget(self.status_bar)
        
        content_layout.addLayout(right_layout, 3)
        
        main_layout.addLayout(content_layout)
        
        central_widget.setLayout(main_layout)
    
    def load_model(self):
        try:
            logger.info("Loading Attention-LSTM model...")
            self.status_label.setText("Status: Loading model...")
            
            feature_columns = [
                'volatility_20', 'volatility_5',
                'atr', 'rsi', 'macd', 'macd_signal', 'macd_diff',
                'sma_7', 'sma_14', 'sma_21', 'kama',
                'high_low_ratio', 'open_close_ratio',
                'price_to_sma_10', 'price_to_sma_20'
            ]
            
            self.model = AttentionLSTM(
                input_size=21,
                hidden_size=256,
                num_layers=2,
                forecast_horizon=6,
                num_heads=4,
                dropout_rate=0.3,
                l2_reg=0.0001,
                device=self.device
            )
            
            model_path = Path('test/models/attention_lstm.pt')
            logger.info(f"Loading model from {model_path}")
            self.model.load(str(model_path))
            
            with open(Path('test/models/scaler.pkl'), 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info("Scaler loaded")
            
            try:
                with open(Path('test/models/feature_columns.pkl'), 'rb') as f:
                    self.feature_columns = pickle.load(f)
                logger.info(f"Loaded {len(self.feature_columns)} feature columns from file")
            except:
                self.feature_columns = feature_columns
                logger.info("Using default feature columns")
            
            logger.info("Initializing DataLoader...")
            self.loader = DataLoader(
                repo_id='zongowo111/v2-crypto-ohlcv-data',
                cache_dir='test/data'
            )
            
            self.status_label.setText(f"Status: Ready (Device: {self.device})")
            logger.info("Model loaded successfully")
        
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            logger.error(traceback.format_exc())
            self.status_label.setText(f"Error: {str(e)}")
    
    def setup_timer(self):
        logger.info(f"Setting up timers with {self.refresh_spin.value()}s refresh interval")
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_predictions)
        self.timer.start(self.refresh_spin.value() * 1000)
        
        self.clock_timer = QTimer()
        self.clock_timer.timeout.connect(self.update_clock)
        self.clock_timer.start(1000)
    
    def initial_update(self):
        logger.info("Performing initial data update...")
        self.update_predictions()
    
    def update_clock(self):
        current_time = datetime.now().strftime('%H:%M:%S')
        self.time_label.setText(f"Time: {current_time}")
    
    def update_predictions(self):
        try:
            logger.info("Fetching data from HuggingFace...")
            df_raw = self.loader.load_klines(symbol='BTCUSDT', timeframe='15m')
            logger.info(f"Loaded {len(df_raw)} candles")
            
            df_raw = df_raw.tail(200)
            
            if len(df_raw) < 100:
                msg = f"Not enough data: {len(df_raw)} < 100"
                logger.warning(msg)
                self.status_bar.setText(msg)
                return
            
            logger.info("Computing technical indicators...")
            df_processed = calculate_technical_indicators(df_raw)
            
            logger.info("Preparing features with price context...")
            feature_data, enhanced_cols = prepare_features_with_price_context(
                df_processed, self.feature_columns
            )
            
            logger.info(f"Feature shape: {feature_data.shape}, cols: {len(enhanced_cols)}")
            
            logger.info("Normalizing features...")
            normalized_data = self.scaler.transform(feature_data)
            
            logger.info("Making predictions...")
            X_pred = normalized_data[-100:]
            pred_norm = self.model.predict(X_pred)
            
            price_min = df_raw['close'].min()
            price_max = df_raw['close'].max()
            pred_denorm = denormalize_prices(pred_norm, self.scaler, price_min, price_max)
            
            self.predicted_klines = features_to_klines_with_price_guidance(
                df_raw, pred_denorm, df_raw
            )
            logger.info(f"Generated {len(self.predicted_klines)} predictions")
            
            current_price = float(df_raw['close'].iloc[-1])
            prev_price = float(df_raw['close'].iloc[-2])
            change_pct = ((current_price - prev_price) / prev_price * 100)
            
            logger.info(f"Current price: ${current_price:.2f}, Change: {change_pct:+.3f}%")
            
            self.price_label.setText(f"Price: ${current_price:,.2f}")
            color = "green" if change_pct >= 0 else "red"
            self.change_label.setText(f"Change: {change_pct:+.3f}%")
            self.change_label.setStyleSheet(f"color: {color};")
            
            self.update_table()
            self.update_chart(df_raw)
            self.update_analysis()
            
            self.status_bar.setText(f"Updated: {datetime.now().strftime('%H:%M:%S')} - {len(self.predicted_klines)} predictions")
            self.last_data = df_raw
            logger.info("Update completed successfully")
        
        except Exception as e:
            logger.error(f"Update error: {str(e)}")
            logger.error(traceback.format_exc())
            self.status_label.setText(f"Error: {str(e)}")
            self.status_bar.setText(f"Error: {str(e)}")
    
    def update_table(self):
        logger.info(f"Updating table with {len(self.predicted_klines)} rows")
        self.table.setRowCount(len(self.predicted_klines))
        for i, kline in enumerate(self.predicted_klines):
            step = QTableWidgetItem(str(kline['index']))
            open_item = QTableWidgetItem(f"${kline['open']:.2f}")
            high_item = QTableWidgetItem(f"${kline['high']:.2f}")
            low_item = QTableWidgetItem(f"${kline['low']:.2f}")
            close_item = QTableWidgetItem(f"${kline['close']:.2f}")
            
            change = ((kline['close'] - kline['open']) / kline['open'] * 100)
            change_item = QTableWidgetItem(f"{change:+.2f}%")
            change_color = "green" if change >= 0 else "red"
            change_item.setForeground(QColor(change_color))
            
            vol_item = QTableWidgetItem(f"{kline['volume']/1e6:.2f}M")
            
            self.table.setItem(i, 0, step)
            self.table.setItem(i, 1, open_item)
            self.table.setItem(i, 2, high_item)
            self.table.setItem(i, 3, low_item)
            self.table.setItem(i, 4, close_item)
            self.table.setItem(i, 5, change_item)
            self.table.setItem(i, 6, vol_item)
    
    def update_chart(self, df_historical):
        logger.info("Generating candlestick chart...")
        html_content = create_candlestick_chart(df_historical, self.predicted_klines)
        if html_content:
            logger.info("Chart generated successfully, rendering...")
            self.chart_view.setHtml(html_content)
        else:
            logger.warning("Failed to generate chart")
    
    def update_analysis(self):
        if len(self.predicted_klines) > 0:
            closes = [k['close'] for k in self.predicted_klines]
            highs = [k['high'] for k in self.predicted_klines]
            lows = [k['low'] for k in self.predicted_klines]
            
            self.final_close_label.setText(f"Final: ${closes[-1]:.2f}")
            self.highest_label.setText(f"High: ${max(highs):.2f}")
            self.lowest_label.setText(f"Low: ${min(lows):.2f}")
            self.range_label.setText(f"Range: ${max(highs) - min(lows):.2f}")
            logger.info(f"Analysis updated: Final=${closes[-1]:.2f}, High=${max(highs):.2f}, Low=${min(lows):.2f}")
    
    def closeEvent(self, event):
        logger.info("Closing application...")
        self.timer.stop()
        self.clock_timer.stop()
        event.accept()

if __name__ == '__main__':
    logger.info("Starting BTC Predictor GUI...")
    app = QApplication(sys.argv)
    window = PredictorWindow()
    window.show()
    logger.info("GUI window displayed")
    sys.exit(app.exec_())
