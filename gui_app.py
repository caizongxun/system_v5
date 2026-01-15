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
import threading
import time

sys.path.insert(0, str(Path(__file__).parent))

from src.utils import load_config
from src.data_loader import DataLoader

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QSpinBox, QCheckBox, QPushButton, QTableWidget, QTableWidgetItem,
    QGridLayout, QGroupBox, QProgressBar, QComboBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtChart import QChart, QChartView, QCandlestickSeries, QCandlestickSet
from PyQt5.QtCore import QDate, QDateTime, QTime

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

def denormalize_features(pred_norm, scaler):
    return scaler.inverse_transform(pred_norm)

def features_to_klines(df_last, pred_denorm):
    pred_klines = []
    current_open = df_last['close'].iloc[-1]
    
    for i, features in enumerate(pred_denorm):
        returns = float(features[0])
        high_low_ratio = float(features[13])
        
        close_price = current_open * (1 + returns)
        high_price = close_price * (1 + high_low_ratio / 2)
        low_price = close_price * (1 - high_low_ratio / 2)
        
        volume = int(df_last['volume'].iloc[-1] * 0.5)
        
        pred_klines.append({
            'index': i + 1,
            'open': max(min(current_open, high_price), low_price),
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
        
        current_open = close_price
    
    return pred_klines

# ==================== Worker Thread ====================

class PredictionWorker(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    result_updated = pyqtSignal(dict)
    
    def __init__(self, model, scaler, feature_columns, device):
        super().__init__()
        self.model = model
        self.scaler = scaler
        self.feature_columns = feature_columns
        self.device = device
        self.running = True
    
    def run(self, refresh_interval=60):
        loader = DataLoader(
            repo_id='zongowo111/v2-crypto-ohlcv-data',
            cache_dir='test/data'
        )
        
        while self.running:
            try:
                df_raw = loader.load_klines(symbol='BTCUSDT', timeframe='15m')
                df_raw = df_raw.tail(200)
                
                if len(df_raw) < 100:
                    self.error.emit(f"Not enough data: {len(df_raw)} < 100")
                    time.sleep(refresh_interval)
                    continue
                
                df_processed = calculate_technical_indicators(df_raw)
                normalized_data = self.scaler.transform(
                    df_processed[self.feature_columns].values
                )
                
                X_pred = normalized_data[-100:]
                pred_norm = self.model.predict(X_pred)
                pred_denorm = denormalize_features(pred_norm, self.scaler)
                pred_klines = features_to_klines(df_raw, pred_denorm)
                
                current_price = float(df_raw['close'].iloc[-1])
                prev_price = float(df_raw['close'].iloc[-2])
                change_pct = ((current_price - prev_price) / prev_price * 100)
                
                self.result_updated.emit({
                    'timestamp': datetime.now(),
                    'price': current_price,
                    'change_pct': change_pct,
                    'klines': pred_klines,
                    'historical': df_raw
                })
                
                time.sleep(refresh_interval)
            
            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
                self.error.emit(f"Error: {str(e)}")
                time.sleep(refresh_interval)

# ==================== Main Window ====================

class PredictorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BTC 15M Attention-LSTM Predictor (Local GUI)")
        self.setGeometry(100, 100, 1400, 900)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.worker_thread = None
        self.worker = None
        
        self.init_ui()
        self.load_model()
        self.start_worker()
    
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
        self.table.setMaximumWidth(500)
        left_layout.addWidget(self.table)
        
        analysis_group = QGroupBox("Analysis")
        analysis_layout = QGridLayout()
        self.final_close_label = QLabel("Final: $0.00")
        self.highest_label = QLabel("High: $0.00")
        self.lowest_label = QLabel("Low: $0.00")
        self.range_label = QLabel("Range: $0.00")
        
        analysis_layout.addWidget(self.final_close_label, 0, 0)
        analysis_layout.addWidget(self.highest_label, 0, 1)
        analysis_layout.addWidget(self.lowest_label, 1, 0)
        analysis_layout.addWidget(self.range_label, 1, 1)
        
        analysis_group.setLayout(analysis_layout)
        left_layout.addWidget(analysis_group)
        left_layout.addStretch()
        
        content_layout.addLayout(left_layout)
        
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Candlestick Chart (Last 100 + 6 Predictions):"))
        
        self.status_bar = QLabel("Waiting for data...")
        right_layout.addWidget(self.status_bar)
        
        content_layout.addLayout(right_layout)
        
        main_layout.addLayout(content_layout)
        
        central_widget.setLayout(main_layout)
    
    def load_model(self):
        try:
            self.status_label.setText("Status: Loading model...")
            
            feature_columns = [
                'returns', 'log_returns',
                'volatility_20', 'volatility_5',
                'atr', 'rsi', 'macd', 'macd_signal', 'macd_diff',
                'sma_7', 'sma_14', 'sma_21', 'kama',
                'high_low_ratio', 'open_close_ratio',
                'price_to_sma_10', 'price_to_sma_20'
            ]
            
            self.model = AttentionLSTM(
                input_size=17,
                hidden_size=256,
                num_layers=2,
                forecast_horizon=6,
                num_heads=4,
                dropout_rate=0.3,
                l2_reg=0.0001,
                device=self.device
            )
            
            model_path = Path('test/models/attention_lstm.pt')
            self.model.load(str(model_path))
            
            with open(Path('test/models/scaler.pkl'), 'rb') as f:
                self.scaler = pickle.load(f)
            
            try:
                with open(Path('test/models/feature_columns.pkl'), 'rb') as f:
                    self.feature_columns = pickle.load(f)
            except:
                self.feature_columns = feature_columns
            
            self.status_label.setText(f"Status: Ready (Device: {self.device})")
            logger.info("Model loaded successfully")
        
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            logger.error(f"Failed to load model: {str(e)}")
    
    def start_worker(self):
        if self.model is None:
            return
        
        self.worker = PredictionWorker(
            self.model,
            self.scaler,
            self.feature_columns,
            self.device
        )
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)
        
        self.worker_thread.started.connect(
            lambda: self.worker.run(self.refresh_spin.value())
        )
        self.worker.result_updated.connect(self.update_results)
        self.worker.error.connect(self.handle_error)
        
        self.worker_thread.start()
    
    def update_results(self, data):
        timestamp = data['timestamp']
        price = data['price']
        change_pct = data['change_pct']
        klines = data['klines']
        
        self.time_label.setText(f"Time: {timestamp.strftime('%H:%M:%S')}")
        self.price_label.setText(f"Price: ${price:,.2f}")
        color = "green" if change_pct >= 0 else "red"
        self.change_label.setText(f"Change: {change_pct:+.3f}%")
        self.change_label.setStyleSheet(f"color: {color};")
        
        self.table.setRowCount(len(klines))
        for i, kline in enumerate(klines):
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
        
        if len(klines) > 0:
            closes = [k['close'] for k in klines]
            highs = [k['high'] for k in klines]
            lows = [k['low'] for k in klines]
            
            self.final_close_label.setText(f"Final: ${closes[-1]:.2f}")
            self.highest_label.setText(f"High: ${max(highs):.2f}")
            self.lowest_label.setText(f"Low: ${min(lows):.2f}")
            self.range_label.setText(f"Range: ${max(highs) - min(lows):.2f}")
        
        self.status_bar.setText(f"Updated: {timestamp.strftime('%H:%M:%S')} - {len(klines)} predictions")
    
    def handle_error(self, error_msg):
        self.status_label.setText(f"Error: {error_msg}")
        logger.error(error_msg)
    
    def closeEvent(self, event):
        if self.worker:
            self.worker.running = False
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PredictorWindow()
    window.show()
    sys.exit(app.exec_())
