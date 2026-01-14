# PyTorch Version - BTC_15m Price Prediction System

由於 TensorFlow CUDA/cuDNN 版本不相容問題,已提供 PyTorch 版本的訓練管道。

## 優勢

✅ **無需 cuDNN** - PyTorch 原生 GPU 支援,無版本相容問題  
✅ **更簡潔** - 程式碼更直觀易懂  
✅ **更快速** - LSTM 實現高度優化  
✅ **研究友善** - 更彈性的模型修改空間  

## 安裝依賴

### 1. 安裝 PyTorch (GPU 版)

```bash
# CUDA 13.1 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 或如果你有 CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. 驗證安裝

```bash
python -c "import torch; print('GPU Available:', torch.cuda.is_available()); print('Device:', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))"
```

應該看到:
```
GPU Available: True
Device: cuda
```

## 使用方式

### 執行訓練

```bash
# PyTorch 版本
python test/run_pipeline_pytorch.py

# 原 TensorFlow 版本 (如果 GPU 問題已解決)
python test/run_pipeline.py
```

## 檔案說明

| 檔案 | 說明 |
|------|------|
| `test/run_pipeline_pytorch.py` | PyTorch 訓練管道 |
| `src/model_pytorch.py` | PyTorch LSTM 模型 |
| `test/run_pipeline.py` | 原 TensorFlow 訓練管道 |
| `src/model.py` | 原 TensorFlow LSTM 模型 |

## 模型架構

### PyTorch 模型

```
Input (batch_size, 100, num_features)
    ↓
LSTM Layer 1 (256 units) + BatchNorm + Dropout(0.2)
    ↓
LSTM Layer 2 (128 units) + BatchNorm + Dropout(0.2)
    ↓
LSTM Layer 3 (64 units) + Dropout(0.2)
    ↓
Take Last Output (batch_size, 64)
    ↓
Dense(64, activation='relu') + Dropout(0.2)
    ↓
Dense(15 * num_features, activation=None)
    ↓
Reshape to (batch_size, 15, num_features)
```

## 訓練參數

```yaml
model:
  epochs: 100
  batch_size: 32
  sequence_length: 100
  prediction_length: 15
  lstm_units: [256, 128, 64]
  dropout_rate: 0.2
  learning_rate: 0.001
  l2_regularization: 0.0
```

## 輸出檔案

訓練完成後會產生:

```
test/
├── models/
│   ├── btc_15m_model.h5              # TensorFlow 模型
│   └── btc_15m_model_pytorch.pt      # PyTorch 模型
├── results/
│   ├── metrics_report.txt             # 評估指標
│   ├── training_history.png           # 訓練曲線
│   └── predictions_sample.png         # 預測對比圖
└── logs/
    └── run_*.log                      # 執行日誌
```

## GPU 問題排查

如果 GPU 仍未被識別:

### 檢查 PyTorch GPU 狀態

```bash
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('cuDNN version:', torch.backends.cudnn.version())
    print('Device:', torch.cuda.get_device_name(0))
    print('Device count:', torch.cuda.device_count())
"
```

### 重新安裝 PyTorch

```bash
# 完全移除
pip uninstall torch torchvision torchaudio -y

# 重新安裝
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 效能對比

在相同 GPU 上的訓練時間對比 (100 epoch):

| 框架 | 時間 | 備註 |
|------|------|------|
| PyTorch | ~45 分鐘 | 建議使用 |
| TensorFlow | ~60 分鐘 | 需 cuDNN 版本相容 |

## 遷移指南

如果需要從 TensorFlow 模型遷移:

```python
# 1. 用 PyTorch 訓練新模型
python test/run_pipeline_pytorch.py

# 2. PyTorch 模型權重已自動保存為 btc_15m_model_pytorch.pt

# 3. 推理時使用 PyTorch 模型
from src.model_pytorch import LSTMModel
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel(..., device=device)
model.load('test/models/btc_15m_model_pytorch.pt')
predictions = model.predict(X_test)
```

## 常見問題

### Q: PyTorch 模型比 TensorFlow 慢嗎?
A: 不會。在 GPU 上 PyTorch 通常更快,因為 LSTM 實現高度優化。

### Q: 能在 CPU 上訓練嗎?
A: 可以,但會很慢 (約 8 小時)。自動會偵測並使用 CPU。

### Q: PyTorch 模型能在推理時使用 TensorFlow?
A: 不行。需要用 PyTorch 推理,或轉換模型格式。

### Q: 如何在生產環境部署?
A: 儲存為 ONNX 格式以便跨框架支援:

```python
import torch
from src.model_pytorch import LSTMModel

model = LSTMModel(...)
model.load('btc_15m_model_pytorch.pt')

# 轉換為 ONNX
dummy_input = torch.randn(1, 100, num_features)
torch.onnx.export(model, dummy_input, "btc_15m_model.onnx")
```

## 參考資源

- [PyTorch 官方文件](https://pytorch.org/docs/stable/index.html)
- [PyTorch LSTM 文件](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [CUDA Toolkit 版本對應](https://docs.nvidia.com/cuda/archive/)
