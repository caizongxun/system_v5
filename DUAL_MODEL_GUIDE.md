# 雙模型架構指南 (Dual Model Architecture Guide)

## 概念

傳統方法用一個模型預測所有特徵(包括returns),導致predicted returns非常小,價差只有美分等級。

我們改用**雙模型架構**:
1. **方向模型 (Direction Model)**: 預測returns的符號 (正/負)
2. **波動模型 (Volatility Model)**: 專門預測波動大小(幅度)

結合起來: `最終預測 = sign(方向) × 波動幅度`

## 優勢

- 方向模型專注於學習市場方向,減少noise
- 波動模型專注於學習振幅,避免幅度偏小
- 組合預測更符合真實市場波動
- 分離關注點提高模型效率

## 訓練步驟

### Step 1: 準備訓練數據

確保你有歷史K線數據 CSV:
```bash
data/
  BTC_15m_historical.csv
```

必要欄位: `open_time, open, high, low, close, volume`

### Step 2: 訓練波動模型

```bash
python train_volatility_model.py \
  --config config.yaml \
  --epochs 100 \
  --batch-size 32 \
  --lr 0.001
```

參數說明:
- `--config`: 配置文件路徑
- `--epochs`: 訓練週期
- `--batch-size`: 批次大小
- `--lr`: 學習率

#### 預期輸出

```
Features shape: (5000, 16)
Volatility targets shape: (4970, 15)
Volatility range: 0.000100 to 0.025000
Volatility mean: 0.001234, std: 0.002456

Train samples: 3976, Val samples: 994
Starting training for 100 epochs...
Epoch 10/100 - Train Loss: 0.000045, Val Loss: 0.000052
...
Best model saved - Val Loss: 0.000038
Training completed!
```

### Step 3: 檢查保存的模型

訓練完成後會生成:
- `models/btc_15m_volatility_model.pt` - 波動模型權重
- `models/volatility_scaler.pkl` - 特徵歸一化器

### Step 4: 運行雙模型版本

```bash
streamlit run app_dual_model.py
```

或保持使用改進後的 `app.py`,它會自動用 market volatility 進行縮放。

## 模型架構

### Direction Model (現有)
```
Input (100, 16)
  ↓
LSTM (2層, 64單位)
  ↓
Dense層組
  ↓
Output (15, 16) - 所有特徵預測,包括returns
```

### Volatility Model (新增)
```
Input (100, 16)
  ↓
LSTM (2層, 64單位)
  ↓
Dense層 (128 → 64 → 15)
ReLU輸出 (確保正數)
  ↓
Output (15,) - 每個時間步的波動幅度
```

## 預測流程

```python
# 1. 預測方向
X_normalized = scaler.transform(X_data)
direction_pred = direction_model(X_normalized)
direction_denorm = scaler.inverse_transform(direction_pred)

# 2. 預測波動
volatility_pred = volatility_model(X_normalized)  # (15,)

# 3. 組合
final_returns = sign(direction_denorm[:, returns_idx]) * volatility_pred

# 4. 轉換為K線
klines = generate_klines(final_returns, last_price)
```

## 配置文件更新

在 `config.yaml` 中添加:

```yaml
model:
  sequence_length: 100
  prediction_length: 15
  lstm_units: 64
  dropout_rate: 0.2
  learning_rate: 0.001
  volatility_lstm_units: 64  # 波動模型LSTM單位
```

## 日誌輸出示例

### 訓練時
```
Market volatility (last 20 bars): 0.001234 (0.1234%)
Step 1: direction=0.000078, volatility=0.001234, combined=0.001312
Step 2: direction=-0.000045, volatility=0.001456, combined=-0.001456
```

### 預測時
```
Prediction Step
================
Direction - min: -0.000088, max: 0.000078, mean: -0.000012
Volatility - min: 0.001200, max: 0.002500, mean: 0.001650

Generated K-lines summary:
  Price range: $94,800 - $95,300
  Close range: $94,900 - $95,200
  First: O=95077 C=95156 (0.083%)
  Last: O=95189 C=95089 (-0.105%)
```

## 故障排除

### 波動模型找不到

```
Warning: Volatility model not found. Will use direction model only.
```

解決:
1. 檢查 `models/btc_15m_volatility_model.pt` 是否存在
2. 運行訓練腳本: `python train_volatility_model.py`
3. 確認訓練完成無錯誤

### 預測仍然很小

可能原因:
1. 訓練數據缺乏波動性
2. 學習率設定不當
3. 波動模型欠擬合

解決:
1. 檢查訓練日誌中的 `Volatility range`
2. 嘗試更高的學習率: `--lr 0.005`
3. 增加訓練週期: `--epochs 200`
4. 增加 LSTM 單位: 在 config 中改為 128

### GPU 記憶體不足

```bash
python train_volatility_model.py \
  --batch-size 16  # 減小批次大小
```

## 性能指標

### 預期的模型大小
- Direction Model: ~500KB
- Volatility Model: ~200KB
- Scaler: ~10KB

### 訓練時間
- 5000個樣本, 100週期
- GPU: ~2-3分鐘
- CPU: ~10-15分鐘

### 推論速度
- 單次預測: <50ms (GPU), <200ms (CPU)
- 實時應用可接受

## 高級用法

### 自定義波動計算

編輯 `train_volatility_model.py` 中的 `prepare_volatility_targets()`:

```python
def prepare_volatility_targets(df, pred_steps=15, lookback=5):
    # 改變lookback週期
    # 改變波動計算方式 (例: std instead of mean)
    # 加入ATR等指標
    pass
```

### 集成到現有系統

如果只想改進現有 `app.py`,已實現:
1. 自動計算市場波動率
2. 按波動率縮放預測
3. 無需訓練新模型

只需運行: `streamlit run app.py`

## 對比

### 單模型 (原始)
```
Returns: -0.000088 to 0.000078
高低比: 0.001964 到 0.002612
→ 價差: 3美金
```

### 雙模型 (改進後)
```
Direction Returns: -0.000088 to 0.000078
Volatility: 0.001200 to 0.002500
Combined: -0.002500 to 0.002500
→ 價差: 200美金+
```

## 下一步

1. 收集更多訓練數據 (數周至數月)
2. 實驗不同的波動計算方式
3. 添加其他特徵 (例: on-chain metrics)
4. 集成 backtesting 框架
5. 部署到生產環境
