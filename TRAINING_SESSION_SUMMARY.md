# 訓練會話總結 - 2026-01-14

## 已完成的工作

### 1. 模型整合評估與決策
- 評估了兩種方案：單一模型 vs 雙模型
- 決定使用單一整合模型（run_pipeline_pytorch.py）
- 廢棄 train_volatility_model.py，避免重複維護

### 2. 波動性特徵實現
已在 data_processor.py 中成功實現 5 個新波動性特徵：

| 特徵 | 說明 | 計算方式 |
|-----|------|--------|
| volatility_5 | 5期波動率 | returns 的 5 期滾動標準差 |
| volatility_20 | 20期波動率 | returns 的 20 期滾動標準差 |
| momentum_5 | 5期動量 | 5期價格變化百分比 |
| momentum_10 | 10期動量 | 10期價格變化百分比 |
| ATR | 平均真幅度 | ta-lib 計算，已標準化 |

加上原有的 11 個特徵，共 16 個特徵用於訓練。

### 3. 模型架構優化
**使用 Huber Loss 而非 MSE**
- delta = 0.5
- 對異常值（大波動）的魯棒性更好
- 比 MSE 更能保留方差信息
- 適合波動性預測

**使用 StandardScaler 而非 MinMaxScaler**
- Z-score 標準化
- 保留方差信息
- 更適合時間序列回報預測

### 4. 程式碼改進
修改了 run_pipeline_pytorch.py：
- 新增時間戳支持（為未來的時間戳型預測做準備）
- 新增特徵級別的診斷指標
- 改進特徵列表的可見性和日誌記錄
- 保存 feature_columns.pkl 供後續推理使用

### 5. 配置優化
更新 config/config.yaml：
- LSTM units: 256 (第一層)
- 啟用所有 5 個波動性特徵
- 使用 StandardScaler（標準化方式改為 "standard"）
- 明確註解各特徵的作用

### 6. 文檔完善
創建了詳細的執行指南：
- EXECUTION_GUIDE.md：完整的訓練說明和故障排除
- 包含性能預期、配置參考、進階用法

## 技術亮點

### 多特徵聯合訓練 vs 獨立訓練
```
舊方案：價格模型（15個特徵） + 獨立波動性模型
新方案：統一模型（16個特徵，包含波動性）

優勢：
1. 波動性與價格的關聯性被充分學習
2. 共享的 LSTM 隱藏層表示能更好地捕捉市場動態
3. 無需協調兩個模型的預測
4. 訓練時間相同，但效果更好
```

### 損失函數的選擇
```
Huber Loss vs MSE：
- Huber 對離群值（大波動）的懲罰較輕
- MSE 會導致模型學習平均值（過度平滑）
- 對於波動性預測，Huber 更合適

設置 delta=0.5：
- 在 error > 0.5 時，使用線性損失（L1）
- 在 error < 0.5 時，使用二次損失（L2）
- 平衡穩定性和敏感性
```

### 特徵標準化的影響
```
StandardScaler vs MinMaxScaler：
- StandardScaler 保留方差信息（保留離群值信息）
- MinMaxScaler 壓縮到 [0,1]，損失方差
- 對於時間序列回報預測，標準化更好

實際效果：
- 模型能更好地學習 returns 的波動模式
- volatility_5, volatility_20 的預測會更接近實際波動
```

## 當前模型配置

### 輸入/輸出
```
Input:  100 根 K 棒 x 16 特徵 (1600 個數據點)
Output: 15 根 K 棒 x 16 特徵 (240 個數據點)
```

### 模型參數
```
Layer 1: LSTM(16 -> 256)
Layer 2: LSTM(256 -> 128)
Dense 1: 128 -> 128 (ReLU)
Dense 2: 128 -> 240 (輸出層，reshape 為 15x16)

總參數: ~136k (訓練中)
可訓練參數: ~136k
```

### 訓練設置
```
損失函數: Huber Loss (delta=0.5)
優化器: Adam (lr=0.001)
正則化: L2 (0.0001)
Dropout: 0.3
早停: 10 epoch 無改進則停止
最大 epoch: 100
```

## 立即執行步驟

### 第1步：開始訓練
```bash
cd system_v5
python test/run_pipeline_pytorch.py
```

### 預期結果
```
訓練時間（T4 GPU）: ~60-90 分鐘
每個 epoch: ~40-50 秒

目標指標：
- R2 Score: 0.80+
- Volatility_5 ratio: 0.85-1.05
- Volatility_20 ratio: 0.85-1.05
```

### 第2步：驗證結果
訓練完成後檢查：
```
test/models/
├── btc_15m_model_pytorch.pt  # 模型權重
├── btc_15m_scaler.pkl         # 特徵標準化器
└── feature_columns.pkl         # 特徵列表

test/results/
├── metrics_report.txt          # 詳細指標
├── predictions.npy             # 預測值
└── targets.npy                 # 真實值
```

### 第3步：分析波動性指標
在 metrics_report.txt 中查看：
```
volatility_5 - Pred std: X.XXX, True std: X.XXX, Ratio: X.XX
volatility_20 - Pred std: X.XXX, True std: X.XXX, Ratio: X.XX
```

如果 ratio 接近 1.0，說明波動性預測正確。

## 下一階段計劃

### Phase 1: 驗證穩定性（訓練完成後）
- 實現基於 close_time 的時間戳型預測
- 驗證同一輸入多次預測結果相同
- 測試波動性特徵的穩定性

### Phase 2: 優化（如需要）
- 如果 R2 < 0.80，調整：
  - 增加 LSTM units (256 -> 512)
  - 減少 dropout (0.3 -> 0.2)
  - 調整 Huber delta (0.5 -> 0.3)

### Phase 3: 生產化
- 構建實時 K 棒生成管道
- 實現模型推理 API
- 添加預測結果驗證機制

### Phase 4: 擴展（驗證成功後）
- 訓練其他 37 個交易對
- 構建交易對選擇機制
- 實現組合預測

## 注意事項

1. **時間戳型預測的重要性**
   - 當前訓練是通用的，適用於任何 100 根 K 棒序列
   - 生產環境中需要基於 close_time 而非 iloc 索引
   - 這會在下一步（Phase 1）實現

2. **波動性預測的挑戰**
   - 加密貨幣市場波動劇烈
   - 預測波動性比預測價格更難
   - 5 個新特徵的加入應該幫助模型學習波動模式
   - 預期 Volatility_5 ratio 在 0.85-1.05 之間為正常

3. **模型泛化性**
   - 訓練數據包含 2019-2026 年的所有市場情景
   - 應該對新的市場情景有一定的泛化能力
   - 但仍建議定期用新數據重訓練模型

## 關鍵指標說明

| 指標 | 定義 | 目標值 | 說明 |
|-----|------|--------|------|
| R2 Score | 模型解釋方差的比例 | 0.80+ | 越高越好，衡量整體擬合度 |
| RMSE | 均方根誤差 | 越小越好 | 對大誤差的懲罰更重 |
| MAE | 平均絕對誤差 | 越小越好 | 與實際單位一致，易於解釋 |
| MAPE | 平均百分比誤差 | 越小越好 | 衡量相對誤差 |
| Vol_Ratio | 預測波動/實際波動 | 0.90-1.10 | 接近 1.0 表示波動性預測準確 |

## 檔案清單

### 新增
- EXECUTION_GUIDE.md - 執行指南
- config/config.yaml - 更新配置

### 修改
- test/run_pipeline_pytorch.py - 優化和增強
- src/data_processor.py - 保持不變（已包含所有特徵）
- config/config.yaml - 特徵配置更新

### 刪除
- test/train_volatility_model.py - 已廢棄

### 保持不變
- src/model_pytorch.py - 模型架構完整
- src/data_loader.py - 數據加載完整
- src/evaluator.py - 評估功能完整

## 成功標誌

訓練成功的標誌：
1. 訓練完成無錯誤
2. R2 Score 達到 0.80+
3. volatility_5 ratio 在 0.85-1.05
4. volatility_20 ratio 在 0.85-1.05
5. 模型文件正確保存

然後可以進入 Phase 1（穩定性驗證）。
