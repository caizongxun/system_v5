# Momentum ML Strategy v1.0 Documentation

## Overview

Momentum ML Strategy v1.0 是為15分鐘加密貨幣短線交易設計的量化交易策略。整合了多層次的動能指標、機器學習輔助判斷，以及嚴格的風險管理機制(1:1.5盈虧比)。

## Strategy Architecture

### 1. Trend Layer - EMA Multiple Timeframes

**指標組合：**
- EMA(20) - 快速趨勢
- EMA(60) - 主趨勢  
- EMA(120) - 長期參考

**邏輯：**
- 上漲信號：EMA20 > EMA60 > EMA120
- 下跌信號：EMA20 < EMA60 < EMA120

**研究基礎：**根據Binance官方15分鐘交易指南，此三層EMA組合在15分鐘週期上表現最為穩定，避免假訊號。

---

### 2. Momentum Layer - Multi-Indicator Score

#### A. MACD (5, 13, 1) - 加密貨幣優化參數

**標準設定：** MACD(12, 26, 9)在15分鐘圖表上反應緩慢

**加密貨幣優化：** (5, 13, 1) 提供更快的訊號
- 權重：40% of momentum score
- 檢測：直方圖方向變化、零軸穿越
- 確認：黃金交叉/死亡交叉強度

#### B. RSI (9) - 超買超賣檢測

**參數：**
- 週期：9（快速反應)
- 超賣：30 (做多)
- 超買：70 (做空)
- 權重：30% of momentum score

**研究支持：**ePlantBrokers研究指出15分鐘圖表上70/30等級能更早捕捉動能轉換。

#### C. Price Position - 價格與EMA20關係

- 權重：30% of momentum score
- 檢測：收盤價在EMA20上或下方

---

### 3. ML Learning Layer - Momentum Score

**計算公式：**
```
Momentum Score = (MACD_Signal × 0.4) + (RSI_Signal × 0.3) + (Price_Trend × 0.3)
```

**信號範圍：** -1.0 to +1.0

**閾值設定：**
- Long Signal: Score > 0.6 (可調)
- Short Signal: Score < -0.6 (可調)

**設計理念：**
- 此非傳統ML模型，而是加權動能學習機制
- 避免過度擬合
- 快速運算，適合實時交易
- 可在TradingView直接回測

---

### 4. Confirmation Layer - Volume & Filters

**成交量確認：**
- 檢測：成交量 > 20週期簡單移動平均
- 邏輯：成交量確認動能強度

**可選過濾器：**
1. 趨勢確認 - 確保順勢操作
2. 成交量過濾 - 排除低流動性
3. MACD背離 - 檢測頂底背離信號 (目前未啟用)

---

## Risk Management System

### Stop Loss & Take Profit Calculation

**標準盈虧比：1:1.5**

**計算方法：**

```
Stop Loss Distance = ATR(14) × 2.0 (可調)
Take Profit Distance = Stop Loss Distance × 1.5
```

**範例：**
- 進場：$50,000
- ATR值：$200
- SL距離：$200 × 2.0 = $400
- TP距離：$400 × 1.5 = $600
- 做多：SL=$49,600 | TP=$50,600
- 淨風險：$400 | 潛在收益：$600

### Position Sizing

- 預設：10% equity per trade
- 可調範圍：1-100%
- 每筆交易僅開一個部位
- 當前部位平倉後才能開新單

---

## Parameter Settings Guide

### Conservative Setup (低風險，高勝率)

```
RSI Length: 14 (較穩定)
RSI Levels: 35/65
MACD: (5, 13, 1)
ATR Multiplier: 2.5
ML Threshold: 0.7
Volume Filter: ON
Trend Filter: ON
```

### Aggressive Setup (高風險，高報酬)

```
RSI Length: 7 (快速反應)
RSI Levels: 25/75
MACD: (5, 13, 1)
ATR Multiplier: 1.5
ML Threshold: 0.4
Volume Filter: OFF
Trend Filter: OFF
```

### Balanced Setup (推薦)

```
RSI Length: 9
RSI Levels: 30/70
MACD: (5, 13, 1)
ATR Multiplier: 2.0
ML Threshold: 0.6
Volume Filter: ON
Trend Filter: ON
Position Size: 10%
Risk-Reward: 1.5
```

---

## How to Use in TradingView

### Step 1: Copy Strategy Code
1. 從 `momentum_ml_strategy_v1.pine` 複製全部代碼
2. 進入TradingView平台
3. Pine Script編輯器 > 新建策略
4. 貼上代碼

### Step 2: Configure Parameters
1. 右上方 "Strategy" 面板
2. 調整各個參數組 (MACD Settings, RSI Settings, etc.)
3. 根據標的調整閾值

### Step 3: Backtest
1. 進入回測面板
2. 選擇交易對：BTC_USDT, ETH_USDT 等
3. 時間框架：15m
4. 回測區間：至少6個月
5. 檢查統計數據：
   - 勝率 (Win Rate)
   - 淨利潤 (Net Profit)
   - 夏普比率 (Sharpe Ratio)
   - 最大回撤 (Max Drawdown)

### Step 4: Alert Setup
1. "Create Alert" 選擇此策略
2. 選擇 "Long Signal" 或 "Short Signal"
3. 設定提醒方式 (Email/Webhook)

---

## Iterative Optimization Process

### Phase 1: Baseline Testing (Week 1)
- 在推薦參數下運行
- 收集20-30筆交易數據
- 記錄勝率、平均盈虧

### Phase 2: Parameter Tuning (Week 2-3)
- 調整RSI週期 (7-14)
- 調整ML閾值 (0.4-0.8)
- 測試不同ATR倍數
- 識別最佳參數組合

### Phase 3: Filter Optimization (Week 3-4)
- A/B測試：開啟/關閉成交量過濾
- 測試趨勢確認的必要性
- 評估背離檢測的邊際效益

### Phase 4: Risk Adjustment (Week 4-5)
- 測試不同盈虧比 (1:1 vs 1:1.5 vs 1:2)
- 調整部位大小
- 優化止損距離

### Phase 5: Live Trading (小資金)
- 從最小部位開始
- 驗證回測結果
- 監控實際訊號質量
- 根據實際績效調整

---

## ML Enhancement Roadmap

### Current Implementation (v1.0)
- 加權動能評分機制
- 簡單規則型機器學習

### Next Steps (v2.0)

**1. Random Forest Integration**
- 訓練模型：20週期歷史動能 → 下個5根K棒方向
- 輸入特徵：RSI、MACD、ATR、Volume
- TradingView限制：需Python後端

**2. LSTM Sentiment Analysis**
- 集成社交媒體情緒指標
- 權重：20% 基本信號 + 30% 情緒信號

**3. Adaptive Parameters**
- 動態調整RSI週期
- 根據市場波動調整ATR倍數

---

## Common Issues & Solutions

### Issue 1: 過多假訊號
**原因：** ML閾值設得過低 | 成交量過濾未啟用
**解決：**
- 提高ML Threshold 到 0.7-0.8
- 啟用 "Require Above-Avg Volume"
- 增加ATR倍數 (2.5-3.0)

### Issue 2: 回測績效好但實盤差
**原因：** 滑點、成交延遲、過度優化
**解決：**
- 在Slippage/Commission標籤增加交易成本
- 縮短測試區間，避免曲線擬合
- 在不同標的和時期測試

### Issue 3: 訊號太少
**原因：** 過濾條件過嚴
**解決：**
- 關閉或放鬆趨勢過濾
- 降低ML閾值到 0.5
- 關閉成交量確認

---

## Backtesting Checklist

- [ ] 選擇流動性充足的交易對
- [ ] 使用真實成交量數據
- [ ] 加入手續費和滑點 (0.1%)
- [ ] 至少測試6個月數據
- [ ] 測試不同市況 (牛市/熊市/盤整)
- [ ] 對比不同時期績效
- [ ] 檢查最大連敗
- [ ] 驗證夏普比率 > 1.0
- [ ] 驗證勝率 > 40%
- [ ] 驗證最大回撤 < 30%

---

## Performance Expectations

### Realistic Targets (Conservative)
- 勝率：45-55%
- 月均報酬：3-8%
- 夏普比率：0.8-1.5
- 最大回撤：15-25%
- 交易頻率：5-15次/天 (15分鐘)

### Stretch Goals (Aggressive)
- 勝率：50-60%
- 月均報酬：8-15%
- 夏普比率：1.5-2.5
- 最大回撤：10-20%

---

## Version History

### v1.0 (2026-01-15)
- 初始版本
- EMA趨勢層
- MACD+RSI動能層
- 加權動能評分
- 基本風險管理
- 成交量確認

### Future Versions
- v2.0: Random Forest + Sentiment
- v3.0: Adaptive Parameters
- v4.0: Multi-timeframe Analysis

---

## Research References

1. "Performance Analysis of Momentum Algorithm in Cryptocurrency Trading" - IEEE Paper
2. "Cryptocurrency Price Prediction using Regression Models on Momentum Indicators" - 2023
3. "Technical Analysis Methods in Cryptocurrency Markets" - Case Study XAU/USD
4. "Momentum, Machine Learning, and Cryptocurrency" - CBS Research, 2021
5. "Top 5 Momentum Indicators Every Day Trader Should Know" - EuropeanBusinessReview, 2025
6. "Binance 15-Minute Trading Strategy Guide" - Official

---

## Disclaimer

此策略僅供教育和研究之用。過去的績效不代表未來的結果。交易涉及風險，包括資本損失的可能。在使用此策略進行實際交易前：

1. 在小資金上充分測試
2. 根據個人風險承受能力調整參數
3. 永遠設定止損
4. 不要用無法承受損失的資金進行交易
5. 定期監控策略績效

---

**Last Updated:** 2026-01-15
**Author:** caizongxun
**Status:** Active Development