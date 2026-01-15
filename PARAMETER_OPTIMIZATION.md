# Parameter Optimization & Iteration Guide

## Strategy Adjustment Methodology

本文檔提供逐步調整策略的完整流程，確保基於實際數據而非猜測進行優化。

---

## Part 1: Baseline Testing (Week 1)

### Objective
- 驗證策略在推薦參數下的基本可行性
- 收集20-30筆交易數據
- 建立績效基準線

### Configuration

**推薦設定（平衡配置）：**
```
MACD Settings:
  Fast Length: 5
  Slow Length: 13
  Signal Length: 1

RSI Settings:
  Length: 9
  Oversold: 30
  Overbought: 70

Volatility & Volume:
  ATR Length: 14
  Volume MA Length: 20

Risk Management:
  Risk-Reward Ratio: 1.5
  Position Size: 10%
  Use ATR for Stops: True
  ATR Multiplier: 2.0

ML Parameters:
  ML Lookback: 20
  ML Signal Threshold: 0.6

Filters:
  Use Trend Filter: True
  Require Above-Avg Volume: True
  Use MACD Divergence: False
```

### Backtesting Setup

**在TradingView上執行以下操作：**

1. 選擇交易對：BTC_USDT
2. 時間框架：15m
3. 回測期間：過去6個月
4. 初始資本：$10,000
5. 佣金：0.1%
6. 滑點：0.05%

### Key Metrics to Record

| Metric | Formula | Target | Note |
|--------|---------|--------|------|
| **Win Rate** | Winning Trades / Total Trades | >45% | 基本可行性 |
| **Net Profit** | Total Profit - Losses | >0 | 是否盈利 |
| **Profit Factor** | Gross Profit / Gross Loss | >1.5 | 風險報酬比 |
| **Sharpe Ratio** | Excess Return / Std Dev | >0.8 | 風險調整報酬 |
| **Max Drawdown** | Peak to Trough % | <30% | 風險控制 |
| **Avg Win/Loss Ratio** | Avg Win / Avg Loss | >1.3 | 單筆盈虧比 |
| **Monthly Return** | Monthly % Return | 3-8% | 現實預期 |
| **Total Trades** | Number of Trades | >20 | 樣本量 |

### Decision Criteria

**如果基準線測試結果：**

✅ **優秀（Pass）**
- Win Rate > 50%
- Profit Factor > 2.0
- Sharpe > 1.2
- Max Drawdown < 20%
➜ 進入參數微調階段

⚠️ **可接受（Accept）**
- Win Rate: 45-50%
- Profit Factor: 1.5-2.0
- Sharpe: 0.8-1.2
- Max Drawdown: 20-30%
➜ 進行適度調整後繼續

❌ **失敗（Fail）**
- Win Rate < 45%
- Profit Factor < 1.5
- Sharpe < 0.8
- Max Drawdown > 30%
➜ 檢查代碼或參數設置

---

## Part 2: RSI Parameter Optimization (Week 2)

### A/B Testing Framework

進行並聯對比測試，每組配置執行完整6個月回測。

### Test Group 1: RSI Length Variations

**目的：** 找到最優的RSI反應速度

**配置變量：**

| Configuration | RSI Length | RSI Oversold | RSI Overbought | Expected | |
|---|---|---|---|---|---|
| **Baseline** | 9 | 30 | 70 | 中等反應 | Ctrl |
| **Fast** | 7 | 30 | 70 | 快速訊號 | Test |
| **Medium** | 12 | 30 | 70 | 平衡 | Test |
| **Slow** | 14 | 30 | 70 | 穩定過濾 | Test |

**評估指標：**

```
For Fast (RSI=7):
  優點：提早進場，捕捉更多動作
  缺點：假訊號多，勝率可能下降
  
For Medium (RSI=12):
  優點：平衡快慢，減少噪音
  缺點：錯失某些快速動作
  
For Slow (RSI=14):
  優點：訊號穩定，高勝率
  缺點：反應慢，進場點差
```

### Test Group 2: RSI Levels Variations

**目的：** 優化超買超賣門檻

**配置變量：**

| Configuration | RSI Oversold | RSI Overbought | Strategy Type | |
|---|---|---|---|---|
| **Conservative** | 35 | 65 | 更晚進場，更穩定 | Test |
| **Baseline** | 30 | 70 | 標準設置 | Ctrl |
| **Aggressive** | 25 | 75 | 更早進場，更多訊號 | Test |
| **Extreme** | 20 | 80 | 極端反轉捕捉 | Test |

**特別說明：**
根據 eMarketInsights 研究，15分鐘圖表上 70/30 級別能早於極端值 (80/20) 捕捉動能轉換。

### Recording Template

**Test Date:** ___________
**Trading Pair:** BTC/USDT
**Period:** Last 6 months

**RSI=7, 30/70 Results:**
- Total Trades: _____
- Win Rate: _____%
- Profit Factor: _____
- Sharpe Ratio: _____
- Max Drawdown: _____%
- Net Profit: $_____
- Notes: _____________________

**優選結果:** ☐ Fast ☐ Baseline ☐ Medium ☐ Slow

---

## Part 3: MACD & ML Threshold Optimization (Week 3)

### MACD Parameters Analysis

**當前設置：** (5, 13, 1) - 針對加密貨幣優化

**為何此設置：**
1. 快速線(5)比標準(12)更靈敏
2. 慢速線(13)vs(26)更適應15分鐘週期
3. 信號線(1)減少延遲

**替代配置測試：**

| Config | Fast | Slow | Signal | Purpose |
|--------|------|------|--------|----------|
| **Baseline** | 5 | 13 | 1 | 當前優化 |
| **Standard** | 12 | 26 | 9 | 傳統設置對比 |
| **Fast-Crypto** | 3 | 10 | 1 | 極快反應 |
| **Smooth-Crypto** | 6 | 15 | 2 | 減少噪音 |

**測試步驟：**
```
1. 設置當前MACD(5,13,1)
2. 運行6個月回測
3. 記錄所有指標
4. 更改為Standard(12,26,9)
5. 重複運行，對比結果
6. 選擇Profit Factor最高的配置
```

### ML Signal Threshold Optimization

**當前設置：** 0.6

**邏輯：**
- 動能分數範圍：-1.0 到 +1.0
- Long訊號：Score > 0.6
- Short訊號：Score < -0.6

**A/B測試配置：**

| Threshold | Sensitivity | Expected Trades | Expected Quality | |
|---|---|---|---|---|
| **0.4** | 極高（敏感） | 100+ | 低勝率，多假訊號 | Test |
| **0.5** | 高 | 60-80 | 中等勝率 | Test |
| **0.6** | 中等（基線） | 40-60 | 高勝率 | Ctrl |
| **0.7** | 低 | 20-40 | 極高勝率，少交易 | Test |
| **0.8** | 極低 | <20 | 難以獲得訊號 | Test |

**測試方式（同時調整）：**
```
固定其他參數，僅改變ML Signal Threshold：

回測6個月，計算：
  - 總交易數
  - 勝率
  - Profit Factor
  - 月均報酬率
  
決策：
  若交易數>30且勝率>50%，則優選該門檻
```

---

## Part 4: Risk Management Tuning (Week 4)

### Stop Loss Distance Testing

**當前設置：** ATR(14) × 2.0

**替代方案測試：**

| Config | Multiplier | Typical Distance (BTC) | Edge | Drawback |
|--------|------------|------------------------|------|----------|
| **Conservative** | 2.5 | $500 | 減少止損被掃 | 盈虧比下降 |
| **Baseline** | 2.0 | $400 | 平衡點 | - |
| **Aggressive** | 1.5 | $300 | 更接近進場點 | 增加止損觸發 |
| **Tight** | 1.0 | $200 | 極小風險 | 高止損率 |

**測試流程：**
```
保持TP = SL × 1.5的關係

測試順序：
  1. SL Distance = ATR × 1.5, TP = SL × 1.5
  2. SL Distance = ATR × 2.0, TP = SL × 1.5
  3. SL Distance = ATR × 2.5, TP = SL × 1.5
  4. SL Distance = ATR × 3.0, TP = SL × 1.5

評估指標優先級：
  a) 勝率 > 45%
  b) Profit Factor > 1.5
  c) 最大連敗 < 5筆
```

### Position Sizing Impact

**當前設置：** 每筆10% equity

**測試配置：**

| Position Size | Impact | Risk |
|---|---|---|
| 5% | 低波動，長期可持續 | 增加交易次數才能達成目標 |
| 10% | 平衡（基線） | 中等波動 |
| 15% | 更快達成目標 | 更大回撤 |
| 20% | 高槓桿 | 高風險，容易爆倉 |

**注意：**
- 不涉及槓桿，純粹倉位大小
- 建議從10%開始逐步測試
- 監控最大連敗情況

---

## Part 5: Filter Optimization (Week 4-5)

### Volume Filter A/B Test

**設置：** 要求成交量 > 20週期移動平均

**測試方案：**

```
Scenario A (Filter ON):
  - 只在成交量 > MA20時進場
  - 預期：更高勝率，更少交易
  - 測試結果記錄：
    Total Trades: ___
    Win Rate: ____%
    
Scenario B (Filter OFF):
  - 成交量條件移除
  - 預期：更多交易，可能更多假訊號
  - 測試結果記錄：
    Total Trades: ___
    Win Rate: ____%
    
決策：
  若Filter ON的Profit Factor > 1.5倍 Filter OFF
  ➜ 保持Filter ON
  否則 ➜ 移除此過濾
```

### Trend Confirmation Filter A/B Test

**設置：** EMA20 > EMA60 > EMA120 確認

**測試方案：**

```
Scenario A (Trend Filter ON):
  - 做多：必須EMA20 > EMA60 > EMA120
  - 做空：必須EMA20 < EMA60 < EMA120
  - 預期：勝率高，交易少

Scenario B (Trend Filter OFF):
  - 移除EMA確認
  - 只靠MACD+RSI訊號
  - 預期：更多逆勢交易

比較結果：
  Filter ON: PF = ___, WR = ___%, Trades = ___
  Filter OFF: PF = ___, WR = ___%, Trades = ___
```

---

## Part 6: Cross-Market & Timeframe Testing

### 多交易對驗證

在基準測試通過後，測試其他交易對確保泛用性。

**測試清單：**

| Pair | Period | Win Rate | PF | Sharpe | Status |
|------|--------|----------|----|---------|---------|
| BTC/USDT | 6m | ____% | ____ | ____ | |
| ETH/USDT | 6m | ____% | ____ | ____ | |
| XRP/USDT | 6m | ____% | ____ | ____ | |
| SOL/USDT | 6m | ____% | ____ | ____ | |
| ADA/USDT | 6m | ____% | ____ | ____ | |

**目標：** 至少3個交易對的勝率 > 45%

### 不同時間範圍測試

基本設置為15m，但也測試：

| Timeframe | Purpose | Expected Impact |
|---|---|---|
| **5m** | 超短線 | 更多訊號，更多假信 |
| **15m** | 標準（基線） | 最優平衡 |
| **30m** | 中線 | 更少訊號，更穩定 |
| **1h** | 長線 | 完全不同表現 |

**記錄** 5m 和 30m 的結果，評估策略的時間框架敏感性。

---

## Part 7: Summary & Final Selection

### Optimization Summary Template

**第一週基準測試（Week 1）**
```
配置：Baseline (RSI=9, MACD=5,13,1, ML=0.6)
期間：BTC/USDT 6 months
結果：WR=48%, PF=1.7, Sharpe=0.95, DD=22%
決定：PASS ➜ 進行調整
```

**第二週RSI優化（Week 2）**
```
測試組：RSI Length (7, 9, 12, 14)
最優：RSI=7, WR=52%, PF=1.9
改進：+4% Win Rate, +0.2 PF
下一步：更新為RSI=7
```

**第三週MACD+ML優化（Week 3）**
```
測試：MACD(5,13,1) vs Standard(12,26,9)
       ML Threshold (0.4-0.8)
結果：MACD(5,13,1)最佳, ML=0.65
改進：Trades從45增加到52
```

**第四週風險優化（Week 4）**
```
ATR測試：1.5x到3.0x
最優：ATR × 2.2
成交量過濾：ON
趨勢過濾：ON
最終配置確定
```

### Final Optimized Configuration

**基於完整4週測試後的最終參數：**

```
MACD Settings:
  Fast Length: 5
  Slow Length: 13
  Signal Length: 1

RSI Settings:
  Length: 7 (or optimized result)
  Oversold: 30
  Overbought: 70

Risk Management:
  Risk-Reward: 1.5
  Position Size: 10%
  ATR Multiplier: 2.2 (or optimized)

ML Settings:
  Signal Threshold: 0.65 (or optimized)
  Lookback: 20

Filters:
  Volume: ON
  Trend: ON
```

---

## Iteration Checklist

### 每週審查（Weekly Review）

- [ ] 回測5個完整交易日新數據
- [ ] 檢查即時訊號質量
- [ ] 記錄每筆交易的進出點
- [ ] 評估是否有系統性的進場偏誤
- [ ] 檢查是否有參數過度優化的跡象

### 每月審查（Monthly Review）

- [ ] 對比上月績效
- [ ] 評估是否需要微調參數
- [ ] 檢查在不同市場環境下的表現
- [ ] 測試新的技術指標組合
- [ ] 更新文檔記錄

### 紅旗指標（Red Flags）

如果出現以下情況，停止交易並調查：

```
1. 單週勝率 < 30%
2. 連續3天沒有訊號
3. 平均交易持續時間 > 4小時（不符合15m短線）
4. 最大單筆虧損 > 初始資本2%
5. 實盤結果與回測結果差異 > 20%
6. 同時出現5筆以上連續虧損
```

---

## Notes

- 每次參數更改只改1-2個變量
- 每次測試至少6個月歷史數據
- 使用相同回測設置（傭金0.1%, 滑點0.05%）
- 記錄所有測試結果便於對比
- 定期在新市況上驗證策略有效性

---

**Last Updated:** 2026-01-15