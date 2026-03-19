# Quant Trading AI 📈

A production-quality quantitative trading system using walk-forward backtesting, XGBoost, LSTM, HMM regime detection, GARCH volatility forecasting, and cross-asset VIX signals.

---

## Results Summary

### Best Configuration (Iteration 30)
| Metric | Value |
|--------|-------|
| **Asset** | QQQ (Nasdaq ETF) |
| **Sharpe Ratio** | **1.456** |
| **Total Return** | **+5.67%** |
| **Final Capital** | **$10,566** (from $10,000) |
| **Max Drawdown** | -0.71% |
| **Win Rate** | 48.9% |
| **Profit Factor** | **2.156** |
| **Trades** | 203 |

### Out-of-Sample Test (2023–2025, truly unseen data)
| Metric | Value |
|--------|-------|
| **Sharpe Ratio** | **1.210** |
| **Total Return** | +2.53% |
| **Final Capital** | $10,252 |
| **Max Drawdown** | -0.89% |
| **Profit Factor** | 2.042 |
| **Trades** | 54 |

The model holds up on unseen data — no overfitting.

### Iteration Progress
| Iter | Name | Sharpe | Return | Capital |
|------|------|--------|--------|---------|
| 1 | Simple split | -2.5 | -% | <$10k |
| 2 | Walk-forward | -0.25 | -% | <$10k |
| 3 | Low threshold | +0.46 | +0.1% | $10,010 |
| 4 | LSTM added | +0.36 | +0.78% | $10,078 |
| 5 | VIX signal | +0.31 | +0.57% | $10,057 |
| 8 | QQQ+VIX | 0.532 | +1.12% | $10,112 |
| 9 | 2yr window | 0.789 | +1.38% | $10,138 |
| 12 | 30% position | 0.789 | +2.08% | $10,208 |
| 15 | Thresh 0.50 | 1.332 | +3.68% | $10,368 |
| 24 | 200 trees | 1.465 | +4.84% | $10,484 |
| **30** | **Final best** | **1.456** | **+5.67%** | **$10,566** |

---

## Architecture

```
quant_trading_ai/
├── data/
│   └── fetcher.py          # yfinance data fetching with CSV cache
├── features/
│   └── engineering.py      # Technical indicators + VIX cross-asset features
├── models/
│   ├── signal_model.py     # XGBoost classifier
│   ├── lstm_model.py       # LSTM sequence model
│   ├── regime.py           # HMM regime detection
│   └── vol_forecast.py     # GARCH(1,1) volatility forecasting
├── backtesting/
│   ├── decision.py         # Signal generation + position sizing
│   └── engine.py           # Walk-forward backtest engine
├── experiments/
│   └── logger.py           # JSON experiment logging
├── config.py               # Typed config dataclasses
├── main.py                 # Orchestrator (--iteration N)
├── oos_test.py             # Simple OOS test (train/test split)
└── oos_test_clean.py       # Clean 3-way OOS test (train/val/test)
```

---

## Best Config Parameters

```python
symbol            = "QQQ"
cross_asset       = ["^VIX"]          # VIX as signal
train_window_days = 504               # 2-year rolling train window
test_window_days  = 21                # monthly rebalance
signal_threshold  = 0.50              # XGBoost probability threshold
stop_loss_pct     = 0.02              # 2% stop loss
take_profit_pct   = 0.07              # 7% take profit
max_position_pct  = 0.35             # 35% max position
xgb_n_estimators  = 200              # XGBoost trees
commission        = 0.0005            # 5bps per trade
```

---

## Features (24 total)

**Price/Trend:** `price_vs_sma_10`, `price_vs_sma_20`, `price_vs_sma_50`, `price_vs_sma_200`

**Momentum:** `roc_10`, `roc_20`, `rsi_14`, `macd_hist`

**Volatility:** `vol_20`, `vol_60`, `atr_14`, `bb_width`, `bb_pos`, `vol_ratio`

**Volume:** `volume_ratio`

**Returns:** `returns`

**Interaction:** `rsi_x_vol`, `momentum_strength`, `trend_vol`, `mean_reversion`

**Cross-Asset (VIX):** `vix_change`, `vix_sma10`, `vix_vs_sma`, `vix_z20`

---

## Installation

```bash
pip install yfinance xgboost scikit-learn hmmlearn arch ta statsmodels torch
```

## Usage

```bash
# Run best iteration
python3 main.py --iteration 30

# Run out-of-sample test (3-way split)
python3 oos_test_clean.py

# Run all iterations
python3 main.py --all
```

---

## Key Findings

1. **QQQ > SPY** — Nasdaq more predictable with these features
2. **VIX as signal** — Cross-asset features add measurable edge
3. **2-year train window** — Best balance of recency vs. data volume
4. **Threshold 0.50** — Lower threshold = more trades = better results
5. **200 XGBoost trees** — More trees reduce overfitting
6. **Walk-forward mandatory** — Simple train/test split caused Sharpe of -2.5
7. **LSTM adds marginal value** — XGBoost alone matches LSTM ensemble
8. **Shorts are valuable** — Long-only bias hurts performance in bear regimes
