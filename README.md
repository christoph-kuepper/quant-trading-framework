# Quant Trading AI 

A production-grade quantitative trading system for QQQ (Nasdaq ETF) using walk-forward backtesting, XGBoost + LSTM ensemble, HMM regime detection, GARCH volatility forecasting, and cross-asset VIX signals.

---

## Results

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
| **Max Drawdown** | -0.89% |
| **Profit Factor** | 2.042 |

The model holds up on unseen data — no overfitting.

### Monte Carlo (1000 Simulations)
| Metric | Value |
|--------|-------|
| **Profitable runs** | 75.4% |
| **Median Sharpe** | 1.927 |
| **Median final capital** | $10,507 |
| **5th percentile** | $9,254 |
| **95th percentile** | $11,659 |

---

## Architecture

```
quant_trading_ai/
├── data/
│   └── fetcher.py              # yfinance + CSV cache + DVC versioning
├── features/
│   └── engineering.py          # 24 technical + VIX cross-asset features
├── models/
│   ├── signal_model.py         # XGBoost classifier
│   ├── lstm_model.py           # LSTM sequence model
│   ├── inference.py            # Unified XGB+LSTM ensemble pipeline
│   ├── regime.py               # HMM regime detection (3 states)
│   └── vol_forecast.py         # GARCH(1,1) volatility forecasting
├── backtesting/
│   ├── decision.py             # Signal generation + position sizing
│   └── engine.py               # Walk-forward engine (commission + slippage)
├── analysis/
│   ├── shap_explainer.py       # Per-trade SHAP explanations
│   ├── ensemble_explainer.py   # XGB vs LSTM agreement + LSTM sensitivity
│   ├── monte_carlo.py          # 1000-run robustness simulation
│   └── kelly.py                # Kelly Criterion position sizing
├── experiments/
│   └── logger.py               # JSON + MLflow experiment logging
├── live/
│   └── alpaca_bridge.py        # Paper trading via Alpaca API
├── tests/
│   ├── test_features.py        # Unit tests: feature engineering
│   └── test_backtest.py        # Unit tests: backtest engine
├── dashboard.py                # Backtest explorer (Plotly, interactive)
├── live_dashboard.py           # Live signal monitor (auto-refresh 60s)
├── config.py                   # Typed config dataclasses
├── main.py                     # Orchestrator (--iteration N)
├── oos_test_clean.py           # Clean 3-way OOS test (train/val/test)
├── Dockerfile                  # Container build
└── docker-compose.yml          # Multi-service compose
```

---

## Features (24 total)

| Category | Features |
|----------|----------|
| **Price/Trend** | `price_vs_sma_10`, `price_vs_sma_20`, `price_vs_sma_50`, `price_vs_sma_200` |
| **Momentum** | `roc_10`, `roc_20`, `rsi_14`, `macd_hist` |
| **Volatility** | `vol_20`, `vol_60`, `atr_14`, `bb_width`, `bb_pos`, `vol_ratio` |
| **Volume** | `volume_ratio` |
| **Returns** | `returns` |
| **Interaction** | `rsi_x_vol`, `momentum_strength`, `trend_vol`, `mean_reversion` |
| **Cross-Asset (VIX)** | `vix_change`, `vix_sma10`, `vix_vs_sma`, `vix_z20` |

---

## Installation

```bash
pip install -r requirements.txt
```

> **Note:** `alpaca-trade-api` is excluded due to a websockets version conflict with yfinance.
> For paper trading install separately: `pip install alpaca-py`

---

## Usage

### Backtesting
```bash
# Run best config (iteration 30)
python3 main.py --iteration 30

# Run all 30+ iterations
python3 main.py --all

# Out-of-sample test (train/val/test split)
python3 oos_test_clean.py
```

### Dashboards
```bash
# Live signal monitor (auto-refreshes every 60s)
streamlit run live_dashboard.py

# Backtest explorer with interactive Plotly charts
streamlit run dashboard.py
```

### Analysis
```bash
# SHAP explanation for a specific trade date
python3 analysis/shap_explainer.py --date 2024-03-14

# Ensemble explainer: XGB vs LSTM agreement + LSTM sensitivity
python3 analysis/ensemble_explainer.py

# Monte Carlo robustness (1000 simulations)
python3 analysis/monte_carlo.py

# Kelly Criterion position sizing
python3 analysis/kelly.py
```

### Paper Trading (Alpaca)
```bash
export ALPACA_API_KEY=your_key
export ALPACA_SECRET_KEY=your_secret
python3 live/alpaca_bridge.py
```

### Docker
```bash
docker-compose up           # Start dashboard
docker-compose --profile backtest up  # Run backtest
docker-compose --profile test up      # Run tests
```

---

## Explainability

Two-layer explainability addressing the "LSTM gap":

**1. XGBoost SHAP** (`analysis/shap_explainer.py`)
- Per-trade feature contribution
- Global importance over time
- "Why did the model buy on 2024-03-14?"

**2. Ensemble Explainer** (`analysis/ensemble_explainer.py`)
- Agreement Score between XGBoost and LSTM
- LSTM sensitivity via input perturbation (CPU-friendly GradientExplainer proxy)
- Warns when SHAP is only "half the truth" (low agreement)
- Side-by-side XGB vs LSTM feature importance comparison

> Key finding: XGBoost drives timing signals (momentum/RSI), LSTM drives risk control (volatility regime). Agreement rate: ~52% — the LSTM actively overrides XGBoost in volatile periods.

---

## Backtest Realism

| Parameter | Value |
|-----------|-------|
| Commission | 0.05% per trade (5bps) |
| Slippage | 0.01% (1bp) worse than close |
| Walk-forward | 2yr train / 1mo test windows |
| Data version | DVC tracked |
| Experiments | MLflow logged |

---

## Key Findings

1. **QQQ > SPY** — Nasdaq more predictable with momentum features
2. **VIX Z-Score is more important than RSI** as a signal driver
3. **2-year train window** — best balance of recency vs. data volume
4. **Threshold 0.50** — lower threshold = more trades = better results
5. **Walk-forward mandatory** — simple train/test caused Sharpe of -2.5
6. **Kelly Criterion** recommends ~31% position (our default: 35%)
7. **75.4% profitable** in Monte Carlo — strategy is robust
