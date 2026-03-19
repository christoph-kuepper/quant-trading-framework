"""
Out-of-Sample Test for the best config (Iter 30 params).

Train period: all data up to end of 2022
Test period:  2023-01-01 onward (never seen during hyperparameter tuning)
"""

import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("oos")

PROJECT_ROOT = os.path.dirname(__file__)
sys.path.insert(0, PROJECT_ROOT)

from data.fetcher import fetch_ohlcv, fetch_cross_assets
from features.engineering import build_features, get_feature_columns
from models.regime import RegimeDetector
from models.signal_model import SignalModel
from models.vol_forecast import VolatilityForecaster
from backtesting.decision import DecisionLayer
from backtesting.engine import BacktestEngine

# Best config from Iter 30
SYMBOL        = "QQQ"
TRAIN_CUTOFF  = "2022-12-31"
TEST_START    = "2023-01-01"
SIGNAL_THRESH = 0.50
STOP_LOSS     = 0.02
TAKE_PROFIT   = 0.07
MAX_POS       = 0.35
N_TREES       = 200
COMMISSION    = 0.0005
INITIAL_CAP   = 10_000.0


def run_oos():
    logger.info("=" * 60)
    logger.info("OUT-OF-SAMPLE TEST — Best Config (Iter 30 params)")
    logger.info(f"Train: up to {TRAIN_CUTOFF} | Test: {TEST_START} onward")
    logger.info("=" * 60)

    raw = fetch_ohlcv(SYMBOL, period="10y")
    cross_data = {}
    try:
        cross_data = fetch_cross_assets(["^VIX"], period="10y")
    except Exception:
        logger.warning("VIX fetch failed, running without cross-asset features")

    df = build_features(raw, cross_data)
    df = df.dropna()
    # Restore date index from raw (build_features resets it)
    date_index = raw.index[len(raw) - len(df):]
    df.index = date_index[:len(df)]

    feature_cols = get_feature_columns(df)
    logger.info(f"Features: {len(feature_cols)} columns")

    # Split
    train_df = df[df.index <= TRAIN_CUTOFF].copy()
    test_df  = df[df.index >= TEST_START].copy()

    logger.info(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")

    if len(train_df) < 200 or len(test_df) < 20:
        logger.error("Not enough data for OOS split")
        return

    X_train = train_df[feature_cols].values
    y_train = (train_df["returns"].shift(-1) > 0).astype(int).values[:-1]
    X_train = X_train[:-1]

    X_test = test_df[feature_cols].values
    y_test  = (test_df["returns"].shift(-1) > 0).astype(int).values[:-1]
    X_test_trim = X_test[:-1]

    # Regime
    regime_model = RegimeDetector(n_regimes=3)
    regime_model.fit(train_df)
    test_regimes = regime_model.predict(test_df)

    # Vol
    vol_model = VolatilityForecaster()
    _, test_garch = vol_model.fit_predict(train_df["returns"], test_df["returns"])

    # XGBoost
    xgb_model = SignalModel(n_estimators=N_TREES, max_depth=3, learning_rate=0.01)
    xgb_model.fit(X_train, y_train)
    train_acc = (xgb_model.predict(X_train) == y_train).mean()
    test_acc  = (xgb_model.predict(X_test_trim) == y_test).mean()
    logger.info(f"XGBoost — Train acc: {train_acc:.4f} | Test acc: {test_acc:.4f}")
    xgb_proba = xgb_model.predict_proba(X_test)

    # Dummy LSTM proba (0.5) — XGB-only for OOS
    lstm_proba = np.full(len(X_test), 0.5)

    # Signals
    decision = DecisionLayer(
        signal_threshold=SIGNAL_THRESH,
        max_position_pct=MAX_POS,
        stop_loss_pct=STOP_LOSS,
        take_profit_pct=TAKE_PROFIT,
    )
    signals = decision.generate_signals(
        xgb_proba=xgb_proba,
        lstm_proba=lstm_proba,
        regimes=test_regimes,
        garch_vol=test_garch.values if len(test_garch) > 0 else test_df["vol_20"].values,
        hist_vol=test_df["vol_20"].values,
    )

    # Backtest
    engine = BacktestEngine(initial_capital=INITIAL_CAP, commission=COMMISSION)
    result = engine.run(test_df["close"], signals)

    # Buy & Hold benchmark
    bh_return = (test_df["close"].iloc[-1] / test_df["close"].iloc[0]) - 1
    bh_capital = INITIAL_CAP * (1 + bh_return)

    print("\n" + "=" * 60)
    print("OUT-OF-SAMPLE RESULTS (2023-2025)")
    print("=" * 60)
    print(f"  Sharpe Ratio:    {result['sharpe']:.3f}")
    print(f"  Total Return:    {result['total_return']*100:.2f}%")
    print(f"  Final Capital:   ${result['final_capital']:.2f}")
    print(f"  Max Drawdown:    {result['max_drawdown']*100:.2f}%")
    print(f"  Win Rate:        {result['win_rate']*100:.1f}%")
    print(f"  Profit Factor:   {result['profit_factor']:.3f}")
    print(f"  Trades:          {result['n_trades']}")
    print()
    print(f"  Buy & Hold QQQ:  {bh_return*100:.2f}% (${bh_capital:.2f})")
    print("=" * 60)

    # Save
    os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)
    engine.save_plot(os.path.join(PROJECT_ROOT, "results", "oos_equity.png"))
    logger.info("OOS equity chart saved to results/oos_equity.png")


if __name__ == "__main__":
    run_oos()
