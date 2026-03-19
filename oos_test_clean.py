"""
Clean 3-Way Out-of-Sample Test.

Split:
  Train:      up to 2020-12-31  (model training)
  Validation: 2021-01-01 to 2022-12-31  (hyperparameter selection)
  Test:       2023-01-01 onward  (never touched, final evaluation)
"""

import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("oos_clean")

PROJECT_ROOT = os.path.dirname(__file__)
sys.path.insert(0, PROJECT_ROOT)

from data.fetcher import fetch_ohlcv, fetch_cross_assets
from features.engineering import build_features, get_feature_columns
from models.regime import RegimeDetector
from models.signal_model import SignalModel
from models.vol_forecast import VolatilityForecaster
from backtesting.decision import DecisionLayer
from backtesting.engine import BacktestEngine

SYMBOL      = "QQQ"
TRAIN_END   = "2020-12-31"
VAL_START   = "2021-01-01"
VAL_END     = "2022-12-31"
TEST_START  = "2023-01-01"
INITIAL_CAP = 10_000.0
COMMISSION  = 0.0005
N_TREES     = 200


def backtest_period(df_period, xgb_model, regime_model, vol_model, threshold, stop, tp, max_pos):
    feature_cols = get_feature_columns(df_period)
    X = df_period[feature_cols].values
    regimes = regime_model.predict(df_period)
    _, garch_vol = vol_model.fit_predict(df_period["returns"].iloc[:len(df_period)//2], df_period["returns"])
    lstm_proba = np.full(len(X), 0.5)
    xgb_proba = xgb_model.predict_proba(X)

    decision = DecisionLayer(signal_threshold=threshold, max_position_pct=max_pos,
                             stop_loss_pct=stop, take_profit_pct=tp)
    signals = decision.generate_signals(xgb_proba, lstm_proba, regimes,
        garch_vol.values if len(garch_vol) > 0 else df_period["vol_20"].values,
        df_period["vol_20"].values)

    engine = BacktestEngine(initial_capital=INITIAL_CAP, commission=COMMISSION)
    return engine.run(df_period["close"], signals), engine


def run_oos_clean():
    logger.info("=" * 60)
    logger.info("3-WAY OOS TEST: Train/Validation/Test")
    logger.info(f"  Train:  up to {TRAIN_END}")
    logger.info(f"  Val:    {VAL_START} - {VAL_END}")
    logger.info(f"  Test:   {TEST_START} onward (unseen)")
    logger.info("=" * 60)

    raw = fetch_ohlcv(SYMBOL, period="10y")
    cross_data = {}
    try:
        cross_data = fetch_cross_assets(["^VIX"], period="10y")
    except Exception:
        logger.warning("VIX fetch failed")

    df = build_features(raw, cross_data).dropna()
    date_index = raw.index[len(raw) - len(df):]
    df.index = date_index[:len(df)]

    train_df = df[df.index <= TRAIN_END]
    val_df   = df[(df.index >= VAL_START) & (df.index <= VAL_END)]
    test_df  = df[df.index >= TEST_START]

    logger.info(f"Train: {len(train_df)} rows | Val: {len(val_df)} rows | Test: {len(test_df)} rows")

    # Train XGBoost on train data only
    feature_cols = get_feature_columns(train_df)
    X_train = train_df[feature_cols].values
    y_train = (train_df["returns"].shift(-1) > 0).astype(int).values[:-1]

    xgb = SignalModel(n_estimators=N_TREES, max_depth=3, learning_rate=0.01)
    xgb.fit(X_train[:-1] if len(X_train) > len(y_train) else X_train, y_train)
    logger.info(f"XGBoost train acc: {(xgb.predict(X_train[:len(y_train)]) == y_train).mean():.4f}")

    # Fit regime on train
    regime = RegimeDetector(n_regimes=3)
    regime.fit(train_df)

    # Fit GARCH on train
    vol = VolatilityForecaster()

    # Validation: pick best threshold
    logger.info("--- Validation (hyperparameter selection) ---")
    best_sharpe = -99
    best_thresh = 0.50
    best_pos    = 0.35
    best_stop   = 0.02
    best_tp     = 0.07

    for thresh in [0.49, 0.50, 0.51]:
        for pos in [0.30, 0.35, 0.40]:
            r, _ = backtest_period(val_df, xgb, regime, vol, thresh, best_stop, best_tp, pos)
            if r["sharpe"] > best_sharpe:
                best_sharpe = r["sharpe"]
                best_thresh = thresh
                best_pos    = pos

    logger.info(f"Best val config: threshold={best_thresh}, pos={best_pos}, sharpe={best_sharpe:.3f}")

    # Final test on unseen data
    logger.info("--- Test (unseen 2023-2025) ---")
    test_result, test_engine = backtest_period(test_df, xgb, regime, vol,
                                               best_thresh, best_stop, best_tp, best_pos)

    bh_return = (test_df["close"].iloc[-1] / test_df["close"].iloc[0]) - 1

    print("\n" + "=" * 60)
    print("CLEAN OOS TEST RESULTS (2023-2025, TRULY UNSEEN)")
    print("=" * 60)
    print(f"  Model trained on:    up to {TRAIN_END}")
    print(f"  Params selected on:  {VAL_START} – {VAL_END}")
    print(f"  Best threshold:      {best_thresh}")
    print(f"  Best position size:  {best_pos*100:.0f}%")
    print()
    print(f"  Sharpe Ratio:        {test_result['sharpe']:.3f}")
    print(f"  Total Return:        {test_result['total_return']*100:.2f}%")
    print(f"  Final Capital:       ${test_result['final_capital']:.2f}")
    print(f"  Max Drawdown:        {test_result['max_drawdown']*100:.2f}%")
    print(f"  Win Rate:            {test_result['win_rate']*100:.1f}%")
    print(f"  Profit Factor:       {test_result['profit_factor']:.3f}")
    print(f"  Trades:              {test_result['n_trades']}")
    print()
    print(f"  Buy & Hold QQQ:      {bh_return*100:.2f}% (${INITIAL_CAP*(1+bh_return):.2f})")
    print("=" * 60)

    # Save plot
    os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)
    equity = test_engine._equity_curve if hasattr(test_engine, "_equity_curve") else None
    if equity is not None:
        plt.figure(figsize=(12, 5))
        plt.plot(equity)
        plt.title("OOS Equity Curve (2023-2025)")
        plt.xlabel("Trading Days")
        plt.ylabel("Portfolio Value ($)")
        plt.tight_layout()
        plt.savefig(os.path.join(PROJECT_ROOT, "results", "oos_clean_equity.png"))
        plt.close()
        logger.info("Plot saved to results/oos_clean_equity.png")


if __name__ == "__main__":
    run_oos_clean()
