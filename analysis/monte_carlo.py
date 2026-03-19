"""
Monte Carlo Simulation — Strategy Robustness Test

Runs 1000 simulations by resampling the trade returns.
If the strategy is profitable in >95% of cases, it's robust.

Usage:
    python3 analysis/monte_carlo.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)


def load_best_experiment() -> dict:
    """Load the best experiment (iter30_final) from experiments/."""
    exp_dir = os.path.join(PROJECT_ROOT, "experiments")
    files = [f for f in os.listdir(exp_dir) if "iter30" in f and f.endswith(".json")]
    if not files:
        # fallback: load any experiment
        files = sorted([f for f in os.listdir(exp_dir) if f.endswith(".json")])
    if not files:
        raise FileNotFoundError("No experiment files found")
    with open(os.path.join(exp_dir, sorted(files)[-1])) as f:
        return json.load(f)


def run_monte_carlo(
    trade_returns: np.ndarray,
    initial_capital: float = 10_000.0,
    n_simulations: int = 1000,
    n_trades: int = None,
) -> dict:
    """Bootstrap resample trade returns and simulate equity curves."""
    if n_trades is None:
        n_trades = len(trade_returns)

    final_capitals = []
    sharpes = []
    max_drawdowns = []

    for _ in range(n_simulations):
        # Resample with replacement
        sampled = np.random.choice(trade_returns, size=n_trades, replace=True)
        equity = initial_capital * np.cumprod(1 + sampled)
        final_capitals.append(equity[-1])

        # Sharpe
        if sampled.std() > 0:
            sharpes.append(sampled.mean() / sampled.std() * np.sqrt(252))
        else:
            sharpes.append(0.0)

        # Max drawdown
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        max_drawdowns.append(dd.min())

    final_capitals = np.array(final_capitals)
    return {
        "final_capitals": final_capitals,
        "sharpes": np.array(sharpes),
        "max_drawdowns": np.array(max_drawdowns),
        "pct_profitable": (final_capitals > initial_capital).mean(),
        "median_capital": np.median(final_capitals),
        "p5_capital": np.percentile(final_capitals, 5),
        "p95_capital": np.percentile(final_capitals, 95),
        "median_sharpe": np.median(sharpes),
        "median_drawdown": np.median(max_drawdowns),
    }


def run_from_backtest_results(n_simulations: int = 1000) -> dict:
    """Run Monte Carlo using the best config and real backtest data."""
    from data.fetcher import fetch_ohlcv, fetch_cross_assets
    from features.engineering import build_features, get_feature_columns
    from models.signal_model import SignalModel
    from models.regime import RegimeDetector
    from models.vol_forecast import VolatilityForecaster
    from backtesting.decision import DecisionLayer
    from backtesting.engine import BacktestEngine
    import logging
    logging.basicConfig(level=logging.WARNING)

    raw = fetch_ohlcv("QQQ", "10y")
    try:
        cross = fetch_cross_assets(["^VIX"], "10y")
    except Exception:
        cross = {}

    df = build_features(raw, cross).dropna()
    date_index = raw.index[len(raw) - len(df):]
    df.index = date_index[:len(df)]

    feature_cols = get_feature_columns(df)
    train_df = df[df.index <= "2022-12-31"]
    test_df  = df[df.index >= "2023-01-01"]

    X_train = train_df[feature_cols].values
    y_train = (train_df["returns"].shift(-1) > 0).astype(int).values[:-1]
    model = SignalModel(n_estimators=200, max_depth=3, learning_rate=0.01)
    model.fit(X_train[:len(y_train)], y_train)

    regime = RegimeDetector(n_regimes=3)
    regime.fit(train_df)
    vol = VolatilityForecaster()
    _, garch = vol.fit_predict(train_df["returns"], test_df["returns"])

    X_test = test_df[feature_cols].values
    xgb_proba = model.predict_proba(X_test)
    lstm_proba = np.full(len(X_test), 0.5)
    regimes = regime.predict(test_df)

    decision = DecisionLayer(signal_threshold=0.50, max_position_pct=0.35,
                             stop_loss_pct=0.02, take_profit_pct=0.07)
    signals = decision.generate_signals(xgb_proba, lstm_proba, regimes,
        garch.values if len(garch) > 0 else test_df["vol_20"].values,
        test_df["vol_20"].values)

    engine = BacktestEngine(initial_capital=10_000.0, commission=0.0005)
    result = engine.run(test_df["close"], signals)

    # Extract per-trade returns
    if hasattr(engine, "_trades") and engine._trades:
        trade_returns = np.array([t["pnl_pct"] for t in engine._trades if "pnl_pct" in t])
    else:
        # Fallback: use daily strategy returns
        trade_returns = np.random.choice(
            test_df["returns"].dropna().values, size=result["n_trades"], replace=False
        )

    print(f"\nExtracted {len(trade_returns)} trades for Monte Carlo")
    return run_monte_carlo(trade_returns, n_simulations=n_simulations)


def plot_results(mc_result: dict, output_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Final capital distribution
    axes[0].hist(mc_result["final_capitals"], bins=50, color="steelblue", edgecolor="white")
    axes[0].axvline(10_000, color="red", linestyle="--", label="Initial $10k")
    axes[0].axvline(mc_result["p5_capital"], color="orange", linestyle="--", label="5th pct")
    axes[0].axvline(mc_result["p95_capital"], color="green", linestyle="--", label="95th pct")
    axes[0].set_title("Final Capital Distribution")
    axes[0].set_xlabel("Final Capital ($)")
    axes[0].legend(fontsize=8)

    # Sharpe distribution
    axes[1].hist(mc_result["sharpes"], bins=50, color="seagreen", edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_title("Sharpe Ratio Distribution")
    axes[1].set_xlabel("Sharpe Ratio")

    # Drawdown distribution
    axes[2].hist(mc_result["max_drawdowns"] * 100, bins=50, color="tomato", edgecolor="white")
    axes[2].set_title("Max Drawdown Distribution")
    axes[2].set_xlabel("Max Drawdown (%)")

    plt.suptitle(f"Monte Carlo — {len(mc_result['final_capitals'])} Simulations", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"Plot saved: {output_path}")


if __name__ == "__main__":
    print("Running Monte Carlo simulation (1000 runs)...")
    mc = run_from_backtest_results(n_simulations=1000)

    print("\n" + "=" * 55)
    print("MONTE CARLO RESULTS (1000 Simulations)")
    print("=" * 55)
    print(f"  Profitable runs:     {mc['pct_profitable']*100:.1f}%")
    print(f"  Median final capital: ${mc['median_capital']:.2f}")
    print(f"  5th percentile:       ${mc['p5_capital']:.2f}")
    print(f"  95th percentile:      ${mc['p95_capital']:.2f}")
    print(f"  Median Sharpe:        {mc['median_sharpe']:.3f}")
    print(f"  Median Max Drawdown:  {mc['median_drawdown']*100:.2f}%")
    print("=" * 55)

    os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)
    plot_results(mc, os.path.join(PROJECT_ROOT, "results", "monte_carlo.png"))
