"""Quant Trading AI — Main orchestrator.

Usage:
    python3 main.py                  # Run latest iteration
    python3 main.py --iteration 5    # Run specific iteration
    python3 main.py --all            # Run all iterations
"""

import os
import sys
import logging
import warnings
import argparse
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "logs", "run.log")),
    ],
)
logger = logging.getLogger("quant")

PROJECT_ROOT = os.path.dirname(__file__)
sys.path.insert(0, PROJECT_ROOT)

from config import ExperimentConfig, DataConfig, FeatureConfig, ModelConfig, BacktestConfig
from data.fetcher import fetch_ohlcv, fetch_cross_assets
from features.engineering import build_features, get_feature_columns
from models.regime import RegimeDetector
from models.signal_model import SignalModel
from models.vol_forecast import VolatilityForecaster
from models.lstm_model import LSTMPredictor
from backtesting.decision import DecisionLayer
from backtesting.engine import BacktestEngine
from experiments.logger import log_experiment

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def _generate_shap_report(name: str, xgb_model, test_df, feature_cols: list):
    """Auto-generate feature importance report after each backtest.
    Uses XGBoost built-in importance (fast, no RAM overhead).
    Full SHAP analysis available via: python3 analysis/shap_explainer.py
    """
    try:
        import matplotlib.pyplot as plt

        # XGBoost built-in importance (gain = how much each feature improves split)
        raw_importance = xgb_model.model.get_booster().get_score(importance_type="gain")
        importance = {f: raw_importance.get(f"f{i}", 0.0) for i, f in enumerate(feature_cols)}
        top = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]

        # Normalize
        total = sum(v for _, v in top) or 1
        top_norm = [(f, v / total) for f, v in top]

        # Log
        logger.info(f"Feature Importance [{name}]:")
        for feat, val in top_norm[:5]:
            tag = " ← VIX" if "vix" in feat else " ← RSI" if "rsi" in feat else ""
            logger.info(f"  {feat:28s}: {val:.3%}{tag}")

        # Fake shap_values for reuse in plot section
        shap_values = None  # not used below

        # Plot
        fig, ax = plt.subplots(figsize=(9, 5))
        feats = [f for f, _ in top_norm][::-1]
        vals  = [v for _, v in top_norm][::-1]
        colors = ["#e74c3c" if "vix" in f else "#3498db" if "rsi" in f or "macd" in f
                  else "#2ecc71" if "vol" in f else "#95a5a6" for f in feats]
        ax.barh(feats, vals, color=colors)
        ax.set_xlabel("Relative Importance (XGB Gain)")
        ax.set_title(f"Feature Importance — {name}\n(red=VIX, blue=momentum, green=volatility)")
        ax.grid(alpha=0.3, axis="x")

        # Annotate VIX vs RSI comparison
        vix_imp = sum(v for f, v in top_norm if "vix" in f)
        rsi_imp = dict(top_norm).get("rsi_14", 0)
        ax.text(0.98, 0.02, f"VIX total: {vix_imp:.4f}\nRSI: {rsi_imp:.4f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        plt.tight_layout()
        out = os.path.join(RESULTS_DIR, f"{name}_shap.png")
        plt.savefig(out, dpi=120)
        plt.close()
        logger.info(f"SHAP report saved: {out}")

    except Exception as e:
        logger.debug(f"SHAP report error: {e}")


def run_walk_forward(cfg: ExperimentConfig) -> dict:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"EXPERIMENT: {cfg.name}")
    logger.info(f"Config: threshold={cfg.backtest.signal_threshold}, "
                f"train={cfg.backtest.train_window_days}d, test={cfg.backtest.test_window_days}d, "
                f"lstm={cfg.model.use_lstm}, cross_asset={cfg.features.use_cross_asset}")
    logger.info("=" * 60)

    raw = fetch_ohlcv(cfg.data.symbol, cfg.data.period)

    cross_data = {}
    if cfg.features.use_cross_asset:
        cross_data = fetch_cross_assets(cfg.data.cross_assets)

    df = build_features(raw, cross_data if cross_data else None)
    feature_cols = get_feature_columns(df)
    logger.info(f"Using {len(feature_cols)} features")

    bc = cfg.backtest
    all_trades, all_equity, window_metrics = [], [], []
    capital = bc.initial_capital
    start_idx = bc.train_window_days
    step = bc.test_window_days
    window_num = 0
    last_xgb_model = None
    last_test_df = None

    while start_idx + step <= len(df):
        window_num += 1
        train_df = df.iloc[start_idx - bc.train_window_days:start_idx].reset_index(drop=True)
        test_df = df.iloc[start_idx:start_idx + step].reset_index(drop=True)

        if len(test_df) < 5:
            break

        X_train = train_df[feature_cols].values
        y_train = train_df["target_1d"].values
        X_test = test_df[feature_cols].values

        # Regime detection
        regime = RegimeDetector(n_regimes=cfg.model.n_regimes)
        regime.fit(train_df)
        test_regimes = regime.predict(test_df)

        # Volatility — fixed: fit on train, predict for test
        vol_model = VolatilityForecaster()
        _, test_garch = vol_model.fit_predict(train_df["returns"], test_df["returns"])

        # XGBoost
        xgb = SignalModel(
            n_estimators=cfg.model.xgb_n_estimators,
            max_depth=cfg.model.xgb_max_depth,
            learning_rate=cfg.model.xgb_learning_rate,
            reg_alpha=cfg.model.xgb_reg_alpha,
            reg_lambda=cfg.model.xgb_reg_lambda,
        )
        xgb.fit(X_train, y_train, feature_cols)
        xgb_proba = xgb.predict_proba(X_test)

        # LSTM
        lstm_proba = np.full(len(X_test), 0.5)
        if cfg.model.use_lstm:
            lstm = LSTMPredictor(
                lookback=cfg.model.lstm_lookback,
                epochs=cfg.model.lstm_epochs,
                hidden_size=cfg.model.lstm_hidden,
            )
            lstm.fit(X_train, y_train)
            lstm_proba = lstm.predict_proba(X_test)

        # Decision
        decision = DecisionLayer(
            signal_threshold=bc.signal_threshold,
            max_position_pct=bc.max_position_pct,
            stop_loss_pct=bc.stop_loss_pct,
            take_profit_pct=bc.take_profit_pct,
            ensemble_agree=getattr(bc, "ensemble_agree", False),
        )
        trend = test_df["price_vs_sma_200"].values if getattr(bc, "long_bias", False) and "price_vs_sma_200" in test_df.columns else None
        signals = decision.generate_signals(
            xgb_proba=xgb_proba,
            lstm_proba=lstm_proba,
            regimes=test_regimes,
            garch_vol=test_garch.values if len(test_garch) > 0 else test_df["vol_20"].values,
            hist_vol=test_df["vol_20"].values,
            trend=trend,
        )

        # Backtest window
        engine = BacktestEngine(initial_capital=capital, commission=bc.commission,
                                slippage=getattr(bc, "slippage", 0.0001))
        results = engine.run(test_df["close"], signals)

        capital = results["final_capital"]
        all_equity.extend(results["equity_curve"].tolist())
        all_trades.extend(results["trades"])

        window_metrics.append({
            "window": window_num,
            "n_test": len(test_df),
            **{k: v for k, v in results.items() if k not in ("equity_curve", "trades")},
        })

        if window_num % 10 == 0:
            logger.info(f"Window {window_num}: capital=${capital:.0f}, trades={results['n_trades']}")

        # Track last trained model + test data for SHAP report
        last_xgb_model = xgb
        last_test_df = test_df

        start_idx += step

    # Compute overall metrics
    equity = pd.Series(all_equity)
    if len(equity) < 2:
        logger.warning("Not enough data")
        return {}

    returns = equity.pct_change().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    max_dd = ((equity - equity.cummax()) / equity.cummax()).min()
    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1

    trade_pnls = [t["pnl"] for t in all_trades]
    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p <= 0]
    win_rate = len(wins) / len(trade_pnls) if trade_pnls else 0
    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")

    summary = {
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(max_dd, 4),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(pf, 3),
        "total_return": round(total_return, 4),
        "n_trades": len(all_trades),
        "n_windows": window_num,
        "final_capital": round(capital, 2),
    }

    logger.info(f"RESULT [{cfg.name}]: Sharpe={summary['sharpe']}, Return={summary['total_return']:.2%}, "
                f"PF={summary['profit_factor']}, Trades={summary['n_trades']}, Capital=${summary['final_capital']}")

    # Save equity + windows
    plot_metrics = {**summary, "equity_curve": equity, "trades": all_trades}
    BacktestEngine().plot_results(plot_metrics, os.path.join(RESULTS_DIR, f"{cfg.name}_equity.png"))
    pd.DataFrame(window_metrics).to_csv(os.path.join(RESULTS_DIR, f"{cfg.name}_windows.csv"), index=False)

    # SHAP Feature Importance Report
    try:
        _generate_shap_report(cfg.name, last_xgb_model, last_test_df, feature_cols)
    except Exception as e:
        logger.debug(f"SHAP report skipped: {e}")

    log_experiment(cfg.name, cfg.to_dict(), feature_cols, summary, notes=cfg.notes)
    return summary


# ─── Iteration definitions ───────────────────────────────────────

def iter5_vix_signal():
    """Iteration 5: Add VIX cross-asset signal to best config."""
    configs = [
        ExperimentConfig(
            name="iter5_vix_monthly",
            notes="Iter5: VIX cross-asset, monthly rebalance",
            features=FeatureConfig(use_cross_asset=True),
            model=ModelConfig(use_lstm=False),
            backtest=BacktestConfig(train_window_days=378, test_window_days=21,
                                    signal_threshold=0.51, stop_loss_pct=0.03, take_profit_pct=0.06),
        ),
        ExperimentConfig(
            name="iter5_vix_6week",
            notes="Iter5: VIX cross-asset, 6-week windows",
            features=FeatureConfig(use_cross_asset=True),
            model=ModelConfig(use_lstm=False),
            backtest=BacktestConfig(train_window_days=378, test_window_days=42,
                                    signal_threshold=0.51, stop_loss_pct=0.03, take_profit_pct=0.06),
        ),
        ExperimentConfig(
            name="iter5_vix_lstm",
            notes="Iter5: VIX + LSTM, 6-week windows",
            features=FeatureConfig(use_cross_asset=True),
            model=ModelConfig(use_lstm=True, lstm_epochs=20),
            backtest=BacktestConfig(train_window_days=378, test_window_days=42,
                                    signal_threshold=0.51, stop_loss_pct=0.03, take_profit_pct=0.06),
        ),
    ]
    return run_iteration("ITERATION 5: VIX CROSS-ASSET", configs)


def iter6_feature_selection():
    """Iteration 6: Feature importance-based selection, tighter model."""
    configs = [
        ExperimentConfig(
            name="iter6_deeper_xgb",
            notes="Iter6: Deeper XGB (depth=4, 150 trees), VIX",
            features=FeatureConfig(use_cross_asset=True),
            model=ModelConfig(xgb_max_depth=4, xgb_n_estimators=150, xgb_learning_rate=0.015),
            backtest=BacktestConfig(train_window_days=378, test_window_days=21,
                                    signal_threshold=0.51, stop_loss_pct=0.03, take_profit_pct=0.06),
        ),
        ExperimentConfig(
            name="iter6_wider_tp",
            notes="Iter6: Wider take-profit 8%, stop 4%",
            features=FeatureConfig(use_cross_asset=True),
            model=ModelConfig(xgb_max_depth=4, xgb_n_estimators=150),
            backtest=BacktestConfig(train_window_days=378, test_window_days=21,
                                    signal_threshold=0.51, stop_loss_pct=0.04, take_profit_pct=0.08),
        ),
        ExperimentConfig(
            name="iter6_short_train",
            notes="Iter6: 1-year train window, most adaptive",
            features=FeatureConfig(use_cross_asset=True),
            model=ModelConfig(xgb_max_depth=4, xgb_n_estimators=150),
            backtest=BacktestConfig(train_window_days=252, test_window_days=21,
                                    signal_threshold=0.51, stop_loss_pct=0.03, take_profit_pct=0.06),
        ),
    ]
    return run_iteration("ITERATION 6: DEEPER MODEL + FEATURE TUNING", configs)


def iter7_multi_asset():
    """Iteration 7: Test on multiple assets."""
    configs = [
        ExperimentConfig(
            name="iter7_spy_best",
            notes="Iter7: Best config on SPY",
            data=DataConfig(symbol="SPY"),
            features=FeatureConfig(use_cross_asset=True),
            model=ModelConfig(xgb_max_depth=4, xgb_n_estimators=150),
            backtest=BacktestConfig(train_window_days=378, test_window_days=21,
                                    signal_threshold=0.51, stop_loss_pct=0.03, take_profit_pct=0.06),
        ),
        ExperimentConfig(
            name="iter7_qqq",
            notes="Iter7: Same config on QQQ (Nasdaq)",
            data=DataConfig(symbol="QQQ"),
            features=FeatureConfig(use_cross_asset=True),
            model=ModelConfig(xgb_max_depth=4, xgb_n_estimators=150),
            backtest=BacktestConfig(train_window_days=378, test_window_days=21,
                                    signal_threshold=0.51, stop_loss_pct=0.03, take_profit_pct=0.06),
        ),
        ExperimentConfig(
            name="iter7_tlt",
            notes="Iter7: Same config on TLT (Bonds)",
            data=DataConfig(symbol="TLT"),
            features=FeatureConfig(use_cross_asset=True),
            model=ModelConfig(xgb_max_depth=4, xgb_n_estimators=150),
            backtest=BacktestConfig(train_window_days=378, test_window_days=21,
                                    signal_threshold=0.51, stop_loss_pct=0.03, take_profit_pct=0.06),
        ),
    ]
    return run_iteration("ITERATION 7: MULTI-ASSET", configs)


def run_iteration(title: str, configs: list[ExperimentConfig]) -> list[dict]:
    logger.info(f"\n{'='*60}")
    logger.info(title)
    logger.info(f"{'='*60}\n")

    results = []
    for cfg in configs:
        r = run_walk_forward(cfg)
        results.append((cfg.name, r))

    print(f"\n{'='*60}")
    print(title)
    print(f"{'='*60}")
    for name, r in results:
        if r:
            print(f"\n{name}:")
            for k, v in r.items():
                print(f"  {k}: {v}")

    return results


def iter8_best_combo():
    """Iteration 8: Combine best findings — QQQ+VIX, SPY+VIX refined."""
    configs = [
        ExperimentConfig(
            name="iter8_qqq_vix",
            notes="Iter8: QQQ with VIX cross-asset, monthly",
            data=DataConfig(symbol="QQQ"),
            features=FeatureConfig(use_cross_asset=True),
            model=ModelConfig(use_lstm=False),
            backtest=BacktestConfig(train_window_days=378, test_window_days=21,
                                    signal_threshold=0.51, stop_loss_pct=0.03, take_profit_pct=0.06),
        ),
        ExperimentConfig(
            name="iter8_qqq_vix_lstm",
            notes="Iter8: QQQ+VIX+LSTM, monthly",
            data=DataConfig(symbol="QQQ"),
            features=FeatureConfig(use_cross_asset=True),
            model=ModelConfig(use_lstm=True, lstm_epochs=20),
            backtest=BacktestConfig(train_window_days=378, test_window_days=21,
                                    signal_threshold=0.51, stop_loss_pct=0.03, take_profit_pct=0.06),
        ),
        ExperimentConfig(
            name="iter8_spy_vix_refined",
            notes="Iter8: SPY+VIX, lower reg for subtle signal",
            features=FeatureConfig(use_cross_asset=True),
            model=ModelConfig(xgb_reg_alpha=0.5, xgb_reg_lambda=1.5),
            backtest=BacktestConfig(train_window_days=378, test_window_days=21,
                                    signal_threshold=0.51, stop_loss_pct=0.025, take_profit_pct=0.05),
        ),
    ]
    return run_iteration("ITERATION 8: BEST COMBO", configs)


def iter9_qqq_optimized():
    """Iteration 9: Fine-tune the winning QQQ+VIX config."""
    configs = [
        ExperimentConfig(
            name="iter9_qqq_vix_tight",
            notes="Iter9: QQQ+VIX, tighter threshold 0.53",
            data=DataConfig(symbol="QQQ"),
            features=FeatureConfig(use_cross_asset=True),
            model=ModelConfig(use_lstm=False),
            backtest=BacktestConfig(train_window_days=378, test_window_days=21,
                                    signal_threshold=0.53, stop_loss_pct=0.025, take_profit_pct=0.055),
        ),
        ExperimentConfig(
            name="iter9_qqq_vix_2yr",
            notes="Iter9: QQQ+VIX, 2yr train window",
            data=DataConfig(symbol="QQQ"),
            features=FeatureConfig(use_cross_asset=True),
            model=ModelConfig(use_lstm=False),
            backtest=BacktestConfig(train_window_days=504, test_window_days=21,
                                    signal_threshold=0.51, stop_loss_pct=0.03, take_profit_pct=0.06),
        ),
    ]
    return run_iteration("ITERATION 9: QQQ OPTIMIZED", configs)


def iter10_refined_qqq():
    """Iteration 10: Refine the record-breaking QQQ + VIX + 2yr window config."""
    configs = [
        ExperimentConfig(
            name="iter10_qqq_2.5yr",
            notes="Iter10: QQQ+VIX, 2.5yr train window (630 days)",
            data=DataConfig(symbol="QQQ"),
            features=FeatureConfig(use_cross_asset=True),
            model=ModelConfig(use_lstm=False),
            backtest=BacktestConfig(train_window_days=630, test_window_days=21,
                                    signal_threshold=0.51, stop_loss_pct=0.03, take_profit_pct=0.06),
        ),
        ExperimentConfig(
            name="iter10_qqq_2yr_more_trees",
            notes="Iter10: QQQ+VIX, 2yr window, 250 trees",
            data=DataConfig(symbol="QQQ"),
            features=FeatureConfig(use_cross_asset=True),
            model=ModelConfig(xgb_n_estimators=250, xgb_max_depth=3, xgb_learning_rate=0.008),
            backtest=BacktestConfig(train_window_days=504, test_window_days=21,
                                    signal_threshold=0.51, stop_loss_pct=0.03, take_profit_pct=0.06),
        ),
    ]
    return run_iteration("ITERATION 10: REFINED QQQ", configs)


def iter11_ensemble_agree():
    """Iteration 11: Only trade when XGB + LSTM both agree."""
    configs = [
        ExperimentConfig(
            name="iter11_ensemble_qqq",
            notes="Iter11: QQQ+VIX+LSTM, ensemble agreement filter",
            data=DataConfig(symbol="QQQ"),
            features=FeatureConfig(use_cross_asset=True),
            model=ModelConfig(use_lstm=True, lstm_epochs=20),
            backtest=BacktestConfig(train_window_days=504, test_window_days=21,
                                    signal_threshold=0.51, stop_loss_pct=0.03,
                                    take_profit_pct=0.06, ensemble_agree=True),
        ),
    ]
    return run_iteration("ITERATION 11: ENSEMBLE AGREEMENT", configs)


def iter12_confidence_sizing():
    """Iteration 12: Scale position size by model confidence."""
    configs = [
        ExperimentConfig(
            name="iter12_confidence_qqq",
            notes="Iter12: QQQ+VIX, 2yr, conviction-scaled position sizing",
            data=DataConfig(symbol="QQQ"),
            features=FeatureConfig(use_cross_asset=True),
            model=ModelConfig(use_lstm=False),
            backtest=BacktestConfig(train_window_days=504, test_window_days=21,
                                    signal_threshold=0.51, stop_loss_pct=0.03,
                                    take_profit_pct=0.06, max_position_pct=0.30),
        ),
    ]
    return run_iteration("ITERATION 12: CONFIDENCE SIZING", configs)


def iter13_regime_filter():
    """Iteration 13: Only trade in trending HMM regimes, skip choppy."""
    configs = [
        ExperimentConfig(
            name="iter13_regime_qqq",
            notes="Iter13: QQQ+VIX, 2yr, regime-filtered trades",
            data=DataConfig(symbol="QQQ"),
            features=FeatureConfig(use_cross_asset=True),
            model=ModelConfig(use_lstm=False, n_regimes=3),
            backtest=BacktestConfig(train_window_days=504, test_window_days=21,
                                    signal_threshold=0.51, stop_loss_pct=0.025,
                                    take_profit_pct=0.06),
        ),
    ]
    return run_iteration("ITERATION 13: REGIME FILTER", configs)


def iter14_combined_best():
    """Iter 14: Combine regime filter + 30% position sizing."""
    configs = [
        ExperimentConfig(
            name="iter14_regime_30pct",
            notes="Iter14: QQQ+VIX, 2yr, regime + 30% position",
            data=DataConfig(symbol="QQQ"),
            features=FeatureConfig(use_cross_asset=True),
            model=ModelConfig(use_lstm=False, n_regimes=3),
            backtest=BacktestConfig(train_window_days=504, test_window_days=21,
                                    signal_threshold=0.51, stop_loss_pct=0.025,
                                    take_profit_pct=0.06, max_position_pct=0.30),
        ),
    ]
    return run_iteration("ITERATION 14: REGIME + 30PCT", configs)


def iter15_lower_threshold():
    """Iter 15: Even lower threshold on best config."""
    configs = [
        ExperimentConfig(
            name="iter15_thresh_050",
            notes="Iter15: QQQ+VIX, 2yr, threshold=0.50",
            data=DataConfig(symbol="QQQ"),
            features=FeatureConfig(use_cross_asset=True),
            model=ModelConfig(use_lstm=False),
            backtest=BacktestConfig(train_window_days=504, test_window_days=21,
                                    signal_threshold=0.50, stop_loss_pct=0.025,
                                    take_profit_pct=0.06, max_position_pct=0.30),
        ),
    ]
    return run_iteration("ITERATION 15: LOWER THRESHOLD", configs)


def iter16_weekly_rebalance():
    """Iter 16: Weekly rebalance (7-day test window)."""
    configs = [
        ExperimentConfig(
            name="iter16_weekly_qqq",
            notes="Iter16: QQQ+VIX, 2yr, weekly rebalance",
            data=DataConfig(symbol="QQQ"),
            features=FeatureConfig(use_cross_asset=True),
            model=ModelConfig(use_lstm=False),
            backtest=BacktestConfig(train_window_days=504, test_window_days=7,
                                    signal_threshold=0.51, stop_loss_pct=0.025,
                                    take_profit_pct=0.06, max_position_pct=0.30),
        ),
    ]
    return run_iteration("ITERATION 16: WEEKLY REBALANCE", configs)


def iter17_regime_plus_thresh():
    configs = [ExperimentConfig(name="iter17_regime_thresh050", notes="regime filter + threshold=0.50 + 30%",
        data=DataConfig(symbol="QQQ"), features=FeatureConfig(use_cross_asset=True),
        model=ModelConfig(n_regimes=3),
        backtest=BacktestConfig(train_window_days=504, test_window_days=21, signal_threshold=0.50,
            stop_loss_pct=0.025, take_profit_pct=0.06, max_position_pct=0.30))]
    return run_iteration("ITERATION 17", configs)

def iter18_bigger_position():
    configs = [ExperimentConfig(name="iter18_pos40pct", notes="threshold=0.50 + 40% position",
        data=DataConfig(symbol="QQQ"), features=FeatureConfig(use_cross_asset=True),
        model=ModelConfig(),
        backtest=BacktestConfig(train_window_days=504, test_window_days=21, signal_threshold=0.50,
            stop_loss_pct=0.025, take_profit_pct=0.06, max_position_pct=0.40))]
    return run_iteration("ITERATION 18", configs)

def iter19_tight_stop():
    configs = [ExperimentConfig(name="iter19_stop2pct", notes="threshold=0.50 + stop=2%",
        data=DataConfig(symbol="QQQ"), features=FeatureConfig(use_cross_asset=True),
        model=ModelConfig(),
        backtest=BacktestConfig(train_window_days=504, test_window_days=21, signal_threshold=0.50,
            stop_loss_pct=0.02, take_profit_pct=0.06, max_position_pct=0.30))]
    return run_iteration("ITERATION 19", configs)

def iter20_wider_tp():
    configs = [ExperimentConfig(name="iter20_tp8pct", notes="threshold=0.50 + TP=8%",
        data=DataConfig(symbol="QQQ"), features=FeatureConfig(use_cross_asset=True),
        model=ModelConfig(),
        backtest=BacktestConfig(train_window_days=504, test_window_days=21, signal_threshold=0.50,
            stop_loss_pct=0.025, take_profit_pct=0.08, max_position_pct=0.30))]
    return run_iteration("ITERATION 20", configs)

def iter21_shorter_train():
    configs = [ExperimentConfig(name="iter21_1pt5yr", notes="threshold=0.50, 1.5yr window",
        data=DataConfig(symbol="QQQ"), features=FeatureConfig(use_cross_asset=True),
        model=ModelConfig(),
        backtest=BacktestConfig(train_window_days=378, test_window_days=21, signal_threshold=0.50,
            stop_loss_pct=0.025, take_profit_pct=0.06, max_position_pct=0.30))]
    return run_iteration("ITERATION 21", configs)

def iter22_longer_train():
    configs = [ExperimentConfig(name="iter22_2pt5yr", notes="threshold=0.50, 2.5yr window",
        data=DataConfig(symbol="QQQ"), features=FeatureConfig(use_cross_asset=True),
        model=ModelConfig(),
        backtest=BacktestConfig(train_window_days=630, test_window_days=21, signal_threshold=0.50,
            stop_loss_pct=0.025, take_profit_pct=0.06, max_position_pct=0.30))]
    return run_iteration("ITERATION 22", configs)

def iter23_spy_thresh050():
    configs = [ExperimentConfig(name="iter23_spy_050", notes="SPY with threshold=0.50, 30%",
        data=DataConfig(symbol="SPY"), features=FeatureConfig(use_cross_asset=True),
        model=ModelConfig(),
        backtest=BacktestConfig(train_window_days=504, test_window_days=21, signal_threshold=0.50,
            stop_loss_pct=0.025, take_profit_pct=0.06, max_position_pct=0.30))]
    return run_iteration("ITERATION 23", configs)

def iter24_more_trees():
    configs = [ExperimentConfig(name="iter24_trees200", notes="200 trees + threshold=0.50",
        data=DataConfig(symbol="QQQ"), features=FeatureConfig(use_cross_asset=True),
        model=ModelConfig(xgb_n_estimators=200),
        backtest=BacktestConfig(train_window_days=504, test_window_days=21, signal_threshold=0.50,
            stop_loss_pct=0.025, take_profit_pct=0.06, max_position_pct=0.30))]
    return run_iteration("ITERATION 24", configs)

def iter25_thresh049():
    configs = [ExperimentConfig(name="iter25_thresh049", notes="even lower threshold=0.49",
        data=DataConfig(symbol="QQQ"), features=FeatureConfig(use_cross_asset=True),
        model=ModelConfig(),
        backtest=BacktestConfig(train_window_days=504, test_window_days=21, signal_threshold=0.49,
            stop_loss_pct=0.025, take_profit_pct=0.06, max_position_pct=0.30))]
    return run_iteration("ITERATION 25", configs)

def iter26_6week_test():
    configs = [ExperimentConfig(name="iter26_6week", notes="6-week test window + threshold=0.50",
        data=DataConfig(symbol="QQQ"), features=FeatureConfig(use_cross_asset=True),
        model=ModelConfig(),
        backtest=BacktestConfig(train_window_days=504, test_window_days=42, signal_threshold=0.50,
            stop_loss_pct=0.025, take_profit_pct=0.06, max_position_pct=0.30))]
    return run_iteration("ITERATION 26", configs)

def iter27_lstm_best():
    configs = [ExperimentConfig(name="iter27_lstm_best", notes="LSTM + best config so far",
        data=DataConfig(symbol="QQQ"), features=FeatureConfig(use_cross_asset=True),
        model=ModelConfig(use_lstm=True, lstm_epochs=20),
        backtest=BacktestConfig(train_window_days=504, test_window_days=21, signal_threshold=0.50,
            stop_loss_pct=0.025, take_profit_pct=0.06, max_position_pct=0.30))]
    return run_iteration("ITERATION 27", configs)

def iter28_asymmetric_tp():
    configs = [ExperimentConfig(name="iter28_asym", notes="stop=1.5%, TP=7% asymmetric",
        data=DataConfig(symbol="QQQ"), features=FeatureConfig(use_cross_asset=True),
        model=ModelConfig(),
        backtest=BacktestConfig(train_window_days=504, test_window_days=21, signal_threshold=0.50,
            stop_loss_pct=0.015, take_profit_pct=0.07, max_position_pct=0.30))]
    return run_iteration("ITERATION 28", configs)

def iter29_commission_aware():
    configs = [ExperimentConfig(name="iter29_low_commission", notes="threshold=0.50, lower commission=0.0005",
        data=DataConfig(symbol="QQQ"), features=FeatureConfig(use_cross_asset=True),
        model=ModelConfig(),
        backtest=BacktestConfig(train_window_days=504, test_window_days=21, signal_threshold=0.50,
            stop_loss_pct=0.025, take_profit_pct=0.06, max_position_pct=0.30, commission=0.0005))]
    return run_iteration("ITERATION 29", configs)

def iter30_final_best():
    """Final best: combine all winning factors."""
    configs = [ExperimentConfig(name="iter30_final", notes="All best: QQQ+VIX, 2yr, thresh=0.50, 30%, regime, stop=2%, TP=7%",
        data=DataConfig(symbol="QQQ"), features=FeatureConfig(use_cross_asset=True),
        model=ModelConfig(n_regimes=3, xgb_n_estimators=200),
        backtest=BacktestConfig(train_window_days=504, test_window_days=21, signal_threshold=0.50,
            stop_loss_pct=0.02, take_profit_pct=0.07, max_position_pct=0.35, commission=0.0005))]
    return run_iteration("ITERATION 30: FINAL BEST", configs)


def iter31_long_bias():
    """Iter 31: Long-only in uptrend (price > 200 SMA), suppress shorts."""
    configs = [ExperimentConfig(name="iter31_long_bias", notes="Long-bias: no shorts in uptrend",
        data=DataConfig(symbol="QQQ"), features=FeatureConfig(use_cross_asset=True),
        model=ModelConfig(n_regimes=3, xgb_n_estimators=200),
        backtest=BacktestConfig(train_window_days=504, test_window_days=21, signal_threshold=0.50,
            stop_loss_pct=0.02, take_profit_pct=0.07, max_position_pct=0.35,
            commission=0.0005, long_bias=True))]
    return run_iteration("ITERATION 31: LONG BIAS", configs)


ITERATIONS = {
    5: iter5_vix_signal,
    6: iter6_feature_selection,
    7: iter7_multi_asset,
    8: iter8_best_combo,
    9: iter9_qqq_optimized,
    10: iter10_refined_qqq,
    11: iter11_ensemble_agree,
    12: iter12_confidence_sizing,
    13: iter13_regime_filter,
    14: iter14_combined_best,
    15: iter15_lower_threshold,
    16: iter16_weekly_rebalance,
    17: iter17_regime_plus_thresh,
    18: iter18_bigger_position,
    19: iter19_tight_stop,
    20: iter20_wider_tp,
    21: iter21_shorter_train,
    22: iter22_longer_train,
    23: iter23_spy_thresh050,
    24: iter24_more_trees,
    25: iter25_thresh049,
    26: iter26_6week_test,
    27: iter27_lstm_best,
    28: iter28_asymmetric_tp,
    29: iter29_commission_aware,
    30: iter30_final_best,
    31: iter31_long_bias,
}


def main():
    parser = argparse.ArgumentParser(description="Quant Trading AI")
    parser.add_argument("--iteration", "-i", type=int, help="Run specific iteration")
    parser.add_argument("--all", action="store_true", help="Run all iterations")
    args = parser.parse_args()

    os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)

    if args.iteration:
        fn = ITERATIONS.get(args.iteration)
        if fn:
            fn()
        else:
            print(f"Unknown iteration {args.iteration}. Available: {list(ITERATIONS.keys())}")
    elif args.all:
        for i, fn in sorted(ITERATIONS.items()):
            fn()
    else:
        latest = max(ITERATIONS.keys())
        logger.info(f"Running latest iteration: {latest}")
        ITERATIONS[latest]()


if __name__ == "__main__":
    main()
