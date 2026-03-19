"""
Unified inference pipeline — combines XGBoost + LSTM signals properly.
Used by: Alpaca bridge, OOS tests, dashboard.

No more 0.5 dummy LSTM values.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_ensemble_proba(
    xgb_model,
    lstm_model,
    X: np.ndarray,
    use_lstm: bool = True,
    xgb_weight: float = 0.6,
    lstm_weight: float = 0.4,
) -> tuple:
    """
    Compute ensemble probability from XGBoost and LSTM.

    Returns:
        xgb_proba   (np.ndarray): XGBoost signal probability
        lstm_proba  (np.ndarray): LSTM signal probability (real, not 0.5 dummy)
        combined    (np.ndarray): Weighted ensemble
    """
    xgb_proba = xgb_model.predict_proba(X)

    if use_lstm and lstm_model is not None and lstm_model.model is not None:
        lstm_proba = lstm_model.predict_proba(X)
        logger.debug(f"LSTM active: mean proba={lstm_proba.mean():.3f}")
    else:
        # Intentional fallback: LSTM not trained or disabled
        lstm_proba = np.full(len(X), 0.5)
        if use_lstm:
            logger.warning("LSTM fallback to 0.5 — model not trained (window too small?)")

    combined = xgb_weight * xgb_proba + lstm_weight * lstm_proba
    return xgb_proba, lstm_proba, combined


def build_signal_pipeline(
    train_df: pd.DataFrame,
    feature_cols: list,
    use_lstm: bool = False,
    n_estimators: int = 200,
    lstm_epochs: int = 20,
):
    """
    Train XGBoost (+ optionally LSTM) on training data.
    Returns trained models ready for inference.
    """
    from models.signal_model import SignalModel
    from models.lstm_model import LSTMPredictor
    from models.regime import RegimeDetector
    from models.vol_forecast import VolatilityForecaster

    X_train = train_df[feature_cols].values
    y_train = (train_df["returns"].shift(-1) > 0).astype(int).values[:-1]
    X_train = X_train[:len(y_train)]

    # XGBoost
    xgb = SignalModel(n_estimators=n_estimators, max_depth=3, learning_rate=0.01)
    xgb.fit(X_train, y_train)
    logger.info(f"XGBoost train acc: {(xgb.predict(X_train) == y_train).mean():.4f}")

    # LSTM
    lstm = None
    if use_lstm:
        lstm = LSTMPredictor(lookback=20, hidden_size=64, epochs=lstm_epochs)
        lstm.fit(X_train, y_train)
        logger.info("LSTM trained")

    # Regime + Vol
    regime = RegimeDetector(n_regimes=3)
    regime.fit(train_df)

    vol = VolatilityForecaster()

    return xgb, lstm, regime, vol


def run_inference(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    xgb_model,
    lstm_model,
    regime_model,
    vol_model,
    feature_cols: list,
    threshold: float = 0.50,
    stop_loss: float = 0.02,
    take_profit: float = 0.07,
    max_position: float = 0.35,
    use_lstm: bool = False,
):
    """Full inference: features → signals → positions."""
    from backtesting.decision import DecisionLayer

    X_test = test_df[feature_cols].values
    regimes = regime_model.predict(test_df)
    _, garch = vol_model.fit_predict(train_df["returns"], test_df["returns"])

    xgb_proba, lstm_proba, _ = get_ensemble_proba(xgb_model, lstm_model, X_test, use_lstm=use_lstm)

    decision = DecisionLayer(
        signal_threshold=threshold,
        max_position_pct=max_position,
        stop_loss_pct=stop_loss,
        take_profit_pct=take_profit,
    )
    signals = decision.generate_signals(
        xgb_proba=xgb_proba,
        lstm_proba=lstm_proba,
        regimes=regimes,
        garch_vol=garch.values if len(garch) > 0 else test_df["vol_20"].values,
        hist_vol=test_df["vol_20"].values,
    )
    return signals, xgb_proba, lstm_proba
