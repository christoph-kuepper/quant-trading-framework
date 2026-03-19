import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DecisionLayer:
    def __init__(
        self,
        signal_threshold: float = 0.55,
        max_position_pct: float = 0.20,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
        vol_scaling: bool = True,
        ensemble_agree: bool = False,
    ):
        self.signal_threshold = signal_threshold
        self.max_position_pct = max_position_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.vol_scaling = vol_scaling
        self.ensemble_agree = ensemble_agree
        self.long_bias_uptrend = False

    def generate_signals(
        self,
        xgb_proba: np.ndarray,
        lstm_proba: np.ndarray,
        regimes: np.ndarray,
        garch_vol: np.ndarray,
        hist_vol: np.ndarray,
        trend: np.ndarray = None,
    ) -> pd.DataFrame:
        n = len(xgb_proba)
        combined_proba = 0.5 * xgb_proba + 0.3 * lstm_proba + 0.2 * 0.5  # ensemble

        signals = pd.DataFrame({
            "xgb_proba": xgb_proba,
            "lstm_proba": lstm_proba,
            "combined_proba": combined_proba,
            "regime": regimes[:n] if len(regimes) >= n else np.pad(regimes, (0, n - len(regimes)), constant_values=0),
            "garch_vol": garch_vol[:n] if len(garch_vol) >= n else np.pad(garch_vol, (0, n - len(garch_vol)), constant_values=np.nan),
        })

        # Signal: 1=long, -1=short, 0=flat
        signals["signal"] = 0
        if self.ensemble_agree:
            # Both XGB and LSTM must agree
            long_mask = (signals["xgb_proba"] > self.signal_threshold) & (signals["lstm_proba"] > self.signal_threshold)
            short_mask = (signals["xgb_proba"] < (1 - self.signal_threshold)) & (signals["lstm_proba"] < (1 - self.signal_threshold))
        else:
            long_mask = signals["combined_proba"] > self.signal_threshold
            short_mask = signals["combined_proba"] < (1 - self.signal_threshold)
        signals.loc[long_mask, "signal"] = 1
        signals.loc[short_mask, "signal"] = -1

        # Long-bias: suppress shorts when price is above 200-day SMA (uptrend)
        if trend is not None:
            trend_arr = np.array(trend)[:n]
            in_uptrend = trend_arr > 0
            signals.loc[in_uptrend & (signals["signal"] == -1), "signal"] = 0

        # Position sizing based on conviction and volatility
        signals["conviction"] = (signals["combined_proba"] - 0.5).abs() * 2
        if self.vol_scaling:
            target_vol = 0.15
            vol = pd.Series(garch_vol[:n]).fillna(pd.Series(hist_vol[:n])).fillna(0.02)
            vol_scalar = np.clip(target_vol / (vol * np.sqrt(252) + 1e-8), 0.2, 2.0)
            signals["position_size"] = signals["conviction"] * self.max_position_pct * vol_scalar
        else:
            signals["position_size"] = signals["conviction"] * self.max_position_pct

        signals["position_size"] = signals["position_size"].clip(0, self.max_position_pct)
        signals["stop_loss"] = self.stop_loss_pct
        signals["take_profit"] = self.take_profit_pct

        long_count = (signals["signal"] == 1).sum()
        short_count = (signals["signal"] == -1).sum()
        flat_count = (signals["signal"] == 0).sum()
        logger.info(f"Signals: {long_count} long, {short_count} short, {flat_count} flat")

        return signals
