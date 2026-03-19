import logging
import warnings
import numpy as np
import pandas as pd
from arch import arch_model

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


class VolatilityForecaster:
    def __init__(self, vol_model: str = "GARCH", p: int = 1, q: int = 1):
        self.vol_model = vol_model
        self.p = p
        self.q = q

    def fit_predict(self, train_returns: pd.Series, test_returns: pd.Series = None) -> tuple[pd.Series, pd.Series]:
        """Fit on train, predict for both train and test. Returns (train_vol, test_vol)."""
        all_returns = pd.concat([train_returns, test_returns]) if test_returns is not None else train_returns
        returns_pct = (all_returns * 100).dropna().replace([np.inf, -np.inf], np.nan).dropna()

        if len(returns_pct) < 200:
            logger.info("Using rolling vol (insufficient data for GARCH)")
            vol = all_returns.rolling(20).std().fillna(method="bfill")
            if test_returns is not None:
                return vol.iloc[:len(train_returns)], vol.iloc[len(train_returns):]
            return vol, pd.Series(dtype=float)

        try:
            model = arch_model(
                returns_pct,
                vol=self.vol_model,
                p=self.p,
                q=self.q,
                mean="Constant",
                rescale=False,
            )
            fitted = model.fit(disp="off", show_warning=False)
            cond_vol = fitted.conditional_volatility / 100

            result = pd.Series(np.nan, index=all_returns.index)
            for idx in cond_vol.index:
                if idx in result.index:
                    result.loc[idx] = cond_vol.loc[idx]
            result = result.ffill().bfill()

            logger.info(f"GARCH({self.p},{self.q}) fitted, avg vol: {result.mean():.4f}")

            if test_returns is not None:
                return result.iloc[:len(train_returns)], result.iloc[len(train_returns):]
            return result, pd.Series(dtype=float)

        except Exception as e:
            logger.warning(f"GARCH failed: {e}, using rolling vol")
            vol = all_returns.rolling(20).std().fillna(method="bfill")
            if test_returns is not None:
                return vol.iloc[:len(train_returns)], vol.iloc[len(train_returns):]
            return vol, pd.Series(dtype=float)
