import logging
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class RegimeDetector:
    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.model = None
        self.scaler = StandardScaler()

    def fit(self, df: pd.DataFrame) -> "RegimeDetector":
        features = df[["returns", "vol_20"]].dropna().values
        features = self.scaler.fit_transform(features)

        self.model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=200,
            random_state=42,
        )
        self.model.fit(features)
        logger.info(f"HMM fitted: {self.n_regimes} regimes, score={self.model.score(features):.1f}")
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        features = df[["returns", "vol_20"]].dropna().values
        features = self.scaler.transform(features)
        return self.model.predict(features)

    def get_regime_stats(self, df: pd.DataFrame, regimes: np.ndarray) -> dict:
        stats = {}
        for r in range(self.n_regimes):
            mask = regimes == r
            stats[r] = {
                "mean_return": df.loc[mask, "returns"].mean(),
                "mean_vol": df.loc[mask, "vol_20"].mean(),
                "count": int(mask.sum()),
            }
        return stats
