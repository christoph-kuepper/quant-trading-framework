import logging
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)


class SignalModel:
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.01,
                 subsample=0.7, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=2.0):
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=5,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        )
        self.scaler = StandardScaler()
        self.feature_cols = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, feature_cols: list[str] = None):
        self.feature_cols = feature_cols
        X_scaled = self.scaler.fit_transform(X_train)

        # Time series cross-validation to check for overfitting
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = []
        for train_idx, val_idx in tscv.split(X_scaled):
            self.model.fit(X_scaled[train_idx], y_train[train_idx])
            val_pred = self.model.predict(X_scaled[val_idx])
            cv_scores.append(accuracy_score(y_train[val_idx], val_pred))

        # Final fit on all training data
        self.model.fit(X_scaled, y_train)
        train_pred = self.model.predict(X_scaled)
        train_acc = accuracy_score(y_train, train_pred)
        logger.info(f"XGBoost train acc: {train_acc:.4f}, CV scores: {[f'{s:.4f}' for s in cv_scores]}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self.scaler.transform(X))[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self.scaler.transform(X))

    def feature_importance(self) -> dict:
        if self.feature_cols is None:
            return {}
        return dict(sorted(zip(self.feature_cols, self.model.feature_importances_), key=lambda x: -x[1]))
