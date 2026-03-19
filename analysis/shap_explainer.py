"""
SHAP Explainability Module

Answers: "Why did the bot buy/sell on a specific date?"
- Global feature importance (bar chart)
- Feature importance over time (heatmap)
- Single-decision explanation (waterfall/force plot)

Usage:
    python3 analysis/shap_explainer.py --date 2024-03-14
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)


def build_explainer(model, X_background: np.ndarray):
    """Build SHAP Explainer from trained XGBoost model."""
    import shap
    # Use generic Explainer for version compatibility
    predict_fn = lambda x: model.model.predict_proba(x)[:, 1]
    explainer = shap.Explainer(predict_fn, X_background[:100])
    return explainer


def global_importance(explainer, X: np.ndarray, feature_names: list,
                      output_path: str = None) -> pd.DataFrame:
    """Mean |SHAP| across all samples — global feature importance."""
    import shap
    sv = explainer(X)
    shap_values = sv.values if hasattr(sv, "values") else sv

    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_shap", ascending=False)

    if output_path:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["steelblue" if "vix" in f else "seagreen" if "rsi" in f or "macd" in f
                  else "tomato" if "vol" in f else "gray" for f in importance["feature"]]
        ax.barh(importance["feature"][::-1], importance["mean_shap"][::-1], color=colors[::-1])
        ax.set_xlabel("Mean |SHAP Value|")
        ax.set_title("Global Feature Importance (SHAP)")
        ax.grid(alpha=0.3, axis="x")
        plt.tight_layout()
        plt.savefig(output_path, dpi=120)
        plt.close()
        print(f"Global importance plot saved: {output_path}")

    return importance


def importance_over_time(explainer, df: pd.DataFrame, feature_names: list,
                         output_path: str = None) -> pd.DataFrame:
    """Show how feature importance shifts over time (quarterly)."""
    import shap
    X = df[feature_names].values
    sv_obj = explainer(X)
    shap_values = sv_obj.values if hasattr(sv_obj, "values") else np.array(sv_obj)

    shap_df = pd.DataFrame(np.abs(shap_values), index=df.index, columns=feature_names)

    # Top 8 features only
    top_features = shap_df.mean().nlargest(8).index.tolist()
    quarterly = shap_df[top_features].resample("Q").mean()

    if output_path:
        fig, ax = plt.subplots(figsize=(14, 5))
        quarterly.plot(ax=ax, linewidth=2)
        ax.set_title("Feature Importance Over Time (Quarterly SHAP)")
        ax.set_ylabel("Mean |SHAP Value|")
        ax.legend(loc="upper left", fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=120)
        plt.close()
        print(f"Importance-over-time plot saved: {output_path}")

    return quarterly


def explain_decision(explainer, row: pd.Series, feature_names: list,
                     output_path: str = None) -> dict:
    """Explain a single trade decision on a specific date."""
    import shap
    X = row[feature_names].values.reshape(1, -1)
    sv_obj = explainer(X)
    sv = sv_obj.values[0] if hasattr(sv_obj, "values") else np.array(sv_obj)[0]
    base_value = float(sv_obj.base_values[0]) if hasattr(sv_obj, "base_values") else 0.5

    explanation = pd.DataFrame({
        "feature": feature_names,
        "value": X[0],
        "shap": sv,
    }).sort_values("shap", key=abs, ascending=False)

    top_positive = explanation[explanation["shap"] > 0].head(3)
    top_negative = explanation[explanation["shap"] < 0].head(3)

    if output_path:
        fig, ax = plt.subplots(figsize=(10, 5))
        top_n = explanation.head(12)
        colors = ["steelblue" if v > 0 else "tomato" for v in top_n["shap"]]
        ax.barh(top_n["feature"][::-1], top_n["shap"][::-1], color=colors[::-1])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP Value (impact on prediction)")
        ax.set_title(f"Decision Explanation — {row.name.date() if hasattr(row.name, 'date') else row.name}")
        ax.grid(alpha=0.3, axis="x")
        plt.tight_layout()
        plt.savefig(output_path, dpi=120)
        plt.close()
        print(f"Decision explanation plot saved: {output_path}")

    return {
        "date": str(row.name),
        "base_value": float(base_value),
        "prediction": float(base_value + sv.sum()),
        "top_bullish": top_positive[["feature", "value", "shap"]].to_dict("records"),
        "top_bearish": top_negative[["feature", "value", "shap"]].to_dict("records"),
    }


def run_full_analysis(explain_date: str = None):
    from data.fetcher import fetch_ohlcv, fetch_cross_assets
    from features.engineering import build_features, get_feature_columns
    from models.signal_model import SignalModel

    print("Loading data...")
    raw   = fetch_ohlcv("QQQ", "10y")
    cross = fetch_cross_assets(["^VIX"], "10y") if True else {}
    df    = build_features(raw, cross).dropna()
    date_index = raw.index[len(raw) - len(df):]
    df.index   = date_index[:len(df)]

    feature_cols = get_feature_columns(df)
    train_df = df[df.index <= "2022-12-31"]
    test_df  = df[df.index >= "2023-01-01"]

    X_train = train_df[feature_cols].values
    y_train = (train_df["returns"].shift(-1) > 0).astype(int).values[:-1]
    model = SignalModel(n_estimators=200, max_depth=3, learning_rate=0.01)
    model.fit(X_train[:len(y_train)], y_train)

    print("Building SHAP explainer...")
    explainer = build_explainer(model, X_train)

    os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)

    # 1. Global importance (sample 100 rows for speed)
    print("Computing global importance...")
    sample = test_df[feature_cols].sample(min(100, len(test_df)), random_state=42)
    global_importance(explainer, sample.values, feature_cols,
                      output_path=os.path.join(PROJECT_ROOT, "results", "shap_global.png"))

    # 2. Importance over time (sample monthly)
    print("Computing importance over time...")
    monthly_sample = test_df[feature_cols].resample("M").last().dropna()
    importance_over_time(explainer, monthly_sample, feature_cols,
                         output_path=os.path.join(PROJECT_ROOT, "results", "shap_over_time.png"))

    # 3. Single decision explanation
    date = explain_date or "2024-03-14"
    idx = test_df.index.searchsorted(pd.Timestamp(date))
    idx = min(idx, len(test_df) - 1)
    if idx >= 0:
        row = test_df.iloc[idx]
        print(f"\nExplaining decision on {date}...")
        result = explain_decision(
            explainer, row, feature_cols,
            output_path=os.path.join(PROJECT_ROOT, "results", f"shap_decision_{date}.png")
        )
        print(f"\n{'='*55}")
        print(f"DECISION EXPLANATION — {result['date']}")
        print(f"{'='*55}")
        print(f"  Model confidence (long): {result['prediction']*100:.1f}%")
        print(f"\n  Top BULLISH factors (pushing toward BUY):")
        for r in result["top_bullish"]:
            print(f"    {r['feature']:25s}: SHAP={r['shap']:+.4f}  (value={r['value']:.4f})")
        print(f"\n  Top BEARISH factors (pushing toward SELL):")
        for r in result["top_bearish"]:
            print(f"    {r['feature']:25s}: SHAP={r['shap']:+.4f}  (value={r['value']:.4f})")
        print(f"{'='*55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2024-03-14", help="Date to explain (YYYY-MM-DD)")
    args = parser.parse_args()
    run_full_analysis(explain_date=args.date)
