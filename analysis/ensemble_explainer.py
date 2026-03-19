"""
Ensemble Explainability — XGBoost + LSTM Transparency

Addresses the "LSTM gap" in SHAP:
- Shows XGB vs LSTM signal contribution separately
- Computes Agreement Score (when do they agree/disagree?)
- Approximates LSTM feature sensitivity via input perturbation
  (manual GradientExplainer equivalent, CPU-friendly)

Key insight: If XGB says SHORT but LSTM says LONG, the SHAP plot
of XGBoost alone is misleading. This module shows the full picture.

Usage:
    python3 analysis/ensemble_explainer.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)


def lstm_sensitivity(lstm_model, X: np.ndarray, feature_names: list,
                     epsilon: float = 0.01) -> np.ndarray:
    """
    Approximate LSTM feature importance via input perturbation.
    For each feature: how much does LSTM output change when we perturb it by epsilon?
    This is a CPU-friendly alternative to DeepExplainer/GradientExplainer.
    """
    if lstm_model is None or lstm_model.model is None:
        return np.zeros(len(feature_names))

    base_proba = lstm_model.predict_proba(X)
    sensitivities = np.zeros(len(feature_names))

    for i in range(len(feature_names)):
        X_perturbed = X.copy()
        X_perturbed[:, i] += epsilon
        perturbed_proba = lstm_model.predict_proba(X_perturbed)
        sensitivities[i] = np.abs(perturbed_proba - base_proba).mean() / epsilon

    # Normalize to sum=1
    total = sensitivities.sum()
    return sensitivities / total if total > 0 else sensitivities


def model_agreement(xgb_proba: np.ndarray, lstm_proba: np.ndarray,
                    threshold: float = 0.50) -> dict:
    """
    Compute agreement metrics between XGBoost and LSTM.
    When they disagree, the ensemble explanation is less reliable.
    """
    xgb_signal  = (xgb_proba > threshold).astype(int)
    lstm_signal = (lstm_proba > threshold).astype(int)

    agreement = (xgb_signal == lstm_signal).mean()
    both_long  = ((xgb_signal == 1) & (lstm_signal == 1)).mean()
    both_short = ((xgb_signal == 0) & (lstm_signal == 0)).mean()
    conflict   = (xgb_signal != lstm_signal).mean()

    return {
        "agreement_rate": agreement,
        "both_long": both_long,
        "both_short": both_short,
        "conflict_rate": conflict,
        "xgb_mean_proba": xgb_proba.mean(),
        "lstm_mean_proba": lstm_proba.mean(),
        "shap_reliable": agreement > 0.70,  # SHAP is reliable when models agree >70%
    }


def plot_ensemble_comparison(xgb_importance: dict, lstm_sensitivity: np.ndarray,
                             feature_names: list, agreement: dict,
                             output_path: str = None):
    """
    Side-by-side: XGBoost importance vs LSTM sensitivity.
    Highlights where they agree and disagree.
    """
    total_xgb = sum(xgb_importance.values()) or 1
    xgb_norm = np.array([xgb_importance.get(f"f{i}", 0) / total_xgb
                         for i in range(len(feature_names))])

    # Top 10 by combined score
    combined = 0.6 * xgb_norm + 0.4 * lstm_sensitivity
    top_idx = np.argsort(combined)[-10:][::-1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # XGBoost importance
    xgb_top = xgb_norm[top_idx]
    lstm_top = lstm_sensitivity[top_idx]
    feat_top = [feature_names[i] for i in top_idx]

    # Agreement color: green if both agree on importance, red if conflict
    colors_xgb  = ["#2ecc71" if abs(xgb_norm[i] - lstm_sensitivity[i]) < 0.03
                   else "#e74c3c" for i in top_idx]

    axes[0].barh(feat_top[::-1], xgb_top[::-1], color=colors_xgb[::-1])
    axes[0].set_title("XGBoost Feature Importance\n(green=agrees with LSTM, red=disagrees)")
    axes[0].set_xlabel("Relative Importance (Gain)")
    axes[0].grid(alpha=0.3, axis="x")

    axes[1].barh(feat_top[::-1], lstm_top[::-1], color=colors_xgb[::-1])
    axes[1].set_title("LSTM Feature Sensitivity\n(input perturbation approximation)")
    axes[1].set_xlabel("Sensitivity (Δoutput / Δinput)")
    axes[1].grid(alpha=0.3, axis="x")

    # Agreement summary
    color_agree = "green" if agreement["shap_reliable"] else "orange"
    fig.suptitle(
        f"Ensemble Explainability | Agreement: {agreement['agreement_rate']:.1%} "
        f"({'✅ SHAP reliable' if agreement['shap_reliable'] else '⚠️ Models disagree — SHAP partial'}) | "
        f"Conflict rate: {agreement['conflict_rate']:.1%}",
        fontsize=11, color=color_agree
    )
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=120)
        plt.close()
        print(f"Ensemble comparison saved: {output_path}")
    else:
        plt.show()


def run_ensemble_analysis():
    from data.fetcher import fetch_ohlcv, fetch_cross_assets
    from features.engineering import build_features, get_feature_columns
    from models.signal_model import SignalModel
    from models.lstm_model import LSTMPredictor

    print("Loading data & training models...")
    raw   = fetch_ohlcv("QQQ", "10y")
    cross = fetch_cross_assets(["^VIX"], "10y")
    df    = build_features(raw, cross).dropna()
    date_index = raw.index[len(raw) - len(df):]
    df.index = date_index[:len(df)]

    feature_cols = get_feature_columns(df)
    train_df = df[df.index <= "2022-12-31"]
    test_df  = df[df.index >= "2023-01-01"].iloc[:100]  # sample for speed

    X_train = train_df[feature_cols].values
    y_train = (train_df["returns"].shift(-1) > 0).astype(int).values[:-1]
    X_test  = test_df[feature_cols].values

    # Train XGBoost
    xgb = SignalModel(n_estimators=200, max_depth=3, learning_rate=0.01)
    xgb.fit(X_train[:len(y_train)], y_train)
    xgb_proba = xgb.predict_proba(X_test)

    # Train LSTM
    print("Training LSTM (this takes ~30s)...")
    lstm = LSTMPredictor(lookback=20, hidden_size=64, epochs=20)
    lstm.fit(X_train[:len(y_train)], y_train)
    lstm_proba = lstm.predict_proba(X_test)

    # Agreement analysis
    agree = model_agreement(xgb_proba, lstm_proba)

    print(f"\n{'='*60}")
    print("ENSEMBLE AGREEMENT ANALYSIS")
    print(f"{'='*60}")
    print(f"  Agreement rate:   {agree['agreement_rate']:.1%}")
    print(f"  Both LONG:        {agree['both_long']:.1%}")
    print(f"  Both SHORT:       {agree['both_short']:.1%}")
    print(f"  Conflict rate:    {agree['conflict_rate']:.1%}")
    print(f"  XGB mean proba:   {agree['xgb_mean_proba']:.3f}")
    print(f"  LSTM mean proba:  {agree['lstm_mean_proba']:.3f}")
    print(f"\n  SHAP reliability: {'✅ Reliable (>70% agreement)' if agree['shap_reliable'] else '⚠️  Partial — LSTM overrides XGB in some windows'}")
    print(f"{'='*60}")

    # LSTM sensitivity
    print("\nComputing LSTM feature sensitivity (perturbation method)...")
    lstm_sens = lstm_sensitivity(lstm, X_test, feature_cols)

    # XGB importance
    raw_imp  = xgb.model.get_booster().get_score(importance_type="gain")
    xgb_imp  = {f: raw_imp.get(f"f{i}", 0) for i, f in enumerate(feature_cols)}
    total    = sum(xgb_imp.values()) or 1

    print("\nTop features comparison:")
    print(f"{'Feature':28s} {'XGB':>8} {'LSTM':>8} {'Agree?':>8}")
    print("-" * 55)
    combined = {f: 0.6*(xgb_imp[f]/total) + 0.4*lstm_sens[i]
                for i, f in enumerate(feature_cols)}
    for f, _ in sorted(combined.items(), key=lambda x: x[1], reverse=True)[:10]:
        i = feature_cols.index(f)
        xv = xgb_imp[f] / total
        lv = lstm_sens[i]
        agree_str = "✅" if abs(xv - lv) < 0.03 else "⚠️"
        print(f"  {f:26s} {xv:8.3f} {lv:8.3f} {agree_str:>8}")

    # Plot
    os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)
    plot_ensemble_comparison(
        xgb_imp, lstm_sens, feature_cols, agree,
        output_path=os.path.join(PROJECT_ROOT, "results", "ensemble_explainer.png")
    )


if __name__ == "__main__":
    run_ensemble_analysis()
