"""
Quant Trading AI — Streamlit Dashboard
Run: streamlit run dashboard.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go

warnings.filterwarnings("ignore")
PROJECT_ROOT = os.path.dirname(__file__)
sys.path.insert(0, PROJECT_ROOT)

st.set_page_config(page_title="Quant Trading AI", page_icon="📈", layout="wide")

@st.cache_data
def load_data():
    from data.fetcher import fetch_ohlcv, fetch_cross_assets
    from features.engineering import build_features, get_feature_columns
    raw = fetch_ohlcv("QQQ", period="10y")
    try:
        cross = fetch_cross_assets(["^VIX"], period="10y")
    except Exception:
        cross = {}
    df = build_features(raw, cross).dropna()
    date_index = raw.index[len(raw) - len(df):]
    df.index = date_index[:len(df)]
    return df, get_feature_columns(df)


@st.cache_resource
def train_model(train_end: str, n_estimators: int):
    from models.signal_model import SignalModel
    from models.regime import RegimeDetector
    from models.vol_forecast import VolatilityForecaster
    df, feature_cols = load_data()
    train_df = df[df.index <= train_end]
    X = train_df[feature_cols].values
    y = (train_df["returns"].shift(-1) > 0).astype(int).values[:-1]
    model = SignalModel(n_estimators=n_estimators, max_depth=3, learning_rate=0.01)
    model.fit(X[:-1], y)
    regime = RegimeDetector(n_regimes=3)
    regime.fit(train_df)
    vol = VolatilityForecaster()
    return model, regime, vol, feature_cols


def run_backtest(df, model, regime_model, vol_model, feature_cols,
                 threshold, stop, tp, pos_size, commission):
    from backtesting.decision import DecisionLayer
    from backtesting.engine import BacktestEngine
    X = df[feature_cols].values
    regimes = regime_model.predict(df)
    _, garch = vol_model.fit_predict(df["returns"].iloc[:len(df)//2], df["returns"])
    xgb_proba = model.predict_proba(X)
    lstm_proba = np.full(len(X), 0.5)
    decision = DecisionLayer(signal_threshold=threshold, max_position_pct=pos_size,
                             stop_loss_pct=stop, take_profit_pct=tp)
    signals = decision.generate_signals(xgb_proba, lstm_proba, regimes,
        garch.values if len(garch) > 0 else df["vol_20"].values, df["vol_20"].values)
    engine = BacktestEngine(initial_capital=10_000.0, commission=commission)
    result = engine.run(df["close"], signals)
    return result, engine, signals


# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
train_end   = st.sidebar.selectbox("Train cutoff", ["2020-12-31", "2021-12-31", "2022-12-31"], index=2)
test_start  = st.sidebar.selectbox("Test start",   ["2021-01-01", "2022-01-01", "2023-01-01"], index=2)
threshold   = st.sidebar.slider("Signal threshold", 0.48, 0.55, 0.50, 0.01)
pos_size    = st.sidebar.slider("Max position size", 0.10, 0.50, 0.35, 0.05)
stop        = st.sidebar.slider("Stop loss %", 0.01, 0.05, 0.02, 0.005)
tp          = st.sidebar.slider("Take profit %", 0.03, 0.12, 0.07, 0.005)
n_trees     = st.sidebar.select_slider("XGB trees", [50, 100, 150, 200, 250], value=200)
commission  = st.sidebar.number_input("Commission (bps)", value=5, min_value=0, max_value=50) / 10000
stress_test = st.sidebar.button("🔥 Stress Test (2020 Crash)")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("📈 Quant Trading AI — QQQ + VIX")
st.caption("Walk-forward backtesting with XGBoost, HMM regime detection & GARCH volatility")

with st.spinner("Loading data & training model..."):
    df, feature_cols = load_data()
    model, regime_model, vol_model, feature_cols = train_model(train_end, n_trees)

test_df = df[df.index >= test_start] if not stress_test else df[(df.index >= "2020-01-01") & (df.index <= "2020-12-31")]
label   = "🧨 Stress Test: 2020 Crash" if stress_test else f"Test period: {test_start} – present"

with st.spinner(f"Running backtest ({label})..."):
    result, engine, signals = run_backtest(test_df, model, regime_model, vol_model,
                                           feature_cols, threshold, stop, tp, pos_size, commission)

# ── Metrics ──────────────────────────────────────────────────────────────────
st.subheader(f"📊 Results — {label}")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Sharpe Ratio",    f"{result['sharpe']:.3f}")
col2.metric("Total Return",    f"{result['total_return']*100:.2f}%")
col3.metric("Max Drawdown",    f"{result['max_drawdown']*100:.2f}%")
col4.metric("Profit Factor",   f"{result['profit_factor']:.2f}")
col5.metric("Trades",          result['n_trades'])

# ── Interactive Equity Curve vs Buy & Hold ────────────────────────────────────
st.subheader("📉 Equity Curve vs Buy & Hold (Interactive)")

eq_data = engine._equity if hasattr(engine, "_equity") else None
bh_values = (test_df["close"] / test_df["close"].iloc[0]) * 10_000

dates = list(test_df.index) if hasattr(test_df.index, '__len__') else list(range(len(bh_values)))

fig_plotly = go.Figure()

# Buy & Hold
fig_plotly.add_trace(go.Scatter(
    x=dates, y=bh_values.values,
    name="Buy & Hold QQQ",
    line=dict(color="gray", width=2, dash="dash"),
    hovertemplate="%{x}<br>B&H: $%{y:.2f}<extra></extra>",
))

# Strategy
if eq_data is not None:
    eq_series = pd.Series(eq_data[:len(dates)])
    strategy_color = "steelblue"
    final_vs_bh = eq_series.iloc[-1] - bh_values.iloc[-1]
    label = f"Strategy ({'▲' if final_vs_bh >= 0 else '▼'} ${abs(final_vs_bh):.0f} vs B&H)"
    fig_plotly.add_trace(go.Scatter(
        x=dates[:len(eq_series)], y=eq_series.values,
        name=label,
        line=dict(color=strategy_color, width=2.5),
        hovertemplate="%{x}<br>Strategy: $%{y:.2f}<extra></extra>",
    ))

fig_plotly.update_layout(
    xaxis_title="Date",
    yaxis_title="Portfolio Value ($)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    height=400,
    margin=dict(l=0, r=0, t=30, b=0),
)
fig_plotly.add_hline(y=10_000, line_dash="dot", line_color="black", opacity=0.3,
                     annotation_text="Initial $10,000")

st.plotly_chart(fig_plotly, use_container_width=True)

# ── SHAP Feature Importance ───────────────────────────────────────────────────
st.subheader("🧠 Feature Importance (SHAP)")
try:
    import shap
    X_sample = test_df[feature_cols].values[:200]
    explainer = shap.TreeExplainer(model.model)
    shap_values = explainer.shap_values(X_sample)
    vals = np.abs(shap_values).mean(axis=0) if isinstance(shap_values, np.ndarray) else np.abs(shap_values[1]).mean(axis=0)
    importance_df = pd.DataFrame({"Feature": feature_cols, "SHAP Importance": vals})
    importance_df = importance_df.sort_values("SHAP Importance", ascending=True).tail(15)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.barh(importance_df["Feature"], importance_df["SHAP Importance"], color="steelblue")
    ax2.set_xlabel("Mean |SHAP Value|")
    ax2.set_title("Top 15 Features by SHAP Importance")
    ax2.grid(alpha=0.3, axis="x")
    st.pyplot(fig2)
    plt.close()
except Exception as e:
    st.warning(f"SHAP not available: {e}")

# ── Signal Distribution ───────────────────────────────────────────────────────
st.subheader("📌 Signal Distribution")
sig_counts = signals["signal"].value_counts().rename({1: "Long", -1: "Short", 0: "Flat"})
st.bar_chart(sig_counts)

# ── Trade Table ────────────────────────────────────────────────────────────────
st.subheader("📋 Raw Settings")
st.json({
    "symbol": "QQQ",
    "train_cutoff": train_end,
    "test_start": test_start,
    "threshold": threshold,
    "position_size": f"{pos_size*100:.0f}%",
    "stop_loss": f"{stop*100:.1f}%",
    "take_profit": f"{tp*100:.1f}%",
    "xgb_trees": n_trees,
    "commission_bps": commission * 10000,
})
