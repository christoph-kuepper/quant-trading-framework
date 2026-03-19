"""
Live Signal Dashboard — Quant Trading AI
Shows real-time model signals, refreshes every 60s.

Run: streamlit run live_dashboard.py
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

warnings.filterwarnings("ignore")
PROJECT_ROOT = os.path.dirname(__file__)
sys.path.insert(0, PROJECT_ROOT)

st.set_page_config(
    page_title="QQQ Live Signals",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Auto-refresh every 60s ────────────────────────────────────────────────────
REFRESH_SEC = 60
st_autorefresh = None
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=REFRESH_SEC * 1000, key="live_refresh")
except ImportError:
    pass  # manual refresh only


@st.cache_data(ttl=300)  # cache 5 min
def fetch_and_build():
    from data.fetcher import fetch_ohlcv, fetch_cross_assets
    from features.engineering import build_features, get_feature_columns
    raw   = fetch_ohlcv("QQQ", period="2y")
    try:
        cross = fetch_cross_assets(["^VIX"], period="2y")
    except Exception:
        cross = {}
    df = build_features(raw, cross).dropna()
    date_index = raw.index[len(raw) - len(df):]
    df.index = date_index[:len(df)]
    return df, get_feature_columns(df)


@st.cache_resource
def get_trained_model():
    from models.signal_model import SignalModel
    from models.regime import RegimeDetector
    from models.vol_forecast import VolatilityForecaster
    df, feature_cols = fetch_and_build()
    train_df = df.iloc[:-60]  # last 60 days = live test
    X = train_df[feature_cols].values
    y = (train_df["returns"].shift(-1) > 0).astype(int).values[:-1]
    model = SignalModel(n_estimators=200, max_depth=3, learning_rate=0.01)
    model.fit(X[:len(y)], y)
    regime = RegimeDetector(n_regimes=3)
    regime.fit(train_df)
    vol = VolatilityForecaster()
    return model, regime, vol, feature_cols


def get_live_signal(df, model, regime, vol, feature_cols):
    """Run model on last row of data."""
    train_df = df.iloc[:-1]
    row = df.iloc[-1]
    X = row[feature_cols].values.reshape(1, -1)
    xgb_prob = float(model.predict_proba(X)[0])
    regimes  = regime.predict(df.iloc[-5:])
    current_regime = int(regimes[-1]) if len(regimes) > 0 else 0

    # Feature importance for this signal
    raw_imp = model.model.get_booster().get_score(importance_type="gain")
    importance = {f: raw_imp.get(f"f{i}", 0.0) for i, f in enumerate(feature_cols)}
    total = sum(importance.values()) or 1
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:8]
    top_norm = [(f, v / total) for f, v in top_features]

    signal = "🟢 LONG" if xgb_prob > 0.50 else "🔴 SHORT" if xgb_prob < 0.50 else "⚪ FLAT"
    signal_val = 1 if xgb_prob > 0.50 else -1 if xgb_prob < 0.50 else 0

    return {
        "signal": signal,
        "signal_val": signal_val,
        "xgb_prob": xgb_prob,
        "conviction": abs(xgb_prob - 0.5) * 2,
        "regime": ["Trending", "Mean-Reverting", "Volatile"][current_regime % 3],
        "top_features": top_norm,
        "date": str(df.index[-1].date()),
        "close": float(row["close"]),
        "vix_z20": float(row.get("vix_z20", float("nan"))),
        "rsi_14": float(row.get("rsi_14", float("nan"))),
    }


# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("🤖 Quant Trading AI — Live Signal Dashboard")
st.caption(f"Model: XGBoost (200 trees) + VIX Cross-Asset | Asset: QQQ | Last update: {datetime.utcnow().strftime('%H:%M:%S UTC')}")

with st.spinner("Loading data & model..."):
    df, feature_cols = fetch_and_build()
    model, regime, vol, feature_cols = get_trained_model()

sig = get_live_signal(df, model, regime, vol, feature_cols)

# ── Current Signal ────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2, col3, col4, col5 = st.columns(5)

signal_color = "green" if sig["signal_val"] == 1 else "red" if sig["signal_val"] == -1 else "gray"
col1.markdown(f"### {sig['signal']}")
col1.caption(f"Date: {sig['date']}")
col2.metric("XGB Probability", f"{sig['xgb_prob']:.1%}")
col3.metric("Conviction", f"{sig['conviction']:.1%}")
col4.metric("QQQ Price", f"${sig['close']:.2f}")
col5.metric("Market Regime", sig["regime"])

st.markdown("---")

# ── Why this signal? ─────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("🧠 Why this signal?")
    vix_total = sum(v for f, v in sig["top_features"] if "vix" in f)
    rsi_val   = dict(sig["top_features"]).get("rsi_14", 0)

    st.info(f"""
**VIX Z-Score:** {sig['vix_z20']:.2f} {'(market calm → bullish)' if sig['vix_z20'] < 0 else '(market fearful → bearish)'}
**RSI-14:** {sig['rsi_14']:.1f} {'(neutral)' if 40 < sig['rsi_14'] < 60 else '(overbought)' if sig['rsi_14'] > 60 else '(oversold)'}

VIX features contribute **{vix_total:.1%}** of decision weight.
RSI contributes **{rsi_val:.1%}** of decision weight.

{"→ **VIX Z-Score is currently MORE important than RSI** ✅" if vix_total > rsi_val else "→ RSI is currently more important than VIX."}
    """)

    # Feature importance bar chart
    fig, ax = plt.subplots(figsize=(7, 4))
    feats  = [f for f, _ in sig["top_features"]][::-1]
    vals   = [v for _, v in sig["top_features"]][::-1]
    colors = ["#e74c3c" if "vix" in f else "#3498db" if "rsi" in f or "macd" in f
              else "#2ecc71" if "vol" in f else "#95a5a6" for f in feats]
    ax.barh(feats, vals, color=colors)
    ax.set_xlabel("Relative Importance")
    ax.set_title("Top Features (red=VIX, blue=momentum, green=vol)")
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col_b:
    st.subheader("📈 Recent Price + Signal History")
    recent = df.iloc[-60:].copy()
    X_recent = recent[feature_cols].values
    proba_recent = model.predict_proba(X_recent)
    signals_recent = np.where(proba_recent > 0.50, 1, np.where(proba_recent < 0.50, -1, 0))
    dates_recent = list(recent.index)

    # Interactive Plotly chart
    fig_live = go.Figure()
    fig_live.add_trace(go.Scatter(
        x=dates_recent, y=recent["close"].values,
        name="QQQ Price", line=dict(color="black", width=1.5),
        hovertemplate="%{x}<br>$%{y:.2f}<extra></extra>",
    ))
    long_idx  = np.where(signals_recent == 1)[0]
    short_idx = np.where(signals_recent == -1)[0]
    fig_live.add_trace(go.Scatter(
        x=[dates_recent[i] for i in long_idx],
        y=recent["close"].values[long_idx],
        mode="markers", name="Long Signal",
        marker=dict(color="green", size=8, symbol="triangle-up"),
    ))
    fig_live.add_trace(go.Scatter(
        x=[dates_recent[i] for i in short_idx],
        y=recent["close"].values[short_idx],
        mode="markers", name="Short Signal",
        marker=dict(color="red", size=8, symbol="triangle-down"),
    ))
    fig_live.update_layout(height=320, margin=dict(l=0,r=0,t=20,b=0),
                           hovermode="x unified", legend=dict(orientation="h"))
    st.plotly_chart(fig_live, use_container_width=True)

    # XGB probability bar
    fig_prob = go.Figure()
    fig_prob.add_trace(go.Bar(
        x=dates_recent, y=proba_recent,
        marker_color=["green" if p > 0.5 else "red" for p in proba_recent],
        name="XGB Probability",
        hovertemplate="%{x}<br>Prob: %{y:.1%}<extra></extra>",
    ))
    fig_prob.add_hline(y=0.5, line_dash="dash", line_color="gray")
    fig_prob.update_layout(height=160, margin=dict(l=0,r=0,t=10,b=0),
                           yaxis_title="Probability", showlegend=False)
    st.plotly_chart(fig_prob, use_container_width=True)

# ── Signal History Table ──────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📋 Last 10 Signals")
history = pd.DataFrame({
    "Date":    [str(d.date()) for d in df.index[-10:]],
    "Close":   [f"${p:.2f}" for p in df["close"].iloc[-10:].values],
    "XGB Prob": [f"{p:.1%}" for p in proba_recent[-10:]],
    "Signal":  ["🟢 LONG" if s == 1 else "🔴 SHORT" if s == -1 else "⚪ FLAT"
                for s in signals_recent[-10:]],
})
st.table(history)

st.caption("💡 Run `python3 live/alpaca_bridge.py` to place paper orders based on this signal.")
