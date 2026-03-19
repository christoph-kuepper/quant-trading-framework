"""Unit tests for backtesting engine and decision layer."""
import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backtesting.engine import BacktestEngine
from backtesting.decision import DecisionLayer


@pytest.fixture
def sample_prices():
    np.random.seed(42)
    n = 100
    price = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.Series(price)


@pytest.fixture
def all_long_signals(sample_prices):
    n = len(sample_prices)
    return pd.DataFrame({
        "signal": [1] * n,
        "position_size": [0.2] * n,
        "stop_loss": [0.02] * n,
        "take_profit": [0.06] * n,
    })


@pytest.fixture
def all_flat_signals(sample_prices):
    n = len(sample_prices)
    return pd.DataFrame({
        "signal": [0] * n,
        "position_size": [0.0] * n,
        "stop_loss": [0.02] * n,
        "take_profit": [0.06] * n,
    })


def test_flat_signals_no_change(sample_prices, all_flat_signals):
    engine = BacktestEngine(initial_capital=10_000)
    result = engine.run(sample_prices, all_flat_signals)
    assert result["n_trades"] == 0
    assert result["final_capital"] == pytest.approx(10_000.0, rel=1e-3)


def test_capital_always_positive(sample_prices, all_long_signals):
    engine = BacktestEngine(initial_capital=10_000)
    result = engine.run(sample_prices, all_long_signals)
    assert result["final_capital"] > 0


def test_max_drawdown_negative(sample_prices, all_long_signals):
    engine = BacktestEngine(initial_capital=10_000)
    result = engine.run(sample_prices, all_long_signals)
    assert result["max_drawdown"] <= 0, "Drawdown should be <= 0"


def test_win_rate_range(sample_prices, all_long_signals):
    engine = BacktestEngine(initial_capital=10_000)
    result = engine.run(sample_prices, all_long_signals)
    assert 0.0 <= result["win_rate"] <= 1.0


def test_decision_layer_threshold():
    n = 50
    xgb_proba = np.array([0.8] * n)
    lstm_proba = np.array([0.5] * n)
    regimes = np.zeros(n)
    vol = np.full(n, 0.01)
    layer = DecisionLayer(signal_threshold=0.55)
    signals = layer.generate_signals(xgb_proba, lstm_proba, regimes, vol, vol)
    assert (signals["signal"] == 1).all(), "High proba should give all long signals"


def test_decision_layer_short():
    n = 50
    xgb_proba = np.array([0.2] * n)
    lstm_proba = np.array([0.5] * n)
    regimes = np.zeros(n)
    vol = np.full(n, 0.01)
    layer = DecisionLayer(signal_threshold=0.55)
    signals = layer.generate_signals(xgb_proba, lstm_proba, regimes, vol, vol)
    assert (signals["signal"] == -1).all(), "Low proba should give all short signals"


def test_position_size_clipped():
    n = 20
    xgb_proba = np.array([0.9] * n)
    lstm_proba = np.array([0.5] * n)
    regimes = np.zeros(n)
    vol = np.full(n, 0.001)  # very low vol → high scalar
    layer = DecisionLayer(signal_threshold=0.50, max_position_pct=0.20)
    signals = layer.generate_signals(xgb_proba, lstm_proba, regimes, vol, vol)
    assert (signals["position_size"] <= 0.20 + 1e-9).all(), "Position must not exceed max"
