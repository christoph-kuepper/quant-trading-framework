"""Unit tests for feature engineering."""
import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from features.engineering import build_features, get_feature_columns


@pytest.fixture
def sample_ohlcv():
    n = 300
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    price = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": price * 0.999, "high": price * 1.01,
        "low": price * 0.99, "close": price,
        "volume": np.random.randint(1_000_000, 5_000_000, n).astype(float),
    }, index=dates)


def test_rsi_range(sample_ohlcv):
    df = build_features(sample_ohlcv).dropna()
    assert df["rsi_14"].between(0, 100).all(), "RSI must be between 0 and 100"


def test_bb_pos_range(sample_ohlcv):
    df = build_features(sample_ohlcv).dropna()
    assert df["bb_pos"].between(-0.5, 1.5).all(), "Bollinger Band position should be approximately 0-1"


def test_vol_positive(sample_ohlcv):
    df = build_features(sample_ohlcv).dropna()
    assert (df["vol_20"] > 0).all(), "Volatility must be positive"
    assert (df["vol_60"] > 0).all()


def test_no_inf(sample_ohlcv):
    df = build_features(sample_ohlcv).dropna()
    feat_cols = get_feature_columns(df)
    assert not np.isinf(df[feat_cols].values).any(), "No infinite values allowed"


def test_feature_count(sample_ohlcv):
    df = build_features(sample_ohlcv).dropna()
    feat_cols = get_feature_columns(df)
    assert len(feat_cols) >= 16, f"Expected >=16 features, got {len(feat_cols)}"


def test_returns_column(sample_ohlcv):
    df = build_features(sample_ohlcv).dropna()
    assert "returns" in df.columns
    assert df["returns"].abs().max() < 0.5, "Returns should be reasonable (<50%)"


def test_sma_features_relative(sample_ohlcv):
    df = build_features(sample_ohlcv).dropna()
    # price_vs_sma should be relative (not raw price levels)
    assert df["price_vs_sma_20"].abs().max() < 1.0, "SMA feature should be relative ratio"
