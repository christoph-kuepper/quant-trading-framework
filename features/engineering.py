import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

ROLLING_WINDOW = 10


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    for w in [10, 20, 50, 200]:
        df[f"sma_{w}"] = df["close"].rolling(w).mean()
    df["ema_12"] = df["close"].ewm(span=12).mean()
    df["ema_26"] = df["close"].ewm(span=26).mean()
    return df


def add_volatility(df: pd.DataFrame) -> pd.DataFrame:
    df["returns"] = df["close"].pct_change()
    df["vol_20"] = df["returns"].rolling(20).std()
    df["vol_60"] = df["returns"].rolling(60).std()

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()

    sma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    df["bb_width"] = (2 * std20) / sma20
    df["bb_pos"] = (df["close"] - (sma20 - 2 * std20)) / (4 * std20)

    return df


def add_momentum(df: pd.DataFrame) -> pd.DataFrame:
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    df["roc_10"] = df["close"].pct_change(10)
    df["roc_20"] = df["close"].pct_change(20)

    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).cumsum()
    df["volume_sma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma_20"]
    return df


def add_cross_asset_features(df: pd.DataFrame, cross_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    for sym, cdf in cross_data.items():
        prefix = sym.replace("^", "").lower()
        if "close" in cdf.columns:
            cross_close = cdf["close"].reindex(df.index, method="ffill")
            df[f"{prefix}_level"] = cross_close
            df[f"{prefix}_change"] = cross_close.pct_change()
            df[f"{prefix}_sma10"] = cross_close.rolling(10).mean()
            df[f"{prefix}_vs_sma"] = cross_close / df[f"{prefix}_sma10"] - 1
            df[f"{prefix}_z20"] = (cross_close - cross_close.rolling(20).mean()) / cross_close.rolling(20).std()
            logger.info(f"Added cross-asset features for {sym}")
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df["target_1d"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df["fwd_return_1d"] = df["close"].shift(-1) / df["close"] - 1
    return df


def build_features(df: pd.DataFrame, cross_data: dict = None) -> pd.DataFrame:
    logger.info("Building features...")
    df = df.copy()
    df = add_moving_averages(df)
    df = add_volatility(df)
    df = add_momentum(df)
    df = add_volume_features(df)

    if cross_data:
        df = add_cross_asset_features(df, cross_data)

    df = add_targets(df)

    for w in [10, 20, 50, 200]:
        df[f"price_vs_sma_{w}"] = df["close"] / df[f"sma_{w}"] - 1

    df["vol_ratio"] = df["vol_20"] / df["vol_60"].replace(0, np.nan)

    # Interaction features
    df["rsi_x_vol"] = df["rsi_14"] * df["vol_20"]
    df["momentum_strength"] = df["roc_10"] * df["volume_ratio"]
    df["trend_vol"] = df["price_vs_sma_50"] * df["vol_20"]
    df["mean_reversion"] = df["bb_pos"] * df["rsi_14"] / 100

    feature_cols = get_feature_columns(df)
    before = len(df)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    logger.info(f"Features built: {len(feature_cols)} features, {before - len(df)} rows dropped, {len(df)} remaining")
    return df


def get_feature_columns(df: pd.DataFrame = None) -> list[str]:
    base = [
        "vol_20", "vol_60", "atr_14", "bb_width", "bb_pos",
        "rsi_14", "macd_hist", "roc_10", "roc_20",
        "volume_ratio", "vol_ratio",
        "price_vs_sma_10", "price_vs_sma_20", "price_vs_sma_50", "price_vs_sma_200",
        "returns",
        "rsi_x_vol", "momentum_strength", "trend_vol", "mean_reversion",
    ]
    cross = []
    if df is not None:
        for col in df.columns:
            if col.startswith("vix_") and col != "vix_level":
                cross.append(col)
    return base + cross
