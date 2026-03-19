import os
import logging
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


def fetch_ohlcv(symbol: str = "SPY", period: str = "10y", interval: str = "1d") -> pd.DataFrame:
    cache_path = os.path.join(CACHE_DIR, f"{symbol.replace('^', '')}_{interval}.csv")

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        logger.info(f"Loaded cached {symbol}: {len(df)} rows")
        return df

    logger.info(f"Downloading {symbol} {period} {interval}...")
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True)
    df = df.droplevel(1, axis=1) if isinstance(df.columns, pd.MultiIndex) else df
    df.columns = [c.lower() for c in df.columns]
    df.index.name = "date"

    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_csv(cache_path)
    logger.info(f"Cached {len(df)} rows for {symbol}")
    return df


def fetch_cross_assets(symbols: list[str] = None, period: str = "10y") -> dict[str, pd.DataFrame]:
    symbols = symbols or ["^VIX"]
    result = {}
    for sym in symbols:
        try:
            result[sym] = fetch_ohlcv(sym, period=period)
        except Exception as e:
            logger.warning(f"Failed to fetch {sym}: {e}")
    return result
