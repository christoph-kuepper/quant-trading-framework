"""
Alpaca Paper-Trading Bridge
Pulls live QQQ data, runs the best model, places virtual orders.

Setup:
    export ALPACA_API_KEY=your_key
    export ALPACA_SECRET_KEY=your_secret
    python3 live/alpaca_bridge.py

Get free paper-trading keys at: https://alpaca.markets
"""

import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

ALPACA_API_KEY    = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL   = "https://paper-api.alpaca.markets"  # paper trading
SYMBOL            = "QQQ"


def get_alpaca_client():
    try:
        import alpaca_trade_api as tradeapi
        return tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)
    except ImportError:
        logger.error("alpaca-trade-api not installed. Run: pip install alpaca-trade-api")
        sys.exit(1)


def fetch_live_data(api, symbol: str, lookback_days: int = 600) -> pd.DataFrame:
    """Fetch recent OHLCV data from Alpaca."""
    end   = datetime.now()
    start = end - timedelta(days=lookback_days)
    bars  = api.get_bars(symbol, "1Day", start=start.isoformat(), end=end.isoformat()).df
    bars  = bars.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    bars.index = pd.to_datetime(bars.index).tz_localize(None)
    logger.info(f"Fetched {len(bars)} bars for {symbol}")
    return bars[["open", "high", "low", "close", "volume"]]


def get_current_signal(ohlcv_df: pd.DataFrame) -> dict:
    """Run the model on live data and return a trading signal."""
    from data.fetcher import fetch_cross_assets
    from features.engineering import build_features, get_feature_columns
    from models.signal_model import SignalModel
    from models.regime import RegimeDetector
    from models.vol_forecast import VolatilityForecaster
    from backtesting.decision import DecisionLayer

    # Fetch VIX
    try:
        cross = fetch_cross_assets(["^VIX"], period="2y")
    except Exception:
        cross = {}

    df = build_features(ohlcv_df, cross).dropna()
    date_index = ohlcv_df.index[len(ohlcv_df) - len(df):]
    df.index = date_index[:len(df)]

    feature_cols = get_feature_columns(df)
    train_df = df.iloc[:-1]  # everything except last row
    test_row  = df.iloc[-1:]

    X_train = train_df[feature_cols].values
    y_train = (train_df["returns"].shift(-1) > 0).astype(int).values[:-1]

    model = SignalModel(n_estimators=200, max_depth=3, learning_rate=0.01)
    model.fit(X_train[:-1], y_train)

    regime = RegimeDetector(n_regimes=3)
    regime.fit(train_df)

    vol = VolatilityForecaster()
    _, garch = vol.fit_predict(train_df["returns"].iloc[:len(train_df)//2], train_df["returns"])

    X_test   = test_row[feature_cols].values
    xgb_prob = model.predict_proba(X_test)[0]
    regimes  = regime.predict(test_row)
    garch_v  = garch.iloc[-1] if len(garch) > 0 else test_row["vol_20"].values[0]

    decision = DecisionLayer(signal_threshold=0.50, max_position_pct=0.35,
                             stop_loss_pct=0.02, take_profit_pct=0.07)
    sigs = decision.generate_signals(
        np.array([xgb_prob]), np.array([0.5]),
        np.array([regimes[0]] if len(regimes) > 0 else [0]),
        np.array([garch_v]), test_row["vol_20"].values,
    )

    return {
        "date":        str(test_row.index[-1].date()),
        "signal":      int(sigs["signal"].iloc[0]),
        "xgb_proba":   float(xgb_prob),
        "conviction":  float(sigs["conviction"].iloc[0]),
        "position_pct": float(sigs["position_size"].iloc[0]),
    }


def place_order(api, signal: dict, dry_run: bool = True):
    """Place a paper trade on Alpaca."""
    action = {1: "BUY", -1: "SELL", 0: "FLAT"}.get(signal["signal"], "FLAT")

    if action == "FLAT":
        logger.info("Signal: FLAT — no order placed")
        return

    # Get current price
    quote   = api.get_latest_trade(SYMBOL)
    price   = float(quote.price)
    account = api.get_account()
    equity  = float(account.equity)
    qty     = int((equity * signal["position_pct"]) / price)

    if qty <= 0:
        logger.warning("Quantity too small, skipping order")
        return

    logger.info(f"Signal: {action} {qty} shares of {SYMBOL} @ ~${price:.2f}")
    logger.info(f"  XGB proba: {signal['xgb_proba']:.3f} | Conviction: {signal['conviction']:.3f}")

    if dry_run:
        logger.info("  [DRY RUN] Order NOT submitted")
        return

    side = "buy" if action == "BUY" else "sell"
    order = api.submit_order(
        symbol=SYMBOL, qty=qty, side=side,
        type="market", time_in_force="day"
    )
    logger.info(f"  Order submitted: {order.id}")


def run_live(dry_run: bool = True):
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        logger.warning("No Alpaca API keys found. Set ALPACA_API_KEY and ALPACA_SECRET_KEY.")
        logger.info("Running in SIMULATION mode with cached yfinance data...")
        from data.fetcher import fetch_ohlcv
        ohlcv = fetch_ohlcv(SYMBOL, "2y")
    else:
        api   = get_alpaca_client()
        ohlcv = fetch_live_data(api, SYMBOL)

    logger.info("Running model on live data...")
    signal = get_current_signal(ohlcv)

    print("\n" + "=" * 50)
    print(f"LIVE SIGNAL — {signal['date']}")
    print("=" * 50)
    print(f"  Signal:       {['SHORT', 'FLAT', 'LONG'][signal['signal']+1]}")
    print(f"  XGB Proba:    {signal['xgb_proba']:.3f}")
    print(f"  Conviction:   {signal['conviction']:.3f}")
    print(f"  Position:     {signal['position_pct']*100:.1f}%")
    print("=" * 50)

    if ALPACA_API_KEY:
        place_order(api, signal, dry_run=dry_run)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Submit real paper orders (not dry run)")
    args = parser.parse_args()
    run_live(dry_run=not args.live)
