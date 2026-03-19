"""
Kelly Criterion & Volatility Targeting for Dynamic Position Sizing.

Kelly fraction: f* = (p * b - q) / b
  p = win rate
  b = avg win / avg loss (odds)
  q = 1 - p (loss rate)

Half-Kelly is used in practice to reduce variance.

Usage:
    from analysis.kelly import KellySizer
    sizer = KellySizer(half_kelly=True)
    size = sizer.compute(win_rate=0.55, avg_win=0.04, avg_loss=0.02)
"""

import numpy as np
import pandas as pd


class KellySizer:
    """Dynamic position sizing using Kelly Criterion + Volatility Targeting."""

    def __init__(self, half_kelly: bool = True, vol_target: float = 0.15,
                 max_position: float = 0.40, min_position: float = 0.05):
        self.half_kelly   = half_kelly
        self.vol_target   = vol_target   # annualized vol target
        self.max_position = max_position
        self.min_position = min_position

    def kelly_fraction(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Compute Kelly fraction from win stats."""
        if avg_loss <= 0 or win_rate <= 0:
            return 0.0
        odds = avg_win / avg_loss
        q    = 1 - win_rate
        f    = (win_rate * odds - q) / odds
        if self.half_kelly:
            f *= 0.5
        return max(0.0, min(f, self.max_position))

    def vol_target_size(self, current_vol: float) -> float:
        """Scale position inversely with volatility (Volatility Targeting)."""
        if current_vol <= 0:
            return self.max_position
        size = self.vol_target / (current_vol * np.sqrt(252))
        return float(np.clip(size, self.min_position, self.max_position))

    def compute(self, win_rate: float, avg_win: float, avg_loss: float,
                current_vol: float = None) -> float:
        """Combine Kelly + Vol Targeting."""
        kelly = self.kelly_fraction(win_rate, avg_win, avg_loss)

        if current_vol is not None:
            vol_size = self.vol_target_size(current_vol)
            # Use geometric mean of both signals
            combined = np.sqrt(kelly * vol_size) if kelly > 0 else vol_size
        else:
            combined = kelly

        return float(np.clip(combined, self.min_position, self.max_position))

    def compute_from_trades(self, trade_pnl_pcts: np.ndarray,
                            current_vol: float = None) -> dict:
        """Compute position size from historical trade list."""
        if len(trade_pnl_pcts) == 0:
            return {"position_size": self.min_position, "kelly": 0, "vol_size": None}

        wins  = trade_pnl_pcts[trade_pnl_pcts > 0]
        losses = trade_pnl_pcts[trade_pnl_pcts < 0]

        win_rate = len(wins) / len(trade_pnl_pcts)
        avg_win  = wins.mean()  if len(wins) > 0  else 0.0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.001

        kelly    = self.kelly_fraction(win_rate, avg_win, avg_loss)
        vol_size = self.vol_target_size(current_vol) if current_vol else None

        size = self.compute(win_rate, avg_win, avg_loss, current_vol)

        return {
            "position_size": size,
            "kelly_fraction": kelly,
            "vol_size": vol_size,
            "win_rate": win_rate,
            "avg_win_pct": avg_win * 100,
            "avg_loss_pct": avg_loss * 100,
            "profit_factor": avg_win / avg_loss if avg_loss > 0 else float("inf"),
        }


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    sizer = KellySizer(half_kelly=True, vol_target=0.15)

    # Example from our best backtest results
    result = sizer.compute_from_trades(
        trade_pnl_pcts=np.array([0.04, 0.06, -0.02, 0.05, -0.02, 0.07, -0.02, 0.04] * 7),
        current_vol=0.012  # typical daily vol
    )

    print("\n" + "=" * 50)
    print("KELLY CRITERION POSITION SIZING")
    print("=" * 50)
    for k, v in result.items():
        if isinstance(v, float):
            print(f"  {k:25s}: {v:.4f}")
        else:
            print(f"  {k:25s}: {v}")
    print("=" * 50)
    print(f"\n  → Recommended position size: {result['position_size']*100:.1f}%")
