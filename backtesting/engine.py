import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class BacktestEngine:
    def __init__(self, initial_capital: float = 10_000.0, commission: float = 0.001,
                 slippage: float = 0.0001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage  # e.g. 0.0001 = 1bp worse execution than close

    def run(self, prices: pd.Series, signals: pd.DataFrame) -> dict:
        n = min(len(prices), len(signals))
        prices = prices.iloc[:n].values
        sig = signals.iloc[:n]

        capital = self.initial_capital
        position = 0.0
        entry_price = 0.0
        equity_curve = []
        trades = []

        for i in range(1, n):
            price = prices[i]
            prev_price = prices[i - 1]
            signal = sig.iloc[i]["signal"]
            pos_size = sig.iloc[i]["position_size"]
            stop_loss = sig.iloc[i]["stop_loss"]
            take_profit = sig.iloc[i]["take_profit"]

            # Check stop loss / take profit on existing position
            if position != 0 and entry_price > 0:
                pnl_pct = (price - entry_price) / entry_price * np.sign(position)
                if pnl_pct <= -stop_loss or pnl_pct >= take_profit:
                    # Slippage on exit: sell slightly below (long) or buy slightly above (short)
                    exit_price = price * (1 - self.slippage) if position > 0 else price * (1 + self.slippage)
                    trade_pnl = position * (exit_price - entry_price)
                    commission = abs(position * exit_price) * self.commission
                    capital += trade_pnl - commission
                    trades.append({
                        "exit_idx": i,
                        "pnl": trade_pnl - commission,
                        "pnl_pct": pnl_pct,
                        "reason": "stop_loss" if pnl_pct <= -stop_loss else "take_profit",
                    })
                    position = 0.0
                    entry_price = 0.0

            # New signal
            if signal != 0 and position == 0:
                # Slippage: buy slightly above close, sell slightly below
                exec_price = price * (1 + self.slippage) if signal == 1 else price * (1 - self.slippage)
                shares = (capital * pos_size) / exec_price
                if signal == -1:
                    shares = -shares
                commission = abs(shares * exec_price) * self.commission
                position = shares
                entry_price = exec_price
                capital -= commission
            elif signal == 0 and position != 0:
                trade_pnl = position * (price - entry_price)
                commission = abs(position * price) * self.commission
                capital += trade_pnl - commission
                trades.append({
                    "exit_idx": i,
                    "pnl": trade_pnl - commission,
                    "pnl_pct": (price - entry_price) / entry_price * np.sign(position),
                    "reason": "signal_exit",
                })
                position = 0.0
                entry_price = 0.0

            # Mark to market
            mtm = capital + position * (price - entry_price) if position != 0 else capital
            equity_curve.append(mtm)

        # Close any remaining position
        if position != 0:
            final_price = prices[-1]
            trade_pnl = position * (final_price - entry_price)
            capital += trade_pnl
            trades.append({
                "exit_idx": n - 1,
                "pnl": trade_pnl,
                "pnl_pct": (final_price - entry_price) / entry_price * np.sign(position),
                "reason": "end_of_backtest",
            })

        equity = pd.Series(equity_curve)
        metrics = self._compute_metrics(equity, trades)
        metrics["equity_curve"] = equity
        metrics["trades"] = trades
        return metrics

    def _compute_metrics(self, equity: pd.Series, trades: list) -> dict:
        if len(equity) < 2:
            return {"sharpe": 0, "max_drawdown": 0, "win_rate": 0, "profit_factor": 0, "total_return": 0, "n_trades": 0}

        returns = equity.pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        max_dd = drawdown.min()

        trade_pnls = [t["pnl"] for t in trades]
        wins = [p for p in trade_pnls if p > 0]
        losses = [p for p in trade_pnls if p <= 0]
        win_rate = len(wins) / len(trade_pnls) if trade_pnls else 0
        profit_factor = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")

        total_return = (equity.iloc[-1] / equity.iloc[0] - 1) if equity.iloc[0] > 0 else 0

        return {
            "sharpe": round(sharpe, 3),
            "max_drawdown": round(max_dd, 4),
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 3),
            "total_return": round(total_return, 4),
            "n_trades": len(trades),
            "total_pnl": round(sum(trade_pnls), 2),
            "final_capital": round(equity.iloc[-1], 2) if len(equity) > 0 else 0,
        }

    def plot_results(self, metrics: dict, output_path: str):
        equity = metrics["equity_curve"]
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        axes[0].plot(equity.values, linewidth=1)
        axes[0].axhline(y=self.initial_capital, color="gray", linestyle="--", alpha=0.5)
        axes[0].set_title(f"Equity Curve (Sharpe: {metrics['sharpe']:.2f}, Return: {metrics['total_return']:.1%})")
        axes[0].set_ylabel("Portfolio Value ($)")
        axes[0].grid(True, alpha=0.3)

        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        axes[1].fill_between(range(len(drawdown)), drawdown.values, 0, alpha=0.4, color="red")
        axes[1].set_title(f"Drawdown (Max: {metrics['max_drawdown']:.1%})")
        axes[1].set_ylabel("Drawdown")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        logger.info(f"Saved plot: {output_path}")
