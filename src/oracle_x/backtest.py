from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class ExecutionModel:
    fee_rate: float = 0.0004
    slippage_bps: float = 2.0
    spread_bps: float = 1.0
    funding_rate_per_bar: float = 0.0

@dataclass
class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    metrics: dict


def run_backtest(close: pd.Series, signal: pd.Series, execution: ExecutionModel = ExecutionModel(), initial_capital: float = 100.0) -> BacktestResult:
    close = close.astype(float)
    signal = signal.reindex(close.index).fillna(0).clip(-1, 1)
    pos = signal.shift(1).fillna(0.0)  # next-bar execution: no same-bar lookahead
    gross = pos * close.pct_change().fillna(0)
    turnover = pos.diff().abs().fillna(pos.abs())
    cost = turnover * (execution.fee_rate + execution.spread_bps / 10000 + execution.slippage_bps / 10000)
    funding = pos.abs() * execution.funding_rate_per_bar
    net = gross - cost - funding
    equity = initial_capital * (1 + net).cumprod()
    dd = equity / equity.cummax() - 1
    trades = pd.DataFrame({"close": close, "signal": signal, "position": pos, "net_return": net, "equity": equity}, index=close.index)
    n = int((turnover > 0).sum())
    metrics = {
        "initial_capital": initial_capital,
        "final_equity": float(equity.iloc[-1]),
        "total_return": float(equity.iloc[-1] / initial_capital - 1),
        "max_drawdown": float(dd.min()),
        "trade_events": n,
        "bars": len(close),
        "sharpe_bar": float(np.sqrt(252) * net.mean() / net.std()) if net.std() else 0.0,
    }
    return BacktestResult(equity, trades, metrics)
