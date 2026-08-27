from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from .backtest import run_backtest, ExecutionModel

@dataclass(frozen=True)
class RobustnessConfig:
    bootstrap_runs: int = 500
    seed: int = 42
    fee_multipliers: tuple[float, ...] = (0.5, 1.0, 2.0)
    slippage_multipliers: tuple[float, ...] = (0.5, 1.0, 2.0)

def bootstrap_max_drawdown(returns: pd.Series, runs: int = 500, seed: int = 42) -> dict:
    r = returns.dropna().to_numpy(float)
    if len(r) < 2: return {"dd_p50": 0.0, "dd_p95": 0.0, "dd_p99": 0.0}
    rng = np.random.default_rng(seed); dds = np.empty(runs)
    for i in range(runs):
        sample = rng.choice(r, size=len(r), replace=True)
        eq = np.cumprod(1 + sample); dds[i] = np.min(eq / np.maximum.accumulate(eq) - 1)
    return {"dd_p50": float(np.quantile(dds,.50)), "dd_p95": float(np.quantile(dds,.95)), "dd_p99": float(np.quantile(dds,.99))}

def execution_stress(close: pd.Series, signal: pd.Series, base: ExecutionModel, cfg: RobustnessConfig = RobustnessConfig()) -> pd.DataFrame:
    rows=[]
    for fm in cfg.fee_multipliers:
        for sm in cfg.slippage_multipliers:
            ex=ExecutionModel(base.fee_rate*fm, base.slippage_bps*sm, base.spread_bps*sm, base.funding_rate_per_bar)
            r=run_backtest(close,signal,ex).metrics
            rows.append({"fee_mult":fm,"slippage_mult":sm,**r})
    return pd.DataFrame(rows)
