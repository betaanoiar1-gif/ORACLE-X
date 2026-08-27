from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class RegimeConfig:
    volatility_window: int = 48
    trend_window: int = 48
    quantiles: int = 3


def classify_regimes(close: pd.Series, config: RegimeConfig | None = None) -> pd.DataFrame:
    cfg = config or RegimeConfig()
    c = close.astype(float)
    ret = c.pct_change()
    vol = ret.rolling(cfg.volatility_window, min_periods=cfg.volatility_window).std()
    trend = c.pct_change(cfg.trend_window)
    vol_rank = vol.rank(pct=True)
    trend_rank = trend.abs().rank(pct=True)
    regime = pd.Series("unknown", index=c.index, dtype="object")
    regime[(trend > 0) & (vol_rank >= 2/3)] = "bull_high_vol"
    regime[(trend > 0) & (vol_rank < 2/3)] = "bull_low_vol"
    regime[(trend < 0) & (vol_rank >= 2/3)] = "bear_high_vol"
    regime[(trend < 0) & (vol_rank < 2/3)] = "bear_low_vol"
    regime[(trend.abs() <= trend.abs().rolling(cfg.trend_window, min_periods=1).median()) & (vol_rank < 1/3)] = "range_low_vol"
    regime[(trend.abs() <= trend.abs().rolling(cfg.trend_window, min_periods=1).median()) & (vol_rank >= 1/3)] = "range_high_vol"
    return pd.DataFrame({"return": ret, "volatility": vol, "trend": trend, "regime": regime})


def regime_statistics(returns: pd.Series, regimes: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"r": returns, "regime": regimes}).dropna()
    if df.empty:
        return pd.DataFrame()
    return df.groupby("regime")["r"].agg(["count", "mean", "std", "sum"])
