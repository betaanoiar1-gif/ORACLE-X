from __future__ import annotations

import pandas as pd


def classify_regime(close: pd.Series, lookback: int = 48, trend_threshold: float = 0.01) -> pd.Series:
    ret = close.pct_change(lookback)
    vol = close.pct_change().rolling(lookback).std()
    q = vol.rolling(lookback * 4, min_periods=lookback).median()
    out = pd.Series("unknown", index=close.index, dtype="object")
    out[(ret > trend_threshold) & (vol <= q)] = "trend_up_low_vol"
    out[(ret < -trend_threshold) & (vol <= q)] = "trend_down_low_vol"
    out[(ret > trend_threshold) & (vol > q)] = "trend_up_high_vol"
    out[(ret < -trend_threshold) & (vol > q)] = "trend_down_high_vol"
    out[(ret.abs() <= trend_threshold) & (vol <= q)] = "range_low_vol"
    out[(ret.abs() <= trend_threshold) & (vol > q)] = "range_high_vol"
    return out
