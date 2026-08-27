from __future__ import annotations

import numpy as np
import pandas as pd


def forward_return(close: pd.Series, horizon: int = 12) -> pd.Series:
    """Future return used only as a target; never feed this into causal features."""
    close = pd.to_numeric(close, errors="coerce")
    return close.shift(-horizon) / close - 1.0


def classify_return(close: pd.Series, horizon: int = 12, threshold: float = 0.0) -> pd.Series:
    r = forward_return(close, horizon)
    return pd.Series(np.where(r > threshold, 1, np.where(r < -threshold, -1, 0)), index=close.index, name="target")


def build_labels(data: pd.DataFrame | pd.Series, horizons: tuple[int, ...] = (1, 3, 6, 12, 24), threshold: float = 0.0) -> pd.DataFrame:
    """Build multi-horizon forward-return targets from an OHLCV frame or close series."""
    close = data["close"] if isinstance(data, pd.DataFrame) else data
    if isinstance(data, pd.DataFrame) and "close" not in data.columns:
        raise ValueError("build_labels requires a 'close' column")
    out = {}
    for h in horizons:
        out[f"forward_return_{h}"] = forward_return(close, h)
        out[f"direction_{h}"] = classify_return(close, h, threshold)
    return pd.DataFrame(out, index=close.index)
