from __future__ import annotations

import numpy as np
import pandas as pd


def forward_return(close: pd.Series, horizon: int = 12) -> pd.Series:
    """Future return used only as a target; never feed this into causal features."""
    return close.shift(-horizon) / close - 1.0


def classify_return(close: pd.Series, horizon: int = 12, threshold: float = 0.0) -> pd.Series:
    r = forward_return(close, horizon)
    return pd.Series(np.where(r > threshold, 1, np.where(r < -threshold, -1, 0)), index=close.index, name="target")
