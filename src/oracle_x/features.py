from __future__ import annotations

import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Causal OHLCV features. Every feature uses current/past bars only."""
    x = df.copy()
    close, high, low, volume = x["close"], x["high"], x["low"], x["volume"]
    out = pd.DataFrame(index=x.index)
    for n in (1, 3, 5, 10, 20, 50):
        out[f"ret_{n}"] = close.pct_change(n)
        out[f"vol_{n}"] = close.pct_change().rolling(n).std()
        out[f"range_{n}"] = ((high - low) / close).rolling(n).mean()
        out[f"volume_z_{n}"] = (volume - volume.rolling(n).mean()) / volume.rolling(n).std()
    out["body"] = (close - x["open"]) / x["open"]
    out["upper_wick"] = (high - np.maximum(x["open"], close)) / close
    out["lower_wick"] = (np.minimum(x["open"], close) - low) / close
    out["ema_fast"] = close / close.ewm(span=12, adjust=False).mean() - 1
    out["ema_slow"] = close / close.ewm(span=48, adjust=False).mean() - 1
    return out.replace([np.inf, -np.inf], np.nan)
