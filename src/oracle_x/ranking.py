from __future__ import annotations

import pandas as pd

DEFAULT_WEIGHTS = {"oos_return": 0.30, "drawdown": 0.20, "stability": 0.20, "consistency": 0.15, "trade_count": 0.15}


def rank_robots(frame: pd.DataFrame, weights: dict | None = None) -> pd.DataFrame:
    w = weights or DEFAULT_WEIGHTS
    x = frame.copy()
    for col in w:
        if col not in x:
            x[col] = 0.0
        lo, hi = x[col].quantile(0.05), x[col].quantile(0.95)
        x[col + "_n"] = ((x[col] - lo) / (hi - lo)).clip(0, 1) if hi > lo else 0.5
    x["rank_score"] = sum(w[c] * x[c + "_n"] for c in w)
    return x.sort_values("rank_score", ascending=False).reset_index(drop=True)
