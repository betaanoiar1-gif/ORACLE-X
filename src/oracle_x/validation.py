from __future__ import annotations

import numpy as np
import pandas as pd


def walk_forward_splits(index, train_size: int, test_size: int, step: int | None = None):
    step = step or test_size
    n = len(index)
    start = 0
    while start + train_size + test_size <= n:
        train = np.arange(start, start + train_size)
        test = np.arange(start + train_size, start + train_size + test_size)
        yield index[train], index[test]
        start += step


def stability_score(returns: pd.Series) -> dict:
    r = returns.dropna()
    if len(r) < 2:
        return {"n": len(r), "mean": float("nan"), "std": float("nan"), "worst": float("nan"), "score": float("nan")}
    mean, std = float(r.mean()), float(r.std())
    return {"n": len(r), "mean": mean, "std": std, "worst": float(r.min()), "score": mean / std if std else 0.0}


from .validation_firewall import ValidationConfig, ValidationReport, validate_robot

__all__ = ["walk_forward_splits", "stability_score", "ValidationConfig", "ValidationReport", "validate_robot"]
