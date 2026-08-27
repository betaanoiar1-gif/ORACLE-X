from __future__ import annotations

import numpy as np


def equity_paths(returns, simulations=1000, seed=42):
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    rng = np.random.default_rng(seed)
    sampled = rng.choice(r, size=(simulations, len(r)), replace=True)
    return np.cumprod(1.0 + sampled, axis=1)


def max_drawdown(path):
    path = np.asarray(path)
    return float(np.min(path / np.maximum.accumulate(path) - 1.0))
