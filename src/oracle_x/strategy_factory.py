from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from .genome import RobotGenome

@dataclass(frozen=True)
class FactoryConfig:
    min_features: int = 1
    max_features: int = 3
    threshold_grid: tuple[float, ...] = (.4, .5, .6, .7)
    hold_grid: tuple[int, ...] = (4, 8, 12, 24)
    stop_grid: tuple[float, ...] = (.01, .02, .03)
    target_grid: tuple[float, ...] = (.02, .04, .06)

def _score_frame(frame: pd.DataFrame) -> pd.Series:
    numeric = frame.select_dtypes(include=[np.number])
    return numeric.replace([np.inf, -np.inf], np.nan).rank(pct=True).mean(axis=1).fillna(.5)

def generate_candidates(features: pd.DataFrame, cfg: FactoryConfig = FactoryConfig(), max_candidates: int = 5000, seed: int = 42) -> list[RobotGenome]:
    cols = [c for c in features.columns if pd.api.types.is_numeric_dtype(features[c])]
    if not cols: raise ValueError("No numeric features available")
    rng = np.random.default_rng(seed)
    candidates: list[RobotGenome] = []
    seen = set()
    attempts = 0
    while len(candidates) < max_candidates and attempts < max_candidates * 20:
        attempts += 1
        k = int(rng.integers(cfg.min_features, min(cfg.max_features, len(cols)) + 1))
        fs = tuple(sorted(rng.choice(cols, size=k, replace=False).tolist()))
        g = RobotGenome(features=fs, direction=int(rng.choice([-1,1])), entry_threshold=float(rng.choice(cfg.threshold_grid)), hold_bars=int(rng.choice(cfg.hold_grid)), stop_loss=float(rng.choice(cfg.stop_grid)), take_profit=float(rng.choice(cfg.target_grid)))
        if g.fingerprint() not in seen:
            seen.add(g.fingerprint()); candidates.append(g)
    return candidates

def signal_from_genome(features: pd.DataFrame, genome: RobotGenome) -> pd.Series:
    missing = [f for f in genome.features if f not in features.columns]
    if missing: raise KeyError(f"Missing features: {missing}")
    x = features[list(genome.features)].replace([np.inf,-np.inf], np.nan)
    score = x.rank(pct=True).mean(axis=1).fillna(.5)
    edge = (score - .5) * 2 * genome.direction
    signal = pd.Series(0.0, index=features.index)
    signal[edge >= genome.entry_threshold] = genome.direction
    return signal
