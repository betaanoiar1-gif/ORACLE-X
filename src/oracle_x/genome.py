from __future__ import annotations

from dataclasses import dataclass, asdict, replace
import hashlib
import json
import random
from typing import Iterable

@dataclass(frozen=True)
class RobotGenome:
    features: tuple[str, ...]
    direction: int = 1
    entry_threshold: float = 0.5
    exit_threshold: float = 0.0
    hold_bars: int = 12
    stop_loss: float = 0.02
    take_profit: float = 0.04
    risk_fraction: float = 0.01
    regime_filter: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "features", tuple(dict.fromkeys(self.features)))
        if not self.features: raise ValueError("Genome requires at least one feature")
        if self.direction not in (-1, 1): raise ValueError("direction must be -1 or 1")
        if self.hold_bars < 1: raise ValueError("hold_bars must be positive")
        if not 0 < self.risk_fraction <= 1: raise ValueError("risk_fraction must be in (0,1]")

    def fingerprint(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()[:20]

    def mutate(self, rng: random.Random, feature_pool: Iterable[str]) -> "RobotGenome":
        pool = tuple(dict.fromkeys(feature_pool))
        g = self
        choice = rng.randrange(7)
        if choice == 0 and pool:
            fs = list(g.features); fs[rng.randrange(len(fs))] = rng.choice(pool); g = replace(g, features=tuple(dict.fromkeys(fs)))
        elif choice == 1: g = replace(g, entry_threshold=max(0.0, min(1.0, g.entry_threshold + rng.gauss(0, .08))))
        elif choice == 2: g = replace(g, hold_bars=max(1, int(round(g.hold_bars + rng.gauss(0, 3)))))
        elif choice == 3: g = replace(g, stop_loss=max(.001, min(.20, g.stop_loss * (1 + rng.gauss(0, .15)))))
        elif choice == 4: g = replace(g, take_profit=max(.001, min(.50, g.take_profit * (1 + rng.gauss(0, .15)))))
        elif choice == 5: g = replace(g, risk_fraction=max(.001, min(.10, g.risk_fraction * (1 + rng.gauss(0, .20)))))
        else: g = replace(g, direction=-g.direction)
        return g

def crossover(a: RobotGenome, b: RobotGenome, rng: random.Random) -> RobotGenome:
    features = tuple(dict.fromkeys([*a.features[:max(1, len(a.features)//2)], *b.features[len(b.features)//2:]]))
    return RobotGenome(features=features or a.features, direction=rng.choice([a.direction,b.direction]), entry_threshold=rng.choice([a.entry_threshold,b.entry_threshold]), exit_threshold=rng.choice([a.exit_threshold,b.exit_threshold]), hold_bars=rng.choice([a.hold_bars,b.hold_bars]), stop_loss=rng.choice([a.stop_loss,b.stop_loss]), take_profit=rng.choice([a.take_profit,b.take_profit]), risk_fraction=rng.choice([a.risk_fraction,b.risk_fraction]), regime_filter=rng.choice([a.regime_filter,b.regime_filter]))
