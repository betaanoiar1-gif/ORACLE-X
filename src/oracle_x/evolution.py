from __future__ import annotations

from dataclasses import replace
import random
from .strategy import StrategyDNA


def mutate(dna: StrategyDNA, seed: int | None = None) -> StrategyDNA:
    rng = random.Random(seed)
    field = rng.choice(["threshold", "hold_bars", "stop_loss", "take_profit", "direction"])
    if field == "threshold":
        return replace(dna, threshold=dna.threshold + rng.uniform(-0.01, 0.01))
    if field == "hold_bars":
        return replace(dna, hold_bars=max(1, dna.hold_bars + rng.choice([-3, -1, 1, 3])))
    if field == "stop_loss":
        return replace(dna, stop_loss=max(0.001, dna.stop_loss * rng.uniform(0.8, 1.2)))
    if field == "take_profit":
        return replace(dna, take_profit=max(0.001, dna.take_profit * rng.uniform(0.8, 1.2)))
    return replace(dna, direction=-dna.direction)


def tournament(population, scores, k=3):
    candidates = random.sample(list(population), min(k, len(population)))
    return max(candidates, key=lambda x: scores.get(x.fingerprint(), float("-inf")))
