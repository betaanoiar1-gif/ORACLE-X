from __future__ import annotations

from dataclasses import dataclass
import random
import pandas as pd

from .strategy import StrategyDNA
from .factory import evaluate_population
from .evolve_engine import EvolutionEngine
from .robot_validator import validate_robot_full
from .validation_firewall import ValidationConfig

@dataclass
class DiscoveryResult:
    evolution: object
    validated: list


def seed_population(features: pd.DataFrame, size: int = 64, seed: int = 42) -> list[StrategyDNA]:
    rng = random.Random(seed)
    columns = [c for c in features.columns if pd.api.types.is_numeric_dtype(features[c])]
    if not columns:
        raise ValueError("No numeric features available")
    population, seen = [], set()
    while len(population) < size:
        dna = StrategyDNA(feature=rng.choice(columns), direction=rng.choice([-1, 1]),
                          threshold=rng.uniform(-1.0, 1.0), hold_bars=rng.randint(1, 24),
                          stop_loss=rng.uniform(0.002, 0.05), take_profit=rng.uniform(0.003, 0.10))
        fp = dna.fingerprint()
        if fp not in seen:
            population.append(dna); seen.add(fp)
    return population


def discover(features, close, population=None, generations=10, execution=None,
             validation=None, initial_capital=100.0):
    if population is None:
        population = seed_population(features)
    engine = EvolutionEngine(features, close, population, execution, initial_capital)
    evolution = engine.run(generations)
    validated, seen = [], set()
    for candidate in evolution.archive:
        rid = candidate.dna.fingerprint()
        if rid in seen: continue
        seen.add(rid)
        validated.append(validate_robot_full(features, close, candidate.dna, execution, validation))
    validated.sort(key=lambda x: (x.accepted, x.validation['oos_metrics']['total_return']), reverse=True)
    return DiscoveryResult(evolution, validated)
