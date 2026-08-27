from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from .data import load_csv
from .features import build_features
from .labels import classify_return
from .predictive import feature_power
from .backtest import ExecutionModel
from .evolve_engine import EvolutionEngine
from .robot_validator import validate_robot_full, RobotValidationResult
from .strategy import StrategyDNA
from .validation_firewall import ValidationConfig

@dataclass
class ResearchBundle:
    market: pd.DataFrame
    features: pd.DataFrame
    target: pd.Series
    feature_power: pd.DataFrame


def research_csv(path: str, horizon: int = 12) -> ResearchBundle:
    market = load_csv(path).ohlcv
    features = build_features(market)
    target = classify_return(market["close"], horizon=horizon)
    power = feature_power(features, target)
    return ResearchBundle(market, features, target, power)


def discover_and_validate(features: pd.DataFrame, close: pd.Series,
                          population: list[StrategyDNA], generations: int = 10,
                          execution: ExecutionModel | None = None,
                          validation: ValidationConfig | None = None,
                          initial_capital: float = 100.0):
    execution = execution or ExecutionModel()
    evolution = EvolutionEngine(features, close, population, execution, initial_capital)
    evolution_result = evolution.run(generations)
    validated: list[RobotValidationResult] = []
    seen: set[str] = set()
    for candidate in evolution_result.archive:
        rid = candidate.dna.fingerprint()
        if rid in seen:
            continue
        seen.add(rid)
        validated.append(validate_robot_full(features, close, candidate.dna, execution, validation))
    validated.sort(key=lambda x: (x.accepted, x.validation["oos_metrics"]["total_return"]), reverse=True)
    return evolution_result, validated
