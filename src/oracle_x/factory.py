from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .strategy import StrategyDNA
from .backtest import ExecutionModel, run_backtest


@dataclass(frozen=True)
class CandidateResult:
    dna: StrategyDNA
    metrics: dict


def signal_from_dna(features: pd.DataFrame, dna: StrategyDNA) -> pd.Series:
    if dna.feature not in features.columns:
        raise KeyError(f"Unknown feature: {dna.feature}")
    x = pd.to_numeric(features[dna.feature], errors="coerce").fillna(0.0)
    signal = (x > dna.threshold).astype(float)
    if dna.direction < 0:
        signal = -signal
    return signal


def evaluate_candidate(features: pd.DataFrame, close: pd.Series, dna: StrategyDNA,
                       execution: ExecutionModel | None = None,
                       initial_capital: float = 100.0) -> CandidateResult:
    signal = signal_from_dna(features, dna)
    result = run_backtest(close, signal, execution or ExecutionModel(), initial_capital)
    metrics = dict(result.metrics)
    metrics["fitness"] = fitness(metrics)
    return CandidateResult(dna, metrics)


def fitness(metrics: dict) -> float:
    total_return = float(metrics.get("total_return", 0.0))
    max_dd = abs(float(metrics.get("max_drawdown", 0.0)))
    events = int(metrics.get("trade_events", 0))
    sharpe = float(metrics.get("sharpe_bar", 0.0))
    if events < 2:
        return float("-inf")
    return (total_return + 0.10 * sharpe) / (1.0 + 2.0 * max_dd)


def evaluate_population(features: pd.DataFrame, close: pd.Series,
                         population: Iterable[StrategyDNA],
                         execution: ExecutionModel | None = None,
                         initial_capital: float = 100.0) -> list[CandidateResult]:
    results = []
    for dna in population:
        results.append(evaluate_candidate(features, close, dna, execution, initial_capital))
    return sorted(results, key=lambda r: r.metrics["fitness"], reverse=True)
