from __future__ import annotations

from dataclasses import dataclass
import random

from .strategy import StrategyDNA
from .evolution import mutate
from .factory import CandidateResult, evaluate_population
from .backtest import ExecutionModel


@dataclass
class GenerationReport:
    generation: int
    population_size: int
    best_id: str
    best_fitness: float
    mean_fitness: float
    best_metrics: dict


@dataclass
class EvolutionResult:
    best: CandidateResult
    reports: list[GenerationReport]
    archive: list[CandidateResult]


class EvolutionEngine:
    def __init__(self, features, close, population: list[StrategyDNA],
                 execution: ExecutionModel | None = None, initial_capital: float = 100.0,
                 elite_fraction: float = 0.15, tournament_size: int = 4,
                 mutation_rate: float = 0.75, seed: int = 42):
        if not population:
            raise ValueError("Population cannot be empty")
        self.features = features
        self.close = close
        self.population = list(population)
        self.execution = execution or ExecutionModel()
        self.initial_capital = initial_capital
        self.elite_fraction = min(max(elite_fraction, 0.01), 0.9)
        self.tournament_size = max(2, tournament_size)
        self.mutation_rate = min(max(mutation_rate, 0.0), 1.0)
        self.rng = random.Random(seed)

    def _tournament(self, results: list[CandidateResult]) -> StrategyDNA:
        sample = self.rng.sample(results, min(self.tournament_size, len(results)))
        return max(sample, key=lambda x: x.metrics["fitness"]).dna

    def _crossover(self, a: StrategyDNA, b: StrategyDNA) -> StrategyDNA:
        fields = ("feature", "direction", "threshold", "hold_bars", "stop_loss", "take_profit")
        values = {f: (getattr(a, f) if self.rng.random() < 0.5 else getattr(b, f)) for f in fields}
        return StrategyDNA(**values)

    def _next_population(self, ranked: list[CandidateResult]) -> list[StrategyDNA]:
        n = len(ranked)
        elite_n = max(1, int(n * self.elite_fraction))
        next_pop = [r.dna for r in ranked[:elite_n]]
        seen = {x.fingerprint() for x in next_pop}
        attempts = 0
        while len(next_pop) < n and attempts < n * 50:
            attempts += 1
            a = self._tournament(ranked)
            b = self._tournament(ranked)
            child = self._crossover(a, b)
            if self.rng.random() < self.mutation_rate:
                child = mutate(child, seed=self.rng.randrange(2**32))
            if child.fingerprint() not in seen:
                next_pop.append(child)
                seen.add(child.fingerprint())
        while len(next_pop) < n:
            child = mutate(ranked[self.rng.randrange(len(ranked))].dna, seed=self.rng.randrange(2**32))
            next_pop.append(child)
        return next_pop

    def run(self, generations: int = 10) -> EvolutionResult:
        if generations < 1:
            raise ValueError("generations must be >= 1")
        reports = []
        archive: list[CandidateResult] = []
        current = self.population
        for generation in range(1, generations + 1):
            ranked = evaluate_population(self.features, self.close, current, self.execution, self.initial_capital)
            finite = [r for r in ranked if r.metrics["fitness"] != float("-inf")]
            if not finite:
                raise RuntimeError("No viable strategies in population")
            archive.extend(finite[:max(1, len(finite) // 5)])
            best = finite[0]
            values = [r.metrics["fitness"] for r in finite]
            reports.append(GenerationReport(generation, len(current), best.dna.fingerprint(), best.metrics["fitness"], sum(values) / len(values), best.metrics))
            current = self._next_population(finite)
        archive.sort(key=lambda r: r.metrics["fitness"], reverse=True)
        return EvolutionResult(archive[0], reports, archive)
