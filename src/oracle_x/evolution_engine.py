from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable, Iterable

from .strategy import StrategyDNA
from .evolution import mutate


@dataclass(frozen=True)
class EvolutionConfig:
    population_size: int = 100
    generations: int = 20
    elite_fraction: float = 0.10
    tournament_size: int = 3
    mutation_rate: float = 0.80
    crossover_rate: float = 0.90
    seed: int = 42


@dataclass(frozen=True)
class Individual:
    dna: StrategyDNA
    score: float = float("-inf")


@dataclass
class EvolutionResult:
    best: Individual
    population: list[Individual]
    history: list[dict]
    archive: list[Individual]


def crossover(a: StrategyDNA, b: StrategyDNA, rng: random.Random) -> StrategyDNA:
    # Uniform gene crossover. The feature itself is inherited from either parent.
    return StrategyDNA(
        feature=a.feature if rng.random() < 0.5 else b.feature,
        direction=a.direction if rng.random() < 0.5 else b.direction,
        threshold=a.threshold if rng.random() < 0.5 else b.threshold,
        hold_bars=a.hold_bars if rng.random() < 0.5 else b.hold_bars,
        stop_loss=a.stop_loss if rng.random() < 0.5 else b.stop_loss,
        take_profit=a.take_profit if rng.random() < 0.5 else b.take_profit,
    )


def _select(population: list[Individual], rng: random.Random, k: int) -> Individual:
    sample = rng.sample(population, min(k, len(population)))
    return max(sample, key=lambda x: x.score)


def evolve(
    initial_population: Iterable[StrategyDNA],
    evaluator: Callable[[StrategyDNA], float],
    config: EvolutionConfig = EvolutionConfig(),
) -> EvolutionResult:
    rng = random.Random(config.seed)
    population = [Individual(dna) for dna in initial_population]
    if not population:
        raise ValueError("initial_population must not be empty")
    if config.population_size < 2:
        raise ValueError("population_size must be >= 2")
    if not 0 < config.elite_fraction <= 1:
        raise ValueError("elite_fraction must be in (0, 1]")

    # Resize initial population deterministically by repeating candidates.
    base = [x.dna for x in population]
    while len(population) < config.population_size:
        population.append(Individual(base[len(population) % len(base)]))
    population = population[: config.population_size]

    archive: dict[str, Individual] = {}
    history: list[dict] = []

    for generation in range(config.generations):
        evaluated = [Individual(x.dna, float(evaluator(x.dna))) for x in population]
        evaluated.sort(key=lambda x: x.score, reverse=True)
        for item in evaluated:
            old = archive.get(item.dna.fingerprint())
            if old is None or item.score > old.score:
                archive[item.dna.fingerprint()] = item

        elite_n = max(1, int(round(config.population_size * config.elite_fraction)))
        elites = evaluated[:elite_n]
        history.append({
            "generation": generation,
            "best_score": elites[0].score,
            "mean_score": sum(x.score for x in evaluated) / len(evaluated),
            "worst_score": evaluated[-1].score,
            "unique_genomes": len({x.dna.fingerprint() for x in evaluated}),
        })

        if generation == config.generations - 1:
            population = evaluated
            break

        next_population = list(elites)
        while len(next_population) < config.population_size:
            p1 = _select(evaluated, rng, config.tournament_size).dna
            p2 = _select(evaluated, rng, config.tournament_size).dna
            child = crossover(p1, p2, rng) if rng.random() < config.crossover_rate else p1
            if rng.random() < config.mutation_rate:
                child = mutate(child, seed=rng.randrange(2**32))
            next_population.append(Individual(child))
        population = next_population

    final = sorted(population, key=lambda x: x.score, reverse=True)
    best = max(archive.values(), key=lambda x: x.score)
    return EvolutionResult(best=best, population=final, history=history, archive=sorted(archive.values(), key=lambda x: x.score, reverse=True))
