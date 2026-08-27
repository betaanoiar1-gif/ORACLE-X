from __future__ import annotations

import argparse
from pathlib import Path

from .backtest import ExecutionModel
from .data import load_csv
from .discovery import seed_population
from .features import build_features
from .labels import classify_return
from .predictive import feature_power
from .reporting import save_discovery
from .discovery import discover
from .validation_firewall import ValidationConfig


def run(csv_path: str, population_size: int = 64, generations: int = 10,
        horizon: int = 12, capital: float = 100.0, seed: int = 42):
    market = load_csv(csv_path).ohlcv
    features = build_features(market).replace([float("inf"), float("-inf")], float("nan"))
    target = classify_return(market["close"], horizon=horizon)
    power = feature_power(features, target)
    ranked = power.dropna(subset=["oos_auc_mean"]).sort_values("oos_auc_mean", ascending=False)
    ranked = ranked[(ranked["oos_auc_mean"] - 0.5).abs() >= 0.0]
    # Keep discovery finite and deterministic; seed only from numerically usable features.
    usable = ranked["feature"].tolist()
    if usable:
        features = features[usable]
    population = seed_population(features, size=population_size, seed=seed)
    execution = ExecutionModel()
    validation = ValidationConfig()
    result = discover(features, market["close"], population, generations, execution, validation, capital)
    output = save_discovery(result)
    return market, features, power, result, output


def main():
    parser = argparse.ArgumentParser(description="ORACLE-X end-to-end research run")
    parser.add_argument("csv")
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--capital", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    _, features, power, result, output = run(args.csv, args.population, args.generations, args.horizon, args.capital, args.seed)
    print(f"ORACLE-X COMPLETE: features={features.shape} generations={len(result.evolution.reports)} validated={len(result.validated)}")
    print(power.head(10).to_string(index=False))
    print(output)


if __name__ == "__main__":
    main()
