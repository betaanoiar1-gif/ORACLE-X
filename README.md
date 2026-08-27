# ORACLE-X

Autonomous Trading Research, Validation, Evolution and Robot Selection Engine.

ORACLE-X is designed as a research laboratory for discovering trading edges, converting validated relationships into executable strategy DNA, testing them under realistic execution assumptions, validating them out-of-sample, evolving robust candidates, and maintaining a ranked robot population.

## Architecture

Data -> Features -> Labels -> Predictive Power -> Strategy Factory -> Backtest -> Robustness -> Walk-Forward -> Regime Analysis -> Evolution -> Ranking -> Portfolio/Risk -> Shadow/Paper -> Monitoring

## Design principles

- No look-ahead leakage.
- Strict temporal separation between research and validation.
- Realistic fees, slippage, spread, funding and execution delay.
- Reproducible experiments and deterministic seeds.
- Survival depends on out-of-sample robustness, not backtest profit alone.
- Every robot has versioned DNA and an auditable validation record.

## Colab

The `notebooks/` directory contains the intended staged execution flow. The project can be cloned into Google Colab and executed stage by stage.

## Status

Foundation release: architecture and core research contracts. Subsequent stages implement and validate each engine against real market data.
