from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

from .backtest import ExecutionModel, run_backtest
from .factory import signal_from_dna
from .strategy import StrategyDNA


@dataclass(frozen=True)
class ValidationConfig:
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    min_trades: int = 10
    min_oos_return: float = 0.0
    max_oos_drawdown: float = 0.30
    min_oos_sharpe: float = 0.0
    walk_forward_splits: int = 5


@dataclass
class ValidationReport:
    robot_id: str
    passed: bool
    train_metrics: dict
    validation_metrics: dict
    oos_metrics: dict
    walk_forward: dict
    rejection_reasons: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _metrics(result) -> dict:
    return dict(result.metrics)


def _slice_metrics(features, close, dna, start, end, execution, capital):
    f = features.iloc[start:end]
    c = close.iloc[start:end]
    signal = signal_from_dna(f, dna)
    return _metrics(run_backtest(c, signal, execution, capital))


def _walk_forward(features, close, dna, execution, capital, splits):
    n = len(close)
    if n < splits + 1:
        return {"splits": 0, "passed": False, "results": []}
    results = []
    edges = np.linspace(0, n, splits + 1, dtype=int)
    for i in range(splits):
        start, end = edges[i], edges[i + 1]
        if end - start < 2:
            continue
        m = _slice_metrics(features, close, dna, start, end, execution, capital)
        results.append(m)
    valid = [m for m in results if m["trade_events"] > 0]
    return {
        "splits": len(results),
        "valid_splits": len(valid),
        "positive_splits": sum(m["total_return"] > 0 for m in valid),
        "mean_return": float(np.mean([m["total_return"] for m in valid])) if valid else 0.0,
        "worst_drawdown": float(min((m["max_drawdown"] for m in valid), default=0.0)),
        "results": results,
    }


def validate_robot(features: pd.DataFrame, close: pd.Series, dna: StrategyDNA,
                   execution: ExecutionModel | None = None,
                   initial_capital: float = 100.0,
                   config: ValidationConfig | None = None) -> ValidationReport:
    cfg = config or ValidationConfig()
    execution = execution or ExecutionModel()
    n = min(len(features), len(close))
    features, close = features.iloc[:n], close.iloc[:n]
    if n < 30:
        raise ValueError("Validation requires at least 30 aligned bars")
    train_end = max(2, int(n * cfg.train_fraction))
    val_end = max(train_end + 1, int(n * (cfg.train_fraction + cfg.validation_fraction)))
    val_end = min(val_end, n - 1)

    train = _slice_metrics(features, close, dna, 0, train_end, execution, initial_capital)
    validation = _slice_metrics(features, close, dna, train_end, val_end, execution, initial_capital)
    oos = _slice_metrics(features, close, dna, val_end, n, execution, initial_capital)
    wf = _walk_forward(features, close, dna, execution, initial_capital, cfg.walk_forward_splits)

    reasons = []
    if oos["trade_events"] < cfg.min_trades:
        reasons.append("insufficient_oos_trades")
    if oos["total_return"] < cfg.min_oos_return:
        reasons.append("oos_return_below_threshold")
    if abs(oos["max_drawdown"]) > cfg.max_oos_drawdown:
        reasons.append("oos_drawdown_exceeded")
    if oos["sharpe_bar"] < cfg.min_oos_sharpe:
        reasons.append("oos_sharpe_below_threshold")
    if wf["valid_splits"] < max(2, cfg.walk_forward_splits // 2):
        reasons.append("insufficient_walk_forward_splits")
    if wf["valid_splits"] and wf["positive_splits"] < int(np.ceil(wf["valid_splits"] * 0.5)):
        reasons.append("walk_forward_consistency_failure")

    return ValidationReport(dna.fingerprint(), not reasons, train, validation, oos, wf, reasons)
