from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .backtest import ExecutionModel, run_backtest
from .factory import signal_from_dna
from .regime_engine import classify_regimes, regime_statistics
from .strategy import StrategyDNA
from .validation_firewall import ValidationConfig, validate_robot

@dataclass
class RobotValidationResult:
    robot_id: str
    accepted: bool
    validation: dict
    regime_stats: dict
    stress: dict
    reasons: list[str]


def stress_test(features: pd.DataFrame, close: pd.Series, dna: StrategyDNA,
                base: ExecutionModel | None = None, capital: float = 100.0) -> dict:
    base = base or ExecutionModel()
    scenarios = {
        "base": base,
        "fees_x2": ExecutionModel(base.fee_rate * 2, base.slippage_bps, base.spread_bps, base.funding_rate_per_bar),
        "slippage_x3": ExecutionModel(base.fee_rate, base.slippage_bps * 3, base.spread_bps, base.funding_rate_per_bar),
        "spread_x3": ExecutionModel(base.fee_rate, base.slippage_bps, base.spread_bps * 3, base.funding_rate_per_bar),
        "friction_heavy": ExecutionModel(base.fee_rate * 2, base.slippage_bps * 3, base.spread_bps * 3, base.funding_rate_per_bar * 2),
    }
    signal = signal_from_dna(features, dna)
    out = {}
    for name, model in scenarios.items():
        out[name] = dict(run_backtest(close, signal, model, capital).metrics)
    return out


def validate_robot_full(features: pd.DataFrame, close: pd.Series, dna: StrategyDNA,
                        execution: ExecutionModel | None = None,
                        config: ValidationConfig | None = None) -> RobotValidationResult:
    execution = execution or ExecutionModel()
    report = validate_robot(features, close, dna, execution, config=config)
    signal = signal_from_dna(features, dna)
    bt = run_backtest(close, signal, execution)
    regimes = classify_regimes(close)
    rs = regime_statistics(bt.trades["net_return"], regimes["regime"])
    stress = stress_test(features, close, dna, execution)
    reasons = list(report.rejection_reasons)
    base_return = stress["base"]["total_return"]
    heavy_return = stress["friction_heavy"]["total_return"]
    if base_return > 0 and heavy_return <= 0:
        reasons.append("friction_stress_failure")
    return RobotValidationResult(dna.fingerprint(), not reasons, report.to_dict(), rs.to_dict(), stress, reasons)
