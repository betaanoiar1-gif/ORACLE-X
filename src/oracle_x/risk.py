from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class RiskConfig:
    max_risk_per_trade: float = .01
    max_gross_exposure: float = 1.0
    max_drawdown_stop: float = .20
    min_stop_distance: float = .002

def position_size(equity: float, entry: float, stop: float, risk_fraction: float, cfg: RiskConfig = RiskConfig()) -> float:
    risk = min(max(risk_fraction, 0.0), cfg.max_risk_per_trade) * equity
    distance = abs(entry-stop)
    if distance <= 0: return 0.0
    return min(risk/distance, equity*cfg.max_gross_exposure/entry)

def risk_fraction_for_stop(entry: float, stop: float, equity: float, units: float) -> float:
    if equity <= 0: return 0.0
    return abs(entry-stop)*units/equity
