from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class RiskConfig:
    risk_per_trade: float = 0.01
    max_position_fraction: float = 1.0
    max_total_exposure: float = 1.0
    max_drawdown_limit: float = 0.25


def position_fraction(equity: float, entry: float, stop: float, cfg: RiskConfig | None = None) -> float:
    cfg = cfg or RiskConfig()
    distance = abs(entry - stop)
    if equity <= 0 or entry <= 0 or distance <= 0:
        return 0.0
    fraction = cfg.risk_per_trade * equity / (equity * distance / entry)
    return max(0.0, min(cfg.max_position_fraction, fraction))


def portfolio_weights(scores: dict[str, float], max_total_exposure: float = 1.0) -> dict[str, float]:
    positive = {k: max(0.0, float(v)) for k, v in scores.items()}
    total = sum(positive.values())
    if total <= 0:
        return {k: 0.0 for k in scores}
    scale = min(1.0, max_total_exposure)
    return {k: scale * v / total for k, v in positive.items()}
