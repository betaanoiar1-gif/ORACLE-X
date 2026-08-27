from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib, json

@dataclass(frozen=True)
class StrategyDNA:
    feature: str
    direction: int = 1
    threshold: float = 0.0
    hold_bars: int = 12
    stop_loss: float = 0.02
    take_profit: float = 0.04

    def fingerprint(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()[:16]

    def to_dict(self):
        d = asdict(self)
        d["id"] = self.fingerprint()
        return d
