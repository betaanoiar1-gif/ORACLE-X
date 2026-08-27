from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json

from .strategy import StrategyDNA


@dataclass
class RobotRecord:
    robot_id: str
    generation: int
    score: float
    dna: dict
    status: str = "candidate"


class RobotRegistry:
    def __init__(self, path: str = "artifacts/robot_registry.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def register(self, dna: StrategyDNA, generation: int, score: float, status: str = "candidate") -> RobotRecord:
        record = RobotRecord(dna.fingerprint(), generation, float(score), dna.to_dict(), status)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), sort_keys=True) + "\n")
        return record

    def load(self) -> list[RobotRecord]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(RobotRecord(**json.loads(line)))
        return rows
