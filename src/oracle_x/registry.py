from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
from .genome import RobotGenome

class RobotRegistry:
    def __init__(self, root: str | Path = "artifacts/robots"):
        self.root=Path(root); self.root.mkdir(parents=True, exist_ok=True)
    def save(self, genome: RobotGenome, metrics: dict, status: str = "candidate") -> Path:
        payload={"genome":genome.__dict__ | {"features":list(genome.features)},"metrics":metrics,"status":status}
        path=self.root/f"{genome.fingerprint()}.json"; path.write_text(json.dumps(payload,indent=2,default=str)); return path
    def load_all(self) -> pd.DataFrame:
        rows=[]
        for p in self.root.glob("*.json"):
            try:
                d=json.loads(p.read_text()); row={"robot_id":p.stem,"status":d.get("status")}; row.update(d.get("metrics",{})); rows.append(row)
            except Exception: continue
        return pd.DataFrame(rows)
