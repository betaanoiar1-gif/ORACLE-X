from __future__ import annotations

import json
from pathlib import Path


def save_discovery(result, output_dir: str = "artifacts/discovery") -> dict:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    reports = [r.__dict__ for r in result.validated]
    (root / "validated_robots.json").write_text(json.dumps(reports, default=str, indent=2), encoding="utf-8")
    generation_rows = [r.__dict__ for r in result.evolution.reports]
    (root / "generations.json").write_text(json.dumps(generation_rows, default=str, indent=2), encoding="utf-8")
    best = result.evolution.best
    (root / "best_robot.json").write_text(json.dumps({"robot_id": best.dna.fingerprint(), "dna": best.dna.__dict__, "metrics": best.metrics}, default=str, indent=2), encoding="utf-8")
    return {"output_dir": str(root), "robots": len(reports), "best_robot": best.dna.fingerprint()}
