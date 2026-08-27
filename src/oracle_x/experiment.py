from __future__ import annotations
import json, time
from pathlib import Path

def save_experiment(name: str, config: dict, result: dict, root: str='artifacts/experiments') -> str:
    stamp=time.strftime('%Y%m%d_%H%M%S'); path=Path(root)/f'{stamp}_{name}'; path.mkdir(parents=True,exist_ok=True)
    (path/'config.json').write_text(json.dumps(config,indent=2,default=str),encoding='utf-8')
    (path/'result.json').write_text(json.dumps(result,indent=2,default=str),encoding='utf-8')
    return str(path)
