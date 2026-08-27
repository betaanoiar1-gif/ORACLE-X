from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class WalkForwardConfig:
    train_bars: int = 10000
    validation_bars: int = 2000
    test_bars: int = 2000
    step_bars: int = 2000

def walk_forward_splits(n: int, cfg: WalkForwardConfig = WalkForwardConfig()):
    start=0
    while start + cfg.train_bars + cfg.validation_bars + cfg.test_bars <= n:
        a=start; b=a+cfg.train_bars; c=b+cfg.validation_bars; d=c+cfg.test_bars
        yield {"train":(a,b),"validation":(b,c),"test":(c,d)}
        start += cfg.step_bars

def apply_split(frame: pd.DataFrame, split: tuple[int,int]) -> pd.DataFrame:
    return frame.iloc[split[0]:split[1]].copy()
