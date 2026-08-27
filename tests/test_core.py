import numpy as np
import pandas as pd
from oracle_x.genome import RobotGenome, crossover
from oracle_x.strategy_factory import signal_from_genome
from oracle_x.backtest import run_backtest
from oracle_x.walk_forward import walk_forward_splits, WalkForwardConfig

def test_genome_fingerprint_stable():
    g=RobotGenome(("x",), hold_bars=5)
    assert g.fingerprint()==RobotGenome(("x",), hold_bars=5).fingerprint()

def test_signal_no_future_shift():
    idx=pd.RangeIndex(100); f=pd.DataFrame({"x":np.linspace(0,1,100)},index=idx)
    g=RobotGenome(("x",),entry_threshold=.5)
    s=signal_from_genome(f,g)
    assert s.index.equals(idx)
    assert s.iloc[0]==0

def test_backtest_finite():
    close=pd.Series(np.linspace(100,110,100)); signal=pd.Series(1.0,index=close.index)
    r=run_backtest(close,signal)
    assert np.isfinite(r.metrics["final_equity"])

def test_walk_forward_count():
    xs=list(walk_forward_splits(14000,WalkForwardConfig(10000,2000,2000,2000)))
    assert len(xs)==1
