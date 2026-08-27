from __future__ import annotations
import numpy as np
import pandas as pd

def correlation_matrix(equity_curves: dict[str, pd.Series]) -> pd.DataFrame:
    if not equity_curves: return pd.DataFrame()
    returns = pd.DataFrame({k: v.pct_change() for k,v in equity_curves.items()})
    return returns.corr()

def allocate(scores: dict[str,float], correlation: pd.DataFrame | None = None, max_weight: float = .40) -> dict[str,float]:
    positive={k:max(0.,float(v)) for k,v in scores.items()}; total=sum(positive.values())
    if total<=0: return {k:0. for k in scores}
    w={k:v/total for k,v in positive.items()}
    for _ in range(20):
        changed=False
        for k in w:
            if w[k]>max_weight: w[k]=max_weight; changed=True
        rem=1-sum(w.values())
        if rem<=1e-9 or not changed: break
        free=[k for k in w if w[k]<max_weight-1e-9]
        s=sum(positive[k] for k in free)
        for k in free: w[k]+=rem*positive[k]/s if s else 0
    return w

def portfolio_equity(equities: dict[str,pd.Series], weights: dict[str,float], initial_capital: float=100.) -> pd.Series:
    frame=pd.DataFrame(equities).ffill().dropna(how='all')
    if frame.empty: return pd.Series(dtype=float)
    base=frame.iloc[0].replace(0,np.nan)
    normalized=frame.div(base).fillna(1.)
    w=pd.Series(weights).reindex(normalized.columns).fillna(0.)
    return initial_capital*(normalized*w).sum(axis=1)
