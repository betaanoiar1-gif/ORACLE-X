from __future__ import annotations

import pandas as pd
import numpy as np

def correlation_matrix(return_series: dict[str,pd.Series]) -> pd.DataFrame:
    return pd.DataFrame(return_series).corr()

def inverse_volatility_weights(returns: pd.DataFrame, max_weight: float = .50) -> pd.Series:
    vol=returns.std().replace(0,np.nan); inv=(1/vol).replace([np.inf,-np.inf],np.nan).fillna(0)
    if inv.sum()==0: return pd.Series(0.0,index=returns.columns)
    w=inv/inv.sum()
    for _ in range(20):
        over=w>max_weight
        if not over.any(): break
        excess=(w[over]-max_weight).sum(); w[over]=max_weight
        under=~over
        if under.any(): w[under]+=excess*w[under]/w[under].sum()
    return w/w.sum()

def portfolio_equity(returns: pd.DataFrame, weights: pd.Series, initial_capital: float = 100.0) -> pd.Series:
    r=returns.mul(weights.reindex(returns.columns).fillna(0),axis=1).sum(axis=1)
    return initial_capital*(1+r.fillna(0)).cumprod()
