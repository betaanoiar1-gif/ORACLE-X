from __future__ import annotations

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def feature_power(features: pd.DataFrame, target: pd.Series, splits: int = 5) -> pd.DataFrame:
    rows = []
    y = target.reindex(features.index)
    mask = y.notna() & features.notna().all(axis=1)
    X, yy = features.loc[mask], y.loc[mask]
    for col in X.columns:
        x = X[[col]]
        aucs = []
        for train, test in TimeSeriesSplit(n_splits=splits).split(x):
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            model.fit(x.iloc[train], (yy.iloc[train] > 0).astype(int))
            p = model.predict_proba(x.iloc[test])[:, 1]
            try:
                aucs.append(roc_auc_score((yy.iloc[test] > 0).astype(int), p))
            except ValueError:
                pass
        rows.append({"feature": col, "oos_auc_mean": sum(aucs)/len(aucs) if aucs else float("nan"), "oos_auc_n": len(aucs)})
    return pd.DataFrame(rows).sort_values("oos_auc_mean", ascending=False)
