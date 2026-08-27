from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def validate_predictive_power(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    splits: int = 5,
    min_auc: float = 0.52,
    min_stable_fraction: float = 0.60,
) -> pd.DataFrame:
    """Walk-forward OOS validation for every feature/target pair.

    Reports mean AUC, dispersion, worst split, and fraction of splits above
    the minimum AUC. No future observations are used for model fitting.
    """
    targets = [c for c in labels.columns if c.startswith("direction_")]
    if not targets:
        raise ValueError("No direction_* targets found")

    rows = []
    for target_name in targets:
        y0 = pd.to_numeric(labels[target_name], errors="coerce")
        for feature_name in features.columns:
            x0 = pd.to_numeric(features[feature_name], errors="coerce")
            mask = x0.notna() & y0.notna()
            x = x0.loc[mask].to_numpy().reshape(-1, 1)
            y = (y0.loc[mask].to_numpy() > 0).astype(int)
            if len(x) < max(100, splits * 20) or np.unique(y).size < 2:
                continue

            aucs = []
            cv = TimeSeriesSplit(n_splits=splits)
            for train, test in cv.split(x):
                if np.unique(y[train]).size < 2 or np.unique(y[test]).size < 2:
                    continue
                model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
                model.fit(x[train], y[train])
                prob = model.predict_proba(x[test])[:, 1]
                aucs.append(float(roc_auc_score(y[test], prob)))

            if not aucs:
                continue
            a = np.asarray(aucs)
            rows.append({
                "feature": feature_name,
                "target": target_name,
                "auc_mean": float(a.mean()),
                "auc_std": float(a.std(ddof=0)),
                "auc_min": float(a.min()),
                "auc_max": float(a.max()),
                "stable_fraction": float(np.mean(a >= min_auc)),
                "splits": int(len(a)),
                "passes": bool(a.mean() >= min_auc and np.mean(a >= min_auc) >= min_stable_fraction),
            })

    if not rows:
        return pd.DataFrame(columns=[
            "feature", "target", "auc_mean", "auc_std", "auc_min",
            "auc_max", "stable_fraction", "splits", "passes"
        ])
    return pd.DataFrame(rows).sort_values(
        ["passes", "auc_mean", "stable_fraction"],
        ascending=[False, False, False]
    ).reset_index(drop=True)


def summarize_feature_stability(validation: pd.DataFrame) -> pd.DataFrame:
    if validation.empty:
        return pd.DataFrame()
    return (
        validation.groupby("feature", as_index=False)
        .agg(
            target_count=("target", "count"),
            mean_auc=("auc_mean", "mean"),
            best_auc=("auc_mean", "max"),
            worst_auc=("auc_min", "min"),
            mean_stability=("stable_fraction", "mean"),
            passing_targets=("passes", "sum"),
        )
        .sort_values(["passing_targets", "mean_auc"], ascending=[False, False])
        .reset_index(drop=True)
    )
