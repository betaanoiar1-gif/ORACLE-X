from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from .data import load_csv
from .features import build_features
from .labels import classify_return
from .predictive import feature_power

@dataclass
class ResearchBundle:
    market: pd.DataFrame
    features: pd.DataFrame
    target: pd.Series
    feature_power: pd.DataFrame


def research_csv(path: str, horizon: int = 12) -> ResearchBundle:
    market = load_csv(path).ohlcv
    features = build_features(market)
    target = classify_return(market["close"], horizon=horizon)
    power = feature_power(features, target)
    return ResearchBundle(market, features, target, power)
