from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

REQUIRED = ("open", "high", "low", "close", "volume")

@dataclass(frozen=True)
class MarketData:
    frame: pd.DataFrame

    def __post_init__(self):
        df = self.frame.copy()
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"Missing OHLCV columns: {missing}")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("MarketData index must be a DatetimeIndex")
        if not df.index.is_monotonic_increasing or df.index.has_duplicates:
            raise ValueError("Index must be unique and chronological")
        object.__setattr__(self, "frame", df)

    @property
    def ohlcv(self) -> pd.DataFrame:
        return self.frame.loc[:, REQUIRED].copy()


def load_csv(path: str, timestamp_col: str = "timestamp") -> MarketData:
    df = pd.read_csv(path)
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found")
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
    df = df.set_index(timestamp_col).sort_index()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return MarketData(df)
