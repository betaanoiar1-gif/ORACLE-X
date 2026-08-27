from __future__ import annotations

import os, time
from pathlib import Path
import requests
import pandas as pd

SYMBOL = os.getenv("ORACLE_SYMBOL", "BTCUSDT")
INTERVAL = os.getenv("ORACLE_INTERVAL", "1h")
TOTAL_CANDLES = int(os.getenv("ORACLE_CANDLES", "100000"))
OUT = Path(os.getenv("ORACLE_DATA_DIR", "data")) / f"{SYMBOL}_{INTERVAL}.csv"
URL = "https://api.binance.com/api/v3/klines"


def download():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    rows, end = [], None
    while len(rows) < TOTAL_CANDLES:
        params = {"symbol": SYMBOL, "interval": INTERVAL, "limit": 1000}
        if end is not None: params["endTime"] = end
        last_error = None
        for attempt in range(6):
            try:
                r = session.get(URL, params=params, timeout=(10, 60), headers={"User-Agent": "ORACLE-X/1.0"})
                r.raise_for_status()
                batch = r.json()
                last_error = None
                break
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(min(2 ** attempt, 20))
        if last_error is not None:
            raise RuntimeError(f"Binance request failed after retries: {last_error}")
        if not batch: break
        rows = batch + rows
        end = batch[0][0] - 1
        print(f"\rDownloaded: {min(len(rows), TOTAL_CANDLES):,}", end="", flush=True)
        if len(batch) < 1000: break
        time.sleep(0.35)
    rows = rows[-TOTAL_CANDLES:]
    cols = ["open_time","open","high","low","close","volume","close_time","quote_volume","trade_count","taker_buy_base_volume","taker_buy_quote_volume","ignore"]
    df = pd.DataFrame(rows, columns=cols)
    df["timestamp"] = pd.to_datetime(df.pop("open_time"), unit="ms", utc=True)
    for c in ["open","high","low","close","volume","quote_volume","trade_count","taker_buy_base_volume","taker_buy_quote_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[["timestamp","open","high","low","close","volume","quote_volume","trade_count","taker_buy_base_volume","taker_buy_quote_volume"]]
    df = df.drop_duplicates("timestamp").sort_values("timestamp").dropna()
    df.to_csv(OUT, index=False)
    print(f"\nSaved: {OUT} | rows={len(df):,} | {df.timestamp.min()} -> {df.timestamp.max()}")
    return df

if __name__ == "__main__":
    download()
