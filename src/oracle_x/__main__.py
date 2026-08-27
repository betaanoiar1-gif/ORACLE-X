from __future__ import annotations

import argparse
from .pipeline import research_csv


def main():
    p = argparse.ArgumentParser(description='ORACLE-X research runner')
    p.add_argument('csv', help='OHLCV CSV path')
    p.add_argument('--horizon', type=int, default=12)
    args = p.parse_args()
    bundle = research_csv(args.csv, horizon=args.horizon)
    print('rows:', len(bundle.market))
    print('features:', bundle.features.shape[1])
    print(bundle.feature_power.head(10).to_string(index=False))

if __name__ == '__main__':
    main()
