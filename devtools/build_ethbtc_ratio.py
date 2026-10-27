#!/usr/bin/env python3
"""Build docs/notebooks/data/deribit_ethbtc_ratio_1m.csv for the Guide.

The mean-reversion chapter of the Guide uses one series: the log of the ETH/BTC
price ratio at one-minute bars, from the committed six hours of Deribit
perpetual trades. This script computes it with screamer and writes two
columns, ``timestamp`` (milliseconds since the epoch, start of the minute) and
``log_ratio``. Each asset's last trade price per minute comes from
``Resample(freq=60_000, agg="last", fill="carry")``, and ``CombineLatest``
aligns the two assets.

Run from the repository root and commit the result:

    python devtools/build_ethbtc_ratio.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from screamer import CombineLatest, Resample

DATA = Path(__file__).resolve().parent.parent / "docs" / "notebooks" / "data"


def closes(path: Path):
    """Last trade price of each minute, as a (values, index) tuple."""
    trades = pd.read_csv(path)
    return Resample(freq=60_000, agg="last", fill="carry")(
        trades["price"].to_numpy(), trades["timestamp"].to_numpy())


def main() -> None:
    prices, ts = CombineLatest()(closes(DATA / "deribit_btc_perp_6h.csv"),
                                 closes(DATA / "deribit_eth_perp_6h.csv"))
    out = pd.DataFrame({"timestamp": np.asarray(ts, dtype=np.int64),
                        "log_ratio": np.log(prices[:, 1] / prices[:, 0])})
    path = DATA / "deribit_ethbtc_ratio_1m.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
