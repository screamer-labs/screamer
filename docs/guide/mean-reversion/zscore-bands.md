---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Trade the reversion with z-score bands

`RollingZscore` measures how far a series is from its recent mean,
`BandSignal` sells when it is far above and buys when it is far below,
`BacktestPriceTarget` runs that position with costs, and `backtest_report`
summarizes the result.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from screamer import (Resample, CombineLatest, RollingZscore, BandSignal, Lag,
                      BacktestPriceTarget, backtest_report)

def closes(path):  # last trade price of each minute
    trades = pd.read_csv(path)
    return Resample(freq=60_000, agg="last", fill="carry")(
        trades["price"].to_numpy(), trades["timestamp"].to_numpy())

data = "../../notebooks/data/"
prices, ts = CombineLatest()(closes(data + "deribit_btc_perp_6h.csv"),
                             closes(data + "deribit_eth_perp_6h.csv"))
df = pd.DataFrame({"ratio": np.log(prices[:, 1] / prices[:, 0])},
                  index=pd.to_datetime(ts, unit="ms"))

df["z"] = RollingZscore(30)(df["ratio"])
df["position"] = Lag(1)(BandSignal(entry=2.0, exit=0.0)(df["z"]))
summary = {}
for name, spread in [("no cost", 0.0), ("2 bp spread", 0.0002)]:
    bt = BacktestPriceTarget(spread=spread, size="weight", initial_capital=1.0)
    out = bt(df["position"], np.exp(df["ratio"]))   # the ETH/BTC price ratio
    df[name] = out[:, 0]                            # equity
    summary[name] = backtest_report(out)[1]
pd.DataFrame(summary).loc[["total_pnl", "round_trips", "win_rate"]]
```

The series is the minute log ETH/BTC ratio from the
[variance ratio page](variance-ratio.md), whose deviations halve in a few
minutes; a 30-minute z-score window spans several half-lives for most of the
session. `BandSignal`
goes short when the z-score reaches 2, long at -2, and flat when it crosses
0. `Lag(1)` trades one minute after the signal, at the next minute's price.
The backtest holds the ratio `ETH / BTC` as one price, with a weight of one
times the capital, so the equity is a return on capital. A real position
holds ETH against BTC, two legs that each pay costs.

```{code-cell} ipython3
fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
top.plot(df.index, df["z"], color="0.3", lw=0.8, label="RollingZscore(30)")
top.hlines([-2, 2], df.index[0], df.index[-1], color="crimson", ls="--", lw=0.8)
top.set_ylabel("z-score")
bottom.plot(df.index, df["no cost"] * 100, color="steelblue", label="no cost")
bottom.plot(df.index, df["2 bp spread"] * 100, color="darkorange",
            label="2 bp spread")
bottom.set_ylabel("return on capital (%)")
bottom.legend(loc="upper left", fontsize=8)
bottom.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
plt.tight_layout()
```

Without costs the 12 round trips earn 0.24% in six hours, and 11 of 12 make
money. The whole gain comes before 02:30; after that the trades break even.
A spread of 2 basis points, one basis point on each fill, takes all of it:
a round trip earns about 2 basis points on average, the cost of its two fills.

## Go deeper

- [`BandSignal`](../../functions_signal/BandSignal.md): entry, exit and stop levels.
- [`BacktestPriceTarget`](../../functions_fin/BacktestPriceTarget.md), [`backtest_report`](../../functions_fin/backtest_report.md): costs, sizing, and the statistics in the report.
- [A pairs trade](../../recipes/pairs-trade.md): ETH against BTC with a hedge ratio, both legs, and a test against luck.
