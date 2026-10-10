---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Your first strategy

This recipe is the shortest complete path from price data to a tested
strategy: load daily bars, compute a trend signal, backtest it with fees and
slippage, summarize the result, and test whether its timing beats chance. The
data is daily Binance BTC/USDT bars from January 2022 to September 2026.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import (EwMean, Sign, FillNa, BacktestOHLCTarget,
                      backtest_report, position_test)

bars = pd.read_csv("../notebooks/data/binance_daily_20.csv", parse_dates=["date"])
btc = bars[bars["symbol"] == "BTC"]
o, h, l, c = (btc[k].to_numpy() for k in ["open", "high", "low", "close"])

# +1 (long one BTC) when the 20-day average is above the 100-day one, -1 below
signal = Sign()(FillNa(0.0)(EwMean(span=20)(c) - EwMean(span=100)(c)))

# 10 bps taker fee plus 5 bps slippage; the target from each close fills at the next open
engine = BacktestOHLCTarget(taker_fee=1e-3, slippage=5e-4)
out = engine(signal, o, h, l, c)

running, summary = backtest_report(out, periods_per_year=365)
test = position_test(engine, signal, o, h, l, c, n=999, periods_per_year=365)

for k in ["total_pnl", "total_cost", "max_drawdown", "sharpe", "round_trips"]:
    print(f"{k:13s} {summary[k]:12.2f}")
print(f"p-value       {test.p_value:12.3f}")
```

`signal` is decided on each day's close. `BacktestOHLCTarget` holds it and trades
it as a market order at the next day's open, so the raw signal goes in without a
manual lag. The output `out` has one row per day with the columns
`[equity, pnl, position, cost]`; `equity` is the cumulative profit in USD of a
one-BTC position. `backtest_report` reads its statistics off that array, with
the Sharpe ratio annualized over 365 trading days.

The rule made about $56,000 on one BTC over 21 round trips, after $3,500 of
costs, with an annualized Sharpe ratio of 0.40. `position_test` replays 999
copies of the same trades at randomly shifted times through the same engine;
15% of them reach a Sharpe ratio at least as high, so this result does not show
that the timing has an edge.

```{code-cell} ipython3
t = btc["date"].to_numpy()
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 5), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
ax0.plot(t, out[:, 0], color="steelblue", label="strategy equity")
ax0.plot(t, c - c[0], color="0.6", lw=0.8, label="buy and hold")
ax0.axhline(0, color="k", lw=0.5)
ax0.set_ylabel("USD, one BTC")
ax0.legend(loc="upper left", fontsize=8)
ax1.step(t, out[:, 2], where="post", color="indigo")
ax1.set_ylabel("position")
plt.tight_layout()
```

Related pages: [`BacktestOHLCTarget`](../functions_fin/BacktestOHLCTarget.md),
[`backtest_report`](../functions_fin/backtest_report.md),
[`position_test`](../functions_fin/position_test.md), and the
[parameter grid recipe](parameter-grid.md).
