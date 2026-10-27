---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Move a strategy from research to a live loop

Build the strategy as a `Pipeline`, call it on stored data for research, and
run the same object live with `strategy.live()`, one trade at a time, into a
paper trader.

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
from screamer import (Input, Pipeline, Resample, Select, CombineLatest, EwMean,
                      Sub, Sign, FillNa, BacktestOHLCTarget)

price = Input("price")
bar = Resample(freq=60_000, agg="ohlc")(price)
close = Select(3)(bar)
spread = Sub()(CombineLatest()(EwMean(span=5)(close), EwMean(span=30)(close)))
strategy = Pipeline(inputs=[price],
                    outputs=[CombineLatest()(Sign()(FillNa(0.0)(spread)), bar)])

trades = pd.read_csv("../../notebooks/data/deribit_btc_perp_6h.csv")
ts, px = trades["timestamp"].to_numpy(), trades["price"].to_numpy()
rows, minute = strategy((px, ts))                # research: every bar at once

session, paper, live = strategy.live(), BacktestOHLCTarget(taker_fee=5e-4), []
for t, p in zip(ts, px):
    session.push("price", int(t), float(p))
    live += [paper(*row)[0] for row in session.result()[0]]   # completed bars
live += [paper(*row)[0] for row in session.flush().result()[0]]
print(f"{len(rows)} bars, paper equity {live[-1]:.1f} USD")
```

```{code-cell} ipython3
t = pd.to_datetime(minute, unit="ms")
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(t, live, color="steelblue")
ax.axhline(0, color="k", lw=0.5)
ax.set_ylabel("equity (USD, 1 BTC position)")
plt.tight_layout()
```

The pipeline turns trades into one-minute OHLC bars, compares a 5-minute and
a 30-minute `EwMean` of the close, and returns rows of `[signal, open, high,
low, close]`, the input order of `BacktestOHLCTarget`. The engine trades each
signal at the next minute's open, holds 1 BTC long or short, and pays a 0.05%
taker fee on each fill, with no slippage. Called on the stored trades, the
pipeline returns every bar at once.

In the live loop, `session.push` feeds one trade and `session.result()` returns
the bars it completed: a minute bar completes when the first trade of the next
minute arrives, and `flush()` closes the last one. The engine `paper`, called
with one row at a time, is the paper trader: it returns
`[equity, pnl, position, cost]` for the bar and keeps its position between
calls. In production, the loop body runs in the feed's callback, and a timer
that calls `session.advance(now)` closes each minute on the clock when no trade
arrives.

The 5/30-minute crossover, paying 5 bp per fill, loses about 550 USD over the
six hours. The page shows the wiring from research to a live loop, not an
edge.

## Go deeper

- [Wire functions into a pipeline](pipeline.md): build and call a pipeline on stored data.
- [Pipelines](../../pipelines.md): `push`, `advance`, `flush`, and `result` of a live session.
- [`BacktestOHLCTarget`](../../functions_fin/BacktestOHLCTarget.md): fills, fees, and the output columns.
- [`Resample`](../../functions_streams/Resample.md): time bars, empty minutes, and the clock.
