---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# From research to a live loop

A [`Pipeline`](../pipelines.md) holds a strategy's whole computation, from raw
trades to a target position, as one object. This recipe builds one, backtests
it on the stored Deribit BTC-perpetual tape, and then runs the same object
event by event as a paper trader: each trade goes in as it arrives, and each
completed bar goes to a backtest engine that tracks the position and pnl.

## The pipeline

`Resample` buckets the trade prices into one-minute bars, `Select` splits the
bar into its four columns, and the trend rule compares a fast and a slow
exponential average of the close. `CombineLatest` puts the signal and the bar
on one row, `[signal, open, high, low, close]`, which is the input order of
`BacktestOHLCTarget`.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import (Input, Pipeline, Resample, Select, CombineLatest, EwMean,
                      Sub, Sign, FillNa, BacktestOHLCTarget)

price = Input("price")
bar = Resample(freq=60_000, agg="ohlc")(price)
o, h, l, c = (Select(k)(bar) for k in range(4))
spread = Sub()(CombineLatest()(EwMean(span=5)(c), EwMean(span=30)(c)))
signal = Sign()(FillNa(0.0)(spread))
strategy = Pipeline(inputs=[price], outputs=[CombineLatest()(signal, o, h, l, c)])
print(strategy)
```

## Research: the whole tape at once

Called with the stored `(prices, timestamps)`, the pipeline returns every bar's
row, and the engine backtests them in one call.

```{code-cell} ipython3
trades = pd.read_csv("../notebooks/data/deribit_btc_perp_6h.csv")
rows, minute = strategy((trades["price"].to_numpy(), trades["timestamp"].to_numpy()))
research = BacktestOHLCTarget(taker_fee=5e-4)(*rows.T)
print(f"{len(rows)} bars, final equity {research[-1, 0]:.1f} USD")
```

## Live: one trade at a time

`strategy.live()` opens a session on the same pipeline. `push` feeds one trade,
and `result()` returns the rows completed since the last call: a minute bar
completes when the first trade of the next minute arrives. A second engine,
called with one row at a time, plays the paper trader. It returns
`(equity, pnl, position, cost)` for that bar and keeps its position between
calls. In production the loop body runs in the feed's callback, and a timer
that calls `session.advance(now)` closes each minute on the clock even when no
trade arrives.

```{code-cell} ipython3
session = strategy.live()
paper = BacktestOHLCTarget(taker_fee=5e-4)
book = []

def on_bar(row):
    equity, pnl, position, cost = paper(*row)    # fills the previous bar's target
    book.append((equity, position))

for ts, p in zip(trades["timestamp"], trades["price"]):
    session.push("price", int(ts), float(p))
    for row in session.result()[0]:              # bars completed by this trade
        on_bar(row)

for row in session.flush().result()[0]:          # the last, partial minute
    on_bar(row)

book = np.array(book)
print(f"{len(book)} bars, final equity {book[-1, 0]:.1f} USD")
```

The paper trader's equity matches the research backtest bar for bar, so the
backtest describes what the live loop would have done on this tape.

```{code-cell} ipython3
t = pd.to_datetime(minute, unit="ms")
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 5), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
ax0.plot(t, research[:, 0], color="0.6", lw=3, label="research (batch)")
ax0.plot(t, book[:, 0], color="steelblue", lw=1, label="paper trader (live loop)")
ax0.axhline(0, color="k", lw=0.5)
ax0.set_ylabel("equity (USD, 1 BTC position)")
ax0.legend(loc="lower left", fontsize=8)
ax1.step(t, book[:, 1], where="post", color="indigo")
ax1.set_ylabel("position")
plt.tight_layout()
```

Related pages: [Pipelines](../pipelines.md),
[`Pipeline`](../functions_dag/Pipeline.md),
[`Resample`](../functions_streams/Resample.md), and the
[live event loop recipe](live-loop.md).
