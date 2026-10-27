---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Wire functions into a pipeline

`Input` names a source and `Pipeline` compiles the functions applied to it into
one callable, which runs on stored arrays or on a live feed with the same
result.

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
from screamer import (Input, Pipeline, Resample, LogReturn, CombineLatest,
                      RollingCorr)

def minute_returns(price):
    return LogReturn(1)(Resample(freq=60_000, agg="last", fill="carry")(price))

btc, eth = Input("btc"), Input("eth")
both = CombineLatest()(minute_returns(btc), minute_returns(eth))
pipe = Pipeline(inputs=[btc, eth], outputs=[RollingCorr(30)(both)])

feeds = [pd.read_csv(f"../../notebooks/data/deribit_{coin}_perp_6h.csv")
         for coin in ["btc", "eth"]]
corr, t = pipe(*[(f["price"].to_numpy(), f["timestamp"].to_numpy())
                 for f in feeds])
print(pipe)
```

Applying a function to an `Input` records a step and computes nothing. Here
each trade feed goes through `Resample`, which keeps the last price of each
minute (`fill="carry"` repeats the previous price in a minute without trades),
and `LogReturn(1)`, the minute's log return. `CombineLatest` puts the two
returns on one row, and `RollingCorr(30)` correlates the BTC and ETH returns
over the last 30 minutes. The call binds the stored `(prices, timestamps)` of
each feed in the order of `inputs` and returns `(values, index)`, with one
row per minute.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(pd.to_datetime(t, unit="ms"), corr, color="steelblue")
ax.set_ylabel("30-minute correlation,\nBTC and ETH returns")
plt.tight_layout()
```

## Go deeper

- [Move a strategy from research to a live loop](research-to-live.md): run a pipeline one trade at a time with `live()`.
- [Pipelines notebook](../../notebooks/08-pipelines.ipynb): inspect a pipeline, give it several outputs, and save it as JSON.
- [Pipelines](../../pipelines.md): when to use a pipeline, the three ways to run it, and its guarantees.
- [`Pipeline`](../../functions_dag/Pipeline.md): constructor, feed forms, and return shapes.
