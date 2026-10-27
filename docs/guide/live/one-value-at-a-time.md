---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Process values one at a time

Construct a function once and call it with one number per event: each call
returns the output for that event and keeps the state for the next one.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import RollingZscore

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
btc = df[df["symbol"] == "BTC"].set_index("date")
close = btc["close"].to_numpy()

# Batch: one call on the whole array
batch = RollingZscore(50)(close)

# Live: one call per value, as each close arrives
score = RollingZscore(50)
live = np.array([score(x) for x in close])
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(btc.index, live, color="steelblue")
ax.axhline(0, color="k", lw=0.5)
ax.set_ylabel("50-day z-score of BTC close")
plt.tight_layout()
```

The loop stands in for a live feed: in production, `score(x)` runs in the
callback that receives each daily close. Both runs give NaN for the first 49
days, the warmup of the 50-day window, and identical values after that. A
batch call on `score`, such as `score(close)`, resets its running state, so
keep one instance for each live feed and use it for nothing else.

## Go deeper

- [Streaming live events](../../notebooks/06-streaming-live-events.ipynb): a live signal event by event, and streams with no end.
- [Streams in depth](../../streaming.md): the running series, lazy iterators, several sources, and errors.
- [Run a strategy in a live loop](live-loop.md): a handler that keeps its state in the functions.
