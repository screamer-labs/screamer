---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# A live event loop

To run on a live feed, construct each operator once and call it with one value
per event. The call returns the new output immediately and the operator keeps
the state it needs for the next event. This recipe replays the Deribit BTC
trade tape as if each trade arrived from a websocket, and flags trades whose
price moves far from its recent average.

## The handler

Everything stateful lives in the operators, so the handler is a plain function.
`EwMean` tracks the price level and `RollingZscore` scores each trade's
deviation from it against the last 200 deviations. `EwMean` on the signed trade
size gives a running order-flow balance: positive when buyers dominate.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import EwMean, RollingZscore

level = EwMean(span=50)
score = RollingZscore(200)
flow = EwMean(span=100)

def on_trade(price, signed_size):
    """Called once per trade; returns (level, z-score, flow)."""
    m = level(price)
    return m, score(price - m), flow(signed_size)
```

## Feeding it

In production, call `on_trade` from your websocket or message-queue callback.
Here a loop over the recorded tape stands in for the feed.

```{code-cell} ipython3
trades = pd.read_csv("../notebooks/data/deribit_btc_perp_6h.csv")

rows, alerts = [], []
for ts, size, price in trades.itertuples(index=False):
    m, z, f = on_trade(price, size)
    rows.append((ts, price, m, z, f))
    if abs(z) > 3:
        alerts.append((ts, price, z))       # act on it here

rows = np.array(rows)
print(f"{len(rows)} trades processed, {len(alerts)} alerts")
```

The top panel shows the trades, the running level, and the alerts; the bottom
panel shows the order-flow balance.

```{code-cell} ipython3
t = pd.to_datetime(rows[:, 0], unit="ms")
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 5), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
ax0.plot(t, rows[:, 1], ".", ms=2, color="0.6", label="trades")
ax0.plot(t, rows[:, 2], color="steelblue", lw=1, label="EwMean(span=50)")
if alerts:
    a = np.array(alerts)
    ax0.plot(pd.to_datetime(a[:, 0], unit="ms"), a[:, 1], "o", mfc="none",
             color="crimson", ms=7, label="|z| > 3")
ax0.set_ylabel("price (USD)")
ax0.legend(loc="lower left", fontsize=8)
ax1.fill_between(t, rows[:, 4], 0, color="seagreen", alpha=0.5)
ax1.set_ylabel("signed flow")
plt.tight_layout()
```

The [streaming notebook](../notebooks/06-streaming-live-events.ipynb) covers
event-by-event calls and lazy generators in more depth. To run a graph of
several operators on more than one live feed, wrap it in a
[`Pipeline`](../pipelines.md).
