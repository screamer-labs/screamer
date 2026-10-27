---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Run functions in a live event loop

Construct the functions once, outside the loop, and call them in the handler
that receives each event: `EwMean` and `RollingZscore` keep their own state, so
the handler is a plain function of the new trade.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import EwMean, RollingZscore

level, score = EwMean(span=50), RollingZscore(200)

def on_trade(price):
    """Called once per trade; returns the price level and the z-score."""
    m = level(price)
    return m, score(price - m)

trades = pd.read_csv("../../notebooks/data/deribit_btc_perp_6h.csv")
rows, alerts = [], []
for ts, price in zip(trades["timestamp"], trades["price"]):
    m, z = on_trade(price)
    rows.append((ts, price, m))
    if abs(z) > 3:
        alerts.append((ts, price))      # act on the alert here
print(f"{len(rows)} trades processed, {len(alerts)} alerts")
```

```{code-cell} ipython3
rows, alerts = np.array(rows), np.array(alerts)
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(pd.to_datetime(rows[:, 0], unit="ms"), rows[:, 1], ".", ms=2,
        color="0.6", label="trades")
ax.plot(pd.to_datetime(rows[:, 0], unit="ms"), rows[:, 2], color="steelblue",
        lw=1, label="EwMean(span=50)")
ax.plot(pd.to_datetime(alerts[:, 0], unit="ms"), alerts[:, 1], "o", mfc="none",
        color="crimson", ms=7, label="|z| > 3")
ax.set_ylabel("BTC price (USD)")
ax.legend(loc="lower left", fontsize=8)
plt.tight_layout()
```

The loop replays six hours of Deribit BTC perpetual trades as if each arrived
from a websocket; in production, the feed's callback calls `on_trade`.
`EwMean(span=50)` tracks the price level, and `RollingZscore(200)` scores each
trade's distance from that level against the last 200 distances. A trade more
than three standard deviations away raises an alert.

## Go deeper

- [Process values one at a time](one-value-at-a-time.md): call a function once per event.
- [Move a strategy from research to a live loop](research-to-live.md): run a whole pipeline and a paper trader on the same loop.
- [Streaming live events](../../notebooks/06-streaming-live-events.ipynb): event-by-event calls and lazy generators.
- [Run screamer in the browser](browser.md): the same handler in JavaScript on a live exchange feed.
