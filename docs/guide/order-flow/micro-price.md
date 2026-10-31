---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Compute the micro-price and queue imbalance from quotes

`QueueImbalance` measures how much more size rests on the best bid than on the
best ask, from -1 to +1, and `MicroPrice` turns that imbalance into a fair
value that leans from the mid toward the side with less size.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import MicroPrice, QueueImbalance

# Synthetic L1 book: both queues drift around 4 BTC; when one empties, the
# price moves a tick toward it and both queues start again at 4 BTC.
rng = np.random.default_rng(0)
n, tick, depth = 20_000, 0.5, 4.0
bid, bid_size, ask_size = np.empty(n), np.empty(n), np.empty(n)
b, qb, qa = 29_000.0, depth, depth
for t in range(n):
    qb += 0.1 * (depth - qb) + rng.normal()
    qa += 0.1 * (depth - qa) + rng.normal()
    if qa <= 0: b, qb, qa = b + tick, depth, depth
    if qb <= 0: b, qb, qa = b - tick, depth, depth
    bid[t], bid_size[t], ask_size[t] = b, qb, qa
ask = bid + tick

imbalance = QueueImbalance()(bid_size, ask_size)
micro = MicroPrice()(bid, ask, bid_size, ask_size)
```

```{code-cell} ipython3
# Scoring only, looks ahead: the direction of the next mid move.
mid = pd.Series((bid + ask) / 2)
next_move = np.sign(mid.diff().shift(-1).replace(0, np.nan)).bfill()
bins = pd.cut(imbalance, [-1, -0.6, -0.2, 0.2, 0.6, 1])
(next_move > 0).groupby(bins, observed=True).mean().rename("share up").round(2)
```

```{code-cell} ipython3
w = slice(2_000, 2_400)
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.fill_between(np.arange(n)[w], bid[w], ask[w], color="0.85", step="post",
                label="bid to ask")
ax.plot(np.arange(n)[w], micro[w], color="crimson", lw=1, label="MicroPrice")
ax.set_xlabel("step")
ax.set_ylabel("price (USD)")
ax.ticklabel_format(axis="y", useOffset=False)
ax.legend(loc="upper left", fontsize=8)
plt.tight_layout()
```

The data files hold trades, not order book sizes, so this page uses a small
synthetic book. Each step, the size at the best bid and at the best ask changes
by a random amount and drifts back toward 4 BTC; when a queue empties, the
price moves one tick of 0.5 USD toward it. A real best bid or ask also moves
when its queue is traded or cancelled away, which is why the imbalance carries
information about the next move.

`QueueImbalance()(bid_size, ask_size)` is
`(bid_size - ask_size) / (bid_size + ask_size)`.
`MicroPrice()(bid, ask, bid_size, ask_size)` is `I * ask + (1 - I) * bid` with
`I = bid_size / (bid_size + ask_size)`, so a heavy bid pulls it toward the ask,
and a balanced book gives the mid. It is the first-order,
weighted-mid form of Stoikov's micro-price. The scoring cell looks ahead to the
next move of the mid: in this simulation, when the imbalance is above 0.6, the
next move was up 73% of the time, and when it is below -0.6, down 64% of the
time.

## Go deeper

- [`QueueImbalance`](../../functions_micro/QueueImbalance.md), [`MicroPrice`](../../functions_micro/MicroPrice.md): the formulas and the empty-book case.
- [`ContOFI`](../../functions_micro/ContOFI.md): order-flow imbalance from changes in the best bid and ask.
- [Backtest against bid and ask quotes (L1)](../backtest/quotes.md): trade against the bid and ask.
- [Toxicity and book notebook](../../notebooks/13-microstructure-toxicity-and-book.ipynb): book pressure next to VPIN and spreads.
