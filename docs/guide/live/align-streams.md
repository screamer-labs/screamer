---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Combine streams that tick at different times

`CombineLatest` joins streams that tick at different times into one row per
timestamp, holding the latest value of each stream, so functions of two inputs
such as `Div` can run on the rows.

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
from screamer import CombineLatest, Div

btc = pd.read_csv("../../notebooks/data/deribit_btc_perp_6h.csv")
eth = pd.read_csv("../../notebooks/data/deribit_eth_perp_6h.csv")

rows, t = CombineLatest()(btc["price"].to_numpy(), eth["price"].to_numpy(),
                          index=[btc["timestamp"].to_numpy(),
                                 eth["timestamp"].to_numpy()])
ratio = Div()(rows[:, 1], rows[:, 0])     # ETH price in BTC

print(f"{len(btc)} BTC trades, {len(eth)} ETH trades, {len(rows)} rows")
```

The data is six hours of Deribit BTC and ETH perpetual trades, each with its
own timestamps in milliseconds. `rows[:, 0]` is the last BTC price and
`rows[:, 1]` the last ETH price at each timestamp `t` where either market
traded. Several trades at the same millisecond, in one market or both, give
one row with the last price of each. The rows
start at the first ETH trade, because the default `emit="when_all"` waits
until every stream has a value; `emit="on_any"` starts at the first trade of
either and fills the missing stream with NaN.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(pd.to_datetime(t, unit="ms"), ratio, color="steelblue", lw=0.8)
ax.set_ylabel("ETH price in BTC")
plt.tight_layout()
```

## Go deeper

- [Multi-stream operators](../../notebooks/07-multi-stream-operators.ipynb): emit modes, `Merge`, `Dropna`, and `Filter`.
- [Streams, values, and alignment](../../multistream.md): the index model and multi-column streams.
- [`CombineLatest`](../../functions_streams/CombineLatest.md): parameters and buffering.
- [Wire functions into a pipeline](pipeline.md): align streams inside a pipeline that runs live.
