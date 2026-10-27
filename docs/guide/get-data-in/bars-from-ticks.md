---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Build bars from trades

`Resample` groups trades into bars: `freq=` sets the bar length in index units,
and `agg="ohlcv"` turns `[price, size]` rows into open, high, low, close, and
volume.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import Resample, to_pandas

trades = pd.read_csv("../../notebooks/data/deribit_btc_perp_6h.csv")
# The timestamps are in milliseconds, so freq=60_000 is one minute.
values, index = Resample(freq=60_000, agg="ohlcv")(
    np.column_stack([trades["price"], trades["volume"].abs()]),
    trades["timestamp"].to_numpy())

columns = ["open", "high", "low", "close", "volume"]
bars = to_pandas(values, index, columns=columns)
bars.index = pd.to_datetime(bars.index, unit="ms")
print(f"{len(trades):,} trades, {len(bars)} one-minute bars")
bars.head()
```

A bar is emitted once a later trade shows that its minute is over, and a minute
without trades gives no bar. `count=50` instead of `freq=` makes bars of 50
trades each.

```{code-cell} ipython3
part = bars.iloc[:120]  # the first 120 bars
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 4.5), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
ax0.vlines(part.index, part["low"], part["high"], color="0.6", lw=0.8,
           label="high to low")
ax0.plot(part.index, part["close"], color="steelblue", lw=0.9, label="close")
ax0.set_ylabel("price (USD)")
ax0.legend(loc="upper left", fontsize=8)
ax1.bar(part.index, part["volume"], width=1 / 1440, color="0.4")
ax1.set_ylabel("volume")
plt.tight_layout()
```

## Go deeper

- [Bars from ticks notebook](../../notebooks/09-bars-from-ticks.ipynb): time, count, volume, and dollar bars, and empty buckets.
- [Custom and multi-column bars notebook](../../notebooks/10-custom-and-multi-column-bars.ipynb): a function as the bar reducer, and bars over several columns.
- [`Resample`](../../functions_streams/Resample.md): every `agg=` option and how empty buckets are filled.
- [Backtest a signal on OHLCV bars](../backtest/index.md): trade a signal on these bars, with costs.
