---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Many assets at once: `(T, K)` arrays

A 2-D array of shape `(T, K)` holds `K` series side by side, with time on the
first axis. A single-input function applied to it processes each column as an
independent series and returns a `(T, K)` result, so one call covers every
asset. A two-input function such as `RollingCorr` instead reads the two columns
of a `(T, 2)` array as its two inputs. This recipe applies both to one-minute
BTC and ETH closes.

## Aligning the assets

The two perpetuals trade at different moments, so their one-minute bars do not
line up exactly. `CombineLatest` joins them on their timestamps and carries each
asset's latest close forward, giving one `(T, 2)` array.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import Resample, CombineLatest, LogReturn, RollingStd, RollingCorr

def closes(path):
    trades = pd.read_csv(path)
    return Resample(freq=60_000, agg="last")(
        trades["price"].to_numpy(), trades["timestamp"].to_numpy())

btc, btc_t = closes("../notebooks/data/deribit_btc_perp_6h.csv")
eth, eth_t = closes("../notebooks/data/deribit_eth_perp_6h.csv")

prices, ts = CombineLatest()(btc, eth, index=[btc_t, eth_t])
t = pd.to_datetime(ts, unit="ms")
prices.shape
```

## One call per statistic

`LogReturn` and `RollingStd` see a `(T, 2)` input and return `(T, 2)`: one
column per asset. `RollingCorr` takes two inputs, so the two columns of the
return array are its `x` and `y`, and the result is one column.

```{code-cell} ipython3
returns = LogReturn()(prices)                 # (T, 2)
vol = RollingStd(30)(returns) * 1e4           # (T, 2), in basis points per minute
corr = RollingCorr(30)(returns)               # (T,), correlation of BTC and ETH
returns.shape, vol.shape, corr.shape
```

```{code-cell} ipython3
fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(9, 6.5), sharex=True)
ax0.plot(t, prices / prices[0] - 1, lw=1)
ax0.set_ylabel("return since start")
ax0.legend(["BTC", "ETH"], fontsize=8, loc="lower left")
ax1.plot(t, vol, lw=1)
ax1.set_ylabel("vol (bps/min)")
ax2.plot(t, corr, color="0.25", lw=1)
ax2.set_ylabel("RollingCorr(30)")
plt.tight_layout()
```

The same rule extends to any number of columns, and to the single-asset
backtest engines: a `(T, K)` target and a `(T, K)` price run `K` independent
backtests and return `(T, K, 4)`, and a `(T, K)` target with one `(T,)` price
runs `K` variants of a strategy on the same market. The [polymorphic API](../polymorphic_api.md)
page defines the shape rules for every input type. To trade the `K` assets as
one book from a shared account, use the portfolio engines in
[Portfolios and pairs](../portfolio_backtesting.md); the
[cross-sectional momentum recipe](cross-sectional-momentum.md) builds such a
book.

Related pages: [`CombineLatest`](../functions_streams/CombineLatest.md),
[`RollingStd`](../functions_rolling/RollingStd.md),
[`RollingCorr`](../functions_fin/RollingCorr.md), and
[`LogReturn`](../functions_fin/LogReturn.md).
