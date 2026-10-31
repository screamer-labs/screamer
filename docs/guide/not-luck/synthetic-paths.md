---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Generate synthetic price paths with the same return distribution

`permute_bars` returns copies of an OHLC series with its daily gaps and bar
shapes in random order, and `permute_returns` does the same for a single price
series: every copy has the same return distribution as the real market but none
of its serial structure.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import (permute_bars, permute_returns, LogReturn, RollingStd,
                      MaxDrawdown)

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
btc = df[df["symbol"] == "BTC"].set_index("date")
o, h, l, c = (btc[k].to_numpy() for k in ["open", "high", "low", "close"])

po, ph, pl, pc = permute_bars(o, h, l, c, 500, seed=1)
paths = permute_returns(c, 500, seed=1)
for name, x in [("real", c), ("bars", pc), ("returns", paths)]:
    vol = RollingStd(30)(LogReturn()(x))
    print(f"{name:8} daily std {np.nanstd(LogReturn()(x)):.4f}, "
          f"spread of 30-day vol {np.nanmean(np.nanstd(vol, axis=0)):.4f}, "
          f"median max drawdown {np.median(MaxDrawdown()(x)[-1]):.0%}")
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(btc.index, pc[:, :20], color="lightsteelblue", lw=0.6)
ax.plot(btc.index, c, color="crimson", lw=1.2, label="BTC close")
ax.set_yscale("log")
ax.set_ylabel("close (USD)")
ax.set_title("BTC and 20 of the 500 permuted copies", fontsize=10)
ax.legend(fontsize=8)
plt.tight_layout()
```

The data are daily Binance BTC/USDT bars from January 2022 to September 2026.
Each of the 500 copies from `permute_bars` splits every bar into its gap (the
open against the previous close) and its shape (the high, low, and close against
the open), shuffles the gaps and the shapes separately, and rebuilds the path
from the first bar. Every copy starts at the first close and ends at the last
one, and every bar is valid. `permute_returns` shuffles the daily log returns of
the close alone.

The daily standard deviation is the same in every copy, because the copies hold
the same returns. What changes is their order. The standard deviation over time
of the 30-day volatility falls from 0.0085 to 0.0059, because calm and volatile
days no longer cluster, and the median copy has a maximum drawdown of 62%
against 67% for BTC. A strategy that profits from momentum, mean reversion, or
volatility clustering has none to find in the copies, which makes them the null
market of [Run a Monte Carlo permutation test on a whole strategy](permutation-test.md).

## Go deeper

- [Run a Monte Carlo permutation test on a whole strategy](permutation-test.md): run a strategy on these copies and get a p-value.
- [I tuned my parameters. Is the result real?](../../notebooks/21-is-the-tuned-result-real.ipynb): a parameter search on permuted bars.
- [`permute_bars`](../../functions_fin/permute_bars.md), [`permute_returns`](../../functions_fin/permute_returns.md): `mode="diff"` and `"level"` for spreads and rates, and volume that moves with its bar.
