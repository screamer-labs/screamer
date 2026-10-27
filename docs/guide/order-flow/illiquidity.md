---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Measure illiquidity from daily bars

`AmihudIlliquidity` measures how far the price moves per unit of traded
value, in either direction, as the trailing mean of `|return| / value`. It
needs only returns and volume, so it works on ordinary OHLCV bars.

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
from screamer import AmihudIlliquidity, LogReturn

daily = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                    parse_dates=["date"])
close = daily.pivot(index="date", columns="symbol", values="close")
value = daily.pivot(index="date", columns="symbol", values="quote_volume")

amihud = AmihudIlliquidity(30)(LogReturn()(close), value)
# The last day, in basis points of return per million USD traded.
latest = pd.Series(1e10 * amihud[-1], index=close.columns).sort_values()
latest.round(2)
```

Each column of `close` and `value` is one of 20 coins, and the functions
apply to every column. `AmihudIlliquidity` takes the return first and the
traded value second, here the `quote_volume` column in USDT, a dollar
stablecoin, and averages over the last 30 days. `latest` holds the last day,
30 September 2026.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(latest.index, latest, color="darkorange")
ax.set_xscale("log")
ax.set_xlabel("AmihudIlliquidity(30), bp per million USD traded")
plt.tight_layout()
```

The ratio spans three orders of magnitude, from 0.09 for BTC to 109 for
ATOM. It suits bars with steady volume, such as daily bars: on one-minute
bars, minutes with a single small trade dominate the average.

## Go deeper

- [`AmihudIlliquidity`](../../functions_micro/AmihudIlliquidity.md): the ratio and how zero volume is handled.
- [Measure price impact](price-impact.md): impact per unit of signed flow, from trades.
- [Price impact notebook](../../notebooks/12-microstructure-price-impact.ipynb): Amihud and Kyle's lambda on BTC and ETH trades.
