---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Measure volatility from closing prices

Take log returns of the close with `LogReturn`, then their standard deviation
over a trailing window with `RollingStd` or with exponential weights with
`EwStd`, and annualize by the square root of the number of periods per year.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import LogReturn, RollingStd, EwStd

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
df = df[df["symbol"] == "BTC"].set_index("date")

annualize = np.sqrt(365)                     # crypto trades every day
df["ret"] = LogReturn()(df["close"])
df["vol_30d"] = RollingStd(30)(df["ret"]) * annualize
df["vol_ew"] = EwStd(span=30)(df["ret"]) * annualize
df[["close", "vol_30d", "vol_ew"]].tail().round(3)
```

A volatility of 0.5 means a standard deviation of 50% of the price over a year.
For daily stock data, which trades on about 252 days a year, use
`np.sqrt(252)`.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(df.index, df["vol_30d"], color="steelblue", label="RollingStd(30)")
ax.plot(df.index, df["vol_ew"], color="darkorange", lw=0.8,
        label="EwStd(span=30)")
ax.set_ylabel("BTC annualized volatility")
ax.legend(loc="upper right", fontsize=8)
plt.tight_layout()
```

`RollingStd` gives each of the last 30 days equal weight, so a large move
raises it for exactly 30 days and then drops out at once. `EwStd` lets the
weight of a move fade, so it reacts sooner and decays smoothly.

## Go deeper

- [Measure volatility from OHLC bars](ohlc.md): a more precise estimate from the same bars.
- [`LogReturn`](../../functions_fin/LogReturn.md), [`RollingStd`](../../functions_rolling/RollingStd.md), [`EwStd`](../../functions_ew/EwStd.md): parameters and warmup.
- [Window statistics notebook](../../notebooks/02-window-statistics.ipynb): rolling and exponentially weighted statistics side by side.
