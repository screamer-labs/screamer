---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Measure volatility from OHLC bars

`RollingYangZhangVol` estimates volatility from the open, high, low and close
of each bar. It uses more of the price path than the close alone, so it gives a
more precise estimate from the same number of bars.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import LogReturn, RollingStd, RollingYangZhangVol

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
df = df[df["symbol"] == "BTC"].set_index("date")

annualize = np.sqrt(365)
df["vol_close"] = RollingStd(30)(LogReturn()(df["close"])) * annualize
df["vol_yz"] = RollingYangZhangVol(30)(
    df["open"], df["high"], df["low"], df["close"]) * annualize
df[["vol_close", "vol_yz"]].tail().round(3)
```

```{code-cell} ipython3
year = df.loc["2025"]
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(year.index, year["vol_close"], color="0.5",
        label="RollingStd(30) of close-to-close log returns")
ax.plot(year.index, year["vol_yz"], color="crimson",
        label="RollingYangZhangVol(30)")
ax.set_ylabel("BTC annualized volatility")
ax.legend(loc="upper right", fontsize=8)
plt.tight_layout()
```

The inputs come in the order open, high, low, close. The Yang-Zhang estimator
combines the variance of the gap from the previous close to the open, the
variance of the open-to-close return, and the Rogers-Satchell estimate from the
high and low of each bar. Its output is per bar, like `RollingStd` of daily
returns, so it is annualized the same way.

The two estimates mostly agree. They part when a bar's range is large
compared with its close-to-close move: on 10 October 2025 BTC fell 16% from
open to low but closed 7% down, and for the next 30 days the Yang-Zhang line,
which sees the low, sits well above the close-to-close line.

## Go deeper

- [`RollingYangZhangVol`](../../functions_rolling/RollingYangZhangVol.md): the formula and its warmup of `window_size + 1` bars.
- [`RollingGarmanKlassVol`](../../functions_rolling/RollingGarmanKlassVol.md), [`RollingParkinsonVol`](../../functions_rolling/RollingParkinsonVol.md), [`RollingRogersSatchellVol`](../../functions_rolling/RollingRogersSatchellVol.md): other OHLC estimators.
- [`EwYangZhangVol`](../../functions_ew/EwYangZhangVol.md): the same estimator with exponential weights.
