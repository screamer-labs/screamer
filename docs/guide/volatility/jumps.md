---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Separate jumps from normal volatility (bipower variation)

`RollingBipowerVar` estimates the variance of a window of returns without its
jumps, and `RollingJumpShare` gives the share of the realized variance that the
jumps contribute.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import LogReturn, RollingSum, RollingBipowerVar, RollingJumpShare

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
df = df[df["symbol"] == "BTC"].set_index("date")
r = LogReturn()(df["close"])

n = 60                                        # days per window
df["vol_all"] = np.sqrt(RollingSum(n)(r**2) * 365 / n)   # realized variance
df["vol_no_jumps"] = np.sqrt(RollingBipowerVar(n)(r) * 365 / n)
df["jump_share"] = RollingJumpShare(n)(r)
zero = df.loc["2022-11-10":"2023-01-31", "jump_share"] == 0
print("jump share 0 from 2022-11-10, first nonzero:", zero.idxmin().date())
df.loc["2023-10-20":"2023-10-25", ["close", "vol_all", "vol_no_jumps", "jump_share"]].round(3)
```

```{code-cell} ipython3
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 5), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
ax0.plot(df.index, df["vol_all"], color="steelblue",
         label="from the realized variance")
ax0.plot(df.index, df["vol_no_jumps"], color="darkorange",
         label="from the bipower variation")
ax0.set_ylabel("BTC annualized vol")
ax0.legend(loc="upper right", fontsize=8)
ax1.plot(df.index, df["jump_share"], color="0.3")
ax1.set_ylabel("jump share")
plt.tight_layout()
```

The realized variance is the sum of squared returns, so a jump counts at its
full size squared. The bipower variation (Barndorff-Nielsen and Shephard 2004)
sums the products of adjacent absolute returns instead. A jump between two
ordinary days is multiplied by their small returns, so it adds little.
Annualized like the realized variance, it is a volatility without the jumps.
`RollingJumpShare` is `1 - BV / RV`, from 0 to 1.

The highest share in these data, 0.54, comes on 23 October 2023, when BTC rose
10% in one day in a quiet market, with an annualized volatility near 33%. Large
moves on consecutive days count as volatility, not as jumps: after the FTX
collapse in November 2022, with returns of -10%, -14% and +10% in a row, the
share is 0 for the next two months, until early January 2023. The estimator is
usually applied to the intraday returns of each day; it works the same on
returns at any regular interval.

## Go deeper

- [`RollingBipowerVar`](../../functions_rolling/RollingBipowerVar.md), [`RollingJumpShare`](../../functions_rolling/RollingJumpShare.md): the formulas and the small-sample scaling.
- [Measure volatility from closing prices](close-to-close.md): the standard deviation of returns.
- [Detect outliers robustly](../denoise/robust-outliers.md): flag the single days that cause the jumps.
