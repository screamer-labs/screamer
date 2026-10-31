---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Z-score, demean and winsorize across assets each day

`CrossDemean`, `CrossZscore` and `CrossWinsorize` compare each asset with the
others on the same day: they subtract the day's mean, divide by the day's
standard deviation (`CrossStd`), and clip the day's extreme values, so scores
from different days share one scale.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import (CrossDemean, CrossStd, CrossWinsorize, CrossZscore,
                      LogReturn, RollingSum)

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
close = df.pivot(index="date", columns="symbol", values="close")
score = RollingSum(30)(LogReturn()(close.to_numpy()))   # (T, K) 30-day return

demeaned = CrossDemean()(score)
dispersion = CrossStd()(score)                          # (T,), one per day
z = CrossZscore()(score)                                # demeaned / dispersion
z_clipped = CrossZscore()(CrossWinsorize(0.1, 0.9)(score))

day = np.nanargmax(np.abs(z)) // z.shape[1]             # most extreme day
print(f"{close.index[day]:%Y-%m-%d}, dispersion {dispersion[day]:.2f}")
table = pd.DataFrame({"score": score[day], "demeaned": demeaned[day],
                      "z": z[day], "winsorized z": z_clipped[day]},
                     index=close.columns).sort_values("z")
table.iloc[[0, 1, -3, -2, -1]].round(2)
```

```{code-cell} ipython3
order = np.argsort(z[day])
x = np.arange(len(order))
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.bar(x - 0.2, z[day][order], 0.4, color="0.6", label="CrossZscore")
ax.bar(x + 0.2, z_clipped[day][order], 0.4, color="steelblue",
       label="CrossZscore after CrossWinsorize(0.1, 0.9)")
ax.set_xticks(x, close.columns[order], rotation=90, fontsize=8)
ax.set_ylabel(f"z-score on {close.index[day]:%Y-%m-%d}")
ax.legend(loc="upper left", fontsize=8)
plt.tight_layout()
```

The data is daily closes of 20 Binance spot pairs quoted in USDT, pivoted to a
`(T, K)` array with one asset per column. The score is each coin's log return
over the last 30 days. The cross-sectional functions read each row, one day,
on its own and skip `NaN` values, so a coin without a score that day is left
out of the mean and the standard deviation.

`CrossDemean` subtracts the day's mean score, which removes the move all coins
shared, and `CrossStd` is the day's sample standard deviation of the scores
(`ddof=1`). `CrossZscore` divides the first by the second. The table shows the
lowest two and highest three coins on the day with the largest z-score in the
sample, 13 November 2024, when DOGE's 30-day log return of 1.23 was 3.86
standard deviations above the mean of the 20 coins. One coin that far out
shrinks every other coin's z-score, because it alone inflates the standard
deviation. `CrossWinsorize(0.1, 0.9)` first clips each score to the day's 10%
and 90% quantiles, so DOGE and ADA both sit at the 90% quantile and the other
coins spread over a wider range of z-scores.

## Go deeper

- [`CrossZscore`](../../functions_cross/CrossZscore.md), [`CrossDemean`](../../functions_cross/CrossDemean.md), [`CrossWinsorize`](../../functions_cross/CrossWinsorize.md), [`CrossStd`](../../functions_cross/CrossStd.md): the row rules and the `NaN` cases.
- [`CrossRank`](../../functions_cross/CrossRank.md): ranks instead of z-scores, which no single value can dominate.
- [Rank assets by momentum](../strategies/cross-sectional-momentum.md): a long-short book from cross-sectional scores.
- [Backtest a portfolio of many assets](multi-asset.md): the `(T, K)` shape the engines take.
