---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Compare each day with the same weekday in past weeks (seasonality, seasonal z-score)

`RollingSeasonalZscore` standardizes each value against the same slot of past
cycles, such as the same hour on past days or the same weekday in past weeks,
so a recurring pattern does not look unusual.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import RollingZscore, RollingSeasonalZscore

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
df = df[df["symbol"] == "BTC"].set_index("date")
log_range = np.log(np.log(df["high"] / df["low"]))   # log of the daily range

df["z_plain"] = RollingZscore(84)(log_range)   # against the last 84 days
df["z_seasonal"] = RollingSeasonalZscore(period=7, window_size=12)(log_range)

weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
            "Saturday", "Sunday"]
by_day = df.groupby(df.index.day_name())[["z_plain", "z_seasonal"]].mean()
saturday = df.loc[df.index.dayofweek == 5, "z_plain"].dropna()
print(f"Saturdays with plain z below 0: {(saturday < 0).mean():.0%}, "
      f"below -1: {(saturday < -1).mean():.0%}")
by_day.reindex(weekdays).round(2).T
```

```{code-cell} ipython3
by_day = by_day.reindex(weekdays)
fig, ax = plt.subplots(figsize=(8, 3.5))
x = np.arange(7)
ax.bar(x - 0.2, by_day["z_plain"], width=0.4, color="0.6",
       label="RollingZscore(84)")
ax.bar(x + 0.2, by_day["z_seasonal"], width=0.4, color="steelblue",
       label="RollingSeasonalZscore(period=7, window_size=12)")
ax.axhline(0, color="k", lw=0.5)
ax.set_xticks(x, weekdays)
ax.set_ylabel("mean z-score of BTC log range")
ax.legend(loc="lower left", fontsize=8)
plt.tight_layout()
```

The committed data are daily, so this page uses the weekly cycle of seven daily
slots. The daily range, high over low, is a one-day volatility estimate, and
BTC moves less at weekends: the median Saturday range is well below that of a
weekday. An ordinary z-score of the log range over the last 84 days
is below 0 on 89% of Saturdays and below -1 on 60%, with a mean of -1.17. The
seasonal z-score compares a Saturday with the last 12 Saturdays, so its mean
is near 0 on every weekday, and a high value means a volatile day for that
weekday.

Step `t` belongs to slot `t mod period`, counted from the first value, so the
input must be on a regular clock with no missing rows; a `NaN` keeps a missing
step in its slot. For an hour-of-day pattern, pass hourly values with
`period=24`: each hour is then compared with the same hour on the last
`window_size` days.

## Go deeper

- [`RollingSeasonalZscore`](../../functions_rolling/RollingSeasonalZscore.md): the slot clock, missing values and warmup.
- [`RollingSeasonalMean`](../../functions_rolling/RollingSeasonalMean.md), [`RollingSeasonalStd`](../../functions_rolling/RollingSeasonalStd.md): the mean and standard deviation of each slot alone.
- [Build bars from trades](../get-data-in/bars-from-ticks.md): hourly bars on a regular clock from a trade tape.
