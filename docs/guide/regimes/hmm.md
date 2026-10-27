---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Estimate the probability of a volatile regime

`RollingHMM` fits a two-state hidden Markov model to a trailing window of
returns, one calm state and one volatile state, and returns at each step the
probability that the series is in the volatile state.

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
from screamer import LogReturn, RollingHMM

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
df = df[df["symbol"] == "BTC"].set_index("date")

df["ret"] = LogReturn()(df["close"])
df["prob"] = RollingHMM(365)(df["ret"])[:, 0]

# Scoring only, looks ahead: the size of the next day's move in each regime.
df["next_move"] = df["ret"].abs().shift(-1)
fitted = df.dropna(subset=["prob"])
volatile = fitted["prob"] > 0.5
regime = volatile.map({False: "calm", True: "volatile"}).rename("regime")
fitted.groupby(regime)["next_move"].agg(["count", "mean"]).round(4)
```

Each state emits returns from its own normal distribution. The model is fit
on the last 365 days, first when the window fills and then every 24 days, and
between fits each new return updates the probability. Column 0 is that
probability; columns 1 and 2 are the mean and standard deviation of the next
return. The last lines look one day ahead to score the regimes and are not
part of the model, which uses only past data. A day is volatile when its
probability is above 0.5. After a volatile day the next absolute return
averaged 2.0%, against 1.5% after a calm day, so the regime carries over to
the next day.

```{code-cell} ipython3
part = df.loc["2025"]
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 4.5), sharex=True)
ax0.bar(part.index, part["ret"], width=1, color="0.5")
ax0.set_ylabel("daily log return")
ax1.plot(part.index, part["prob"], color="crimson", lw=0.8)
ax1.axhline(0.5, color="0.5", ls="--", lw=0.8)
ax1.set_ylabel("P(volatile)")
plt.tight_layout()
```

The probability jumps towards 1 on a large move and falls back within a day
or two, so on BTC daily returns the volatile state is short.

## Go deeper

- [`RollingHMM`](../../functions_rolling/RollingHMM.md): the fit, the state ordering, and `emission="ar1"` for a trending and a reverting state.
- [Detect a change in mean or spread](changepoint.md): a change detector instead of two fixed states.
- [Measure volatility](../volatility/index.md): the volatility itself.
