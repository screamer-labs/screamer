---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Handle missing values and warmup

Fill a gap with `Ffill` (the last value) or `FillNa` (a constant), and choose
with `start_policy` what a rolling function returns before its window is full.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import Ffill, FillNa, RollingMean

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
close = df[df["symbol"] == "BTC"].set_index("date")["close"].iloc[:30].copy()
close.iloc[[6, 7, 14]] = np.nan                  # three missing days

out = pd.DataFrame({"close": close})
out["ffill"] = Ffill()(close)                    # carry the last price forward
out["fillna"] = FillNa(-1.0)(close)              # mark the gap with a constant
out["mean"] = RollingMean(5)(close)              # NaN in warmup and at each gap
out["mean_ffill"] = RollingMean(5)(out["ffill"])
out["mean_early"] = RollingMean(5, start_policy="expanding")(out["ffill"])
out.head(16).round(0)
```

`RollingMean(5)` returns `NaN` for the first four days, while its window holds
fewer than five values; `start_policy="expanding"` averages the values seen so
far instead. A `NaN` input gives a `NaN` output and is left out of the window,
so the mean recovers on the next day.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(out.index, out["ffill"], color="0.6", ls="--", label="Ffill")
ax.plot(out.index, out["close"], "o-", color="0.2", ms=3,
        label="close with gaps")
ax.plot(out.index, out["mean_early"], color="steelblue",
        label='RollingMean(5, start_policy="expanding") after Ffill')
ax.set_ylabel("price (USD)")
ax.legend(loc="lower left", fontsize=8)
plt.tight_layout()
```

## Go deeper

- [NaN and warmup](../../nan_and_warmup.md): the NaN policy of each function and every `start_policy`.
- [NaN handling notebook](../../notebooks/05-nan-handling.ipynb): the three NaN policies on worked examples.
- [`Ffill`](../../functions_preprocessing/Ffill.md) and [`FillNa`](../../functions_preprocessing/FillNa.md): carry the last value or fill with a constant.
