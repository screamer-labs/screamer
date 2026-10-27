---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Handle missing values and warmup

Fill gaps with `FillNa` (a constant) or `Ffill` (the last value) before a
rolling function, and choose with `start_policy` what the function returns
before its window is full.

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
from screamer import Resample, FillNa, RollingMean

trades = pd.read_csv("../../notebooks/data/deribit_btc_perp_6h.csv")
# Traded volume per 10 seconds; a 10-second bucket without trades is NaN.
volume, ts = Resample(freq=10_000, agg="sum", fill="nan")(
    trades["volume"].abs().to_numpy(), trades["timestamp"].to_numpy())

df = pd.DataFrame({"volume": volume}, index=pd.to_datetime(ts, unit="ms"))
df["mean_raw"] = RollingMean(6)(df["volume"])       # NaN at each gap
df["filled"] = FillNa(0.0)(df["volume"])            # no trades, zero volume
df["mean_filled"] = RollingMean(6)(df["filled"])    # NaN in the first 5 rows
df["mean_early"] = RollingMean(6, start_policy="expanding")(df["filled"])
df.head(10).round(0)
```

`RollingMean` skips a `NaN` input: it returns `NaN` at that step and its window
keeps the last six finite values, which on the raw series can span several
minutes. After `FillNa`, the window always covers one minute. Warmup is the
first five rows, where the window holds fewer than six values: the default
`start_policy="strict"` returns `NaN` there, and `"expanding"` averages the
values seen so far.

```{code-cell} ipython3
part = df.iloc[60:240]  # minutes 10 to 40
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(part.index, part["mean_filled"], color="steelblue",
        label="RollingMean(6) after FillNa(0)")
ax.plot(part.index, part["mean_raw"], ".", color="darkorange", ms=4,
        label="RollingMean(6) on the raw series")
ax.set_ylabel("mean volume per 10 s")
ax.legend(loc="upper right", fontsize=8)
plt.tight_layout()
```

The raw mean averages only buckets that had trades, so it sits above the filled
mean and has no value at an empty bucket.

## Go deeper

- [NaN and warmup](../../nan_and_warmup.md): the NaN policy of each function and every `start_policy`.
- [NaN handling notebook](../../notebooks/05-nan-handling.ipynb): the three NaN policies on worked examples.
- [`FillNa`](../../functions_preprocessing/FillNa.md) and [`Ffill`](../../functions_preprocessing/Ffill.md): fill with a constant or carry the last value.
