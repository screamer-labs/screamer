---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Migrating from pandas rolling and TA-Lib

Most pandas `rolling`/`ewm` expressions and TA-Lib calls have a direct screamer
equivalent: construct the function with its window or span, then call it on the
data. The tables below map the common ones. Where the definitions differ, the
[Conventions](../conventions.md) page gives the details.

## From pandas

| pandas | screamer |
|---|---|
| `s.rolling(20).mean()` | `RollingMean(20)(x)` |
| `s.rolling(20).std()` | `RollingStd(20)(x)` (sample std, `ddof=1`, as in pandas) |
| `s.rolling(20, min_periods=1).min()` / `.max()` | `RollingMin(20)(x)` / `RollingMax(20)(x)` (output from the first sample) |
| `s.rolling(20, min_periods=1).median()` | `RollingMedian(20)(x)` (output from the first sample) |
| `s.rolling(20).quantile(0.9)` | `RollingQuantile(20, 0.9)(x)` |
| `s.rolling(20).corr(t)` | `RollingCorr(20)(x, y)` |
| `s.rolling(20, min_periods=1).mean()` | `RollingMean(20, start_policy="expanding")(x)` |
| `s.ewm(span=10).mean()` | `EwMean(span=10)(x)` (the `adjust=True` form) |
| `s.expanding().mean()` | `ExpandingMean()(x)` |
| `s.diff()` / `s.shift(1)` | `Diff()(x)` / `Lag(1)(x)` |
| `s.pct_change()` | `Return()(x)` |
| `s.ffill()` | `Ffill()(x)` |
| `df.rolling(20).mean()` on `K` columns | `RollingMean(20)(df.to_numpy())`, shape `(T, K)` |

Each screamer call returns a NumPy array. To keep the pandas index, assign the
result to a column, `df["ma"] = RollingMean(20)(df["close"])`, or wrap it with
`pd.Series(result, index=df.index)`. A screamer function also accepts one value
at a time, so the object you test on a DataFrame column can run on a live feed.

## From TA-Lib

| TA-Lib | screamer |
|---|---|
| `talib.SMA(c, 20)` | `RollingMean(20)(c)` |
| `talib.EMA(c, 10)` | `EwMean(span=10)(c)` (bias-corrected; differs in the first samples) |
| `talib.WMA(c, 20)` | `WMA(20)(c)` |
| `talib.RSI(c, 14)` | `RollingRSI(14)(c)` (Wilder smoothing) |
| `talib.MACD(c)` | `MACD()(c)`, columns: line, signal, histogram (built on `EwMean`; differs in the first samples) |
| `talib.BBANDS(c, 20)` | `BollingerBands(20, 2.0)(c)`, columns: lower, middle, upper (bands use the sample std; TA-Lib uses the population std) |
| `talib.ATR(h, l, c, 14)` | `ATR(14)(h, l, c)` |
| `talib.STOCH(h, l, c, 14, 3, 0, 3, 0)` | `Stoch(14, 3, 3)(h, l, c)`, columns: %K, %D |
| `talib.KAMA(c, 30)` | `KAMA(30)(c)` |

TA-Lib returns a tuple for a multi-output indicator; screamer returns one
`(T, k)` array whose columns are listed above.

## RSI: Wilder or Cutler

The most common surprise when porting a strategy is RSI. TA-Lib and most
charting platforms use Wilder's smoothing, which `RollingRSI` uses by default.
An RSI built from `diff()` and `rolling(14).mean()` in pandas is Cutler's
variant, available as `method="cutler"`. On one-minute BTC closes the two differ
by up to tens of points.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import Resample, RollingRSI

trades = pd.read_csv("../notebooks/data/deribit_btc_perp_6h.csv")
close, ts = Resample(freq=60_000, agg="last")(
    trades["price"].to_numpy(), trades["timestamp"].to_numpy())
t = pd.to_datetime(ts, unit="ms")

wilder = RollingRSI(14)(close)
cutler = RollingRSI(14, method="cutler")(close)

fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 4.5), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
ax0.plot(t, wilder, color="steelblue", label='RollingRSI(14), Wilder (TA-Lib)')
ax0.plot(t, cutler, color="darkorange", lw=0.9, label='method="cutler" (pandas rolling)')
ax0.set_ylabel("RSI")
ax0.legend(fontsize=8, loc="lower left")
ax1.plot(t, cutler - wilder, color="0.3")
ax1.axhline(0, color="k", lw=0.5)
ax1.set_ylabel("difference")
plt.tight_layout()
```
