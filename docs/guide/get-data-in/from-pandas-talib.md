---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Replace pandas rolling and TA-Lib calls

Construct the screamer function with the same window or span and call it on the
data: `s.rolling(20).mean()` becomes `RollingMean(20)(s)` and
`talib.RSI(c, 14)` becomes `RollingRSI(14)(c)`.

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
from screamer import RollingMean, RollingStd, EwMean, RollingRSI

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
close = df[df["symbol"] == "BTC"].set_index("date")["close"]

out = pd.DataFrame(index=close.index)
out["sma"] = RollingMean(20)(close)    # close.rolling(20).mean(), talib.SMA
out["std"] = RollingStd(20)(close)     # close.rolling(20).std()
out["ema"] = EwMean(span=10)(close)    # close.ewm(span=10).mean()
out["rsi"] = RollingRSI(14)(close)     # talib.RSI(c, 14), Wilder smoothing
# RSI as built from diff() and rolling(14).mean() in pandas:
out["rsi_cutler"] = RollingRSI(14, method="cutler")(close)
out.tail().round(1)
```

```{code-cell} ipython3
year = out.loc["2025"]
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 4.5), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
ax0.plot(year.index, year["rsi"], color="steelblue",
         label="RollingRSI(14), Wilder (TA-Lib)")
ax0.plot(year.index, year["rsi_cutler"], color="darkorange", lw=0.9,
         label='method="cutler" (pandas rolling)')
ax0.set_ylabel("RSI")
ax0.legend(fontsize=8, loc="lower left")
ax1.plot(year.index, year["rsi_cutler"] - year["rsi"], color="0.3")
ax1.axhline(0, color="k", lw=0.5)
ax1.set_ylabel("difference")
plt.tight_layout()
```

RSI is the most common surprise when porting a strategy. TA-Lib and most
charting platforms use Wilder's smoothing, which `RollingRSI` uses by default.
An RSI built from `diff()` and `rolling(14).mean()` in pandas is Cutler's
variant, `method="cutler"`. On daily BTC closes in 2025 the two differ by up to
20 points.

## Go deeper

- [From pandas](#from-pandas) and [From TA-Lib](#from-ta-lib), below: the screamer call for each common pandas and TA-Lib call.
- [Conventions](../../conventions.md): where a definition differs from pandas or TA-Lib, and why.
- [`RollingRSI`](../../functions_rolling/RollingRSI.md): the Wilder and Cutler methods.
- [Compute indicators on an OHLCV DataFrame](ohlcv-dataframe.md): several indicators as new columns.

### From pandas

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

Each call returns a NumPy array. Assign it to a column to keep the pandas index,
or wrap it with `pd.Series(result, index=df.index)`.

### From TA-Lib

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

TA-Lib returns a tuple for a function with several outputs; screamer returns one
`(T, k)` array whose columns are listed above.
