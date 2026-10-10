---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Indicators on a pandas OHLCV DataFrame

screamer functions take NumPy arrays or pandas columns and return NumPy arrays,
so adding indicators to an OHLCV DataFrame is one assignment per indicator. This
recipe builds one-minute bars from six hours of Deribit BTC-perpetual trades and
adds Bollinger Bands, RSI, and ATR as new columns.

## The DataFrame

If you already have an OHLCV DataFrame, skip this cell. Here `Resample` builds
one from the committed trade tape: `agg="ohlcv"` turns `[price, size]` rows into
open, high, low, close, and volume per one-minute bucket, and `to_pandas` names
the columns. A minute with no trades produces no bar.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import Resample, to_pandas, BollingerBands, RollingRSI, ATR

trades = pd.read_csv("../notebooks/data/deribit_btc_perp_6h.csv")
values, index = Resample(freq=60_000, agg="ohlcv")(
    np.column_stack([trades["price"], trades["volume"].abs()]),
    trades["timestamp"].to_numpy())

df = to_pandas(values, index, columns=["open", "high", "low", "close", "volume"])
df.index = pd.to_datetime(df.index, unit="ms")
df.head()
```

## Adding indicator columns

A single-output function returns one column. A multi-output function returns a
`(T, k)` array, which fills `k` columns in one assignment. `ATR` takes three
inputs, passed in the order its page documents: high, low, close.

```{code-cell} ipython3
df[["bb_lower", "bb_mid", "bb_upper"]] = BollingerBands(20, 2.0)(df["close"])
df["rsi"] = RollingRSI(14)(df["close"])
df["atr"] = ATR(14)(df["high"], df["low"], df["close"])

df.iloc[25:30, 3:].round(2)
```

The first rows of each indicator are `NaN` until its window has filled. The
[NaN and warmup](../nan_and_warmup.md) guide explains the rule and the
`start_policy` options.

```{code-cell} ipython3
fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(9, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1, 1]})
ax0.fill_between(df.index, df["bb_lower"], df["bb_upper"], color="steelblue", alpha=0.15,
                 label="Bollinger Bands (20, 2)")
ax0.plot(df.index, df["bb_mid"], color="steelblue", lw=0.8)
ax0.plot(df.index, df["close"], color="0.2", lw=0.9, label="close")
ax0.set_ylabel("price (USD)")
ax0.legend(loc="upper left", fontsize=8)
ax1.plot(df.index, df["rsi"], color="darkorange")
ax1.axhline(70, color="0.5", lw=0.6, ls="--")
ax1.axhline(30, color="0.5", lw=0.6, ls="--")
ax1.set_ylabel("RSI(14)")
ax2.plot(df.index, df["atr"], color="seagreen")
ax2.set_ylabel("ATR(14)")
plt.tight_layout()
```

Related pages: [`BollingerBands`](../functions_rolling/BollingerBands.md),
[`RollingRSI`](../functions_rolling/RollingRSI.md),
[`ATR`](../functions_rolling/ATR.md), [`Resample`](../functions_streams/Resample.md),
and the [financial indicators notebook](../notebooks/03-financial-indicators.ipynb).
