---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Measure trend strength with ADX

`ADX` measures how strongly a series trends, whatever its direction, and
returns it together with `+DI` and `-DI`, which tell the direction: up while
`+DI` is above `-DI`, down while it is below.

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
from screamer import ADX, PlusDI

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
df = df[df["symbol"] == "BTC"].set_index("date")

dmi = ADX(14)(df["high"], df["low"], df["close"])
df["plus_di"], df["minus_di"], df["adx"] = dmi[:, 0], dmi[:, 1], dmi[:, 2]
df["plus_di_alone"] = PlusDI(14)(df["high"], df["low"], df["close"])
df[["plus_di", "plus_di_alone", "minus_di", "adx"]].tail().round(1)
```

The inputs come in the order high, low, close, and the three output columns
are `+DI`, `-DI` and `ADX`. `PlusDI` computes `+DI` alone and gives the same
values, as the table shows; `MinusDI` does the same for `-DI`. `+DI` and `-DI` are
`NaN` for the first 14 bars and `ADX` for the first 27, and all three match
TA-Lib.

```{code-cell} ipython3
part = df.loc["2024":"2025"]
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(part.index, part["plus_di"], color="seagreen", lw=0.8, label="+DI")
ax.plot(part.index, part["minus_di"], color="crimson", lw=0.8, label="-DI")
ax.plot(part.index, part["adx"], color="0.1", lw=1.5, label="ADX(14)")
ax.axhline(25, color="0.5", ls="--", lw=0.8)
ax.set_ylabel("BTC directional index")
ax.legend(loc="upper left", fontsize=8)
plt.tight_layout()
```

A rule of thumb that goes back to Wilder (1978) reads `ADX` above 25 (the
dashed line) as a trending market and below 20 as a market without a trend.

## Go deeper

- [`ADX`](../../functions_rolling/ADX.md): the formulas, the warmup of each output, and the TA-Lib seeding.
- [`PlusDI`](../../functions_rolling/PlusDI.md), [`MinusDI`](../../functions_rolling/MinusDI.md), [`ADXR`](../../functions_rolling/ADXR.md): the directional indicators alone, and a smoothed ADX.
- [Financial indicators notebook](../../notebooks/03-financial-indicators.ipynb): ADX next to other indicators.
