---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Detect a trend with a moving-average crossover

Two `EwMean` averages of different lengths cross when the trend changes;
`Sign` of the fast average minus the slow one is +1 while the fast average is
above (an uptrend) and -1 while it is below.

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
from screamer import EwMean, Sign

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
close = df[df["symbol"] == "BTC"].set_index("date")["close"]

out = pd.DataFrame({"close": close})
out["fast"] = EwMean(span=20)(close)
out["slow"] = EwMean(span=100)(close)
out["trend"] = Sign()(out["fast"] - out["slow"])
out.tail().round(0)
```

`EwMean` gives a value from the first sample on, so the early values of the
slow average rest on few prices; leave out the first 100 days or so before
trading on the crossover.

```{code-cell} ipython3
part = out.loc["2024":"2025"]
print("state changes:", int((part["trend"].diff().abs() > 0).sum()))
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(part.index, part["close"], color="0.6", lw=0.8, label="close")
ax.plot(part.index, part["fast"], color="crimson", label="EwMean(span=20)")
ax.plot(part.index, part["slow"], color="steelblue", label="EwMean(span=100)")
ax.fill_between(part.index, 0, 1, where=part["trend"] > 0, color="seagreen",
                alpha=0.15, transform=ax.get_xaxis_transform(),
                label="fast above slow")
ax.set_ylabel("BTC price (USD)")
ax.legend(loc="upper left", fontsize=8)
plt.tight_layout()
```

In 2024 and 2025 the crossover changes state 7 times, against 17 times for the
sign of the 50-day slope on the [rolling slope page](slope.md).

## Go deeper

- [`EwMean`](../../functions_ew/EwMean.md): the weights, and how `span`, `halflife` and `alpha` relate.
- [Smooth a series with less lag](../denoise/low-lag.md): averages that turn sooner, for a faster crossover.
- [Measure a trend with a rolling slope](slope.md): direction from a fitted line.
