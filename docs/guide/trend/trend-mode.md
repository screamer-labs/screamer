---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Tell trending periods from cycling periods

`TrendMode` returns 1.0 while a price series trends and 0.0 while it moves in
cycles, with John Ehlers' rule as in TA-Lib's `HT_TRENDMODE`.

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
from screamer import TrendMode

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
close = df[df["symbol"] == "BTC"].set_index("date")["close"]

out = pd.DataFrame({"close": close})
out["mode"] = TrendMode()(close)
out["mode"].loc["2025"].resample("QE").mean().round(2)
```

`TrendMode` measures the dominant cycle of the series and reads cycle for half
a period after each cycle turn, and while the cycle advances at its own rate.
It reads trend when the smoothed price is at least 1.5% from the cycle's
trendline, so use it on prices: on returns, which stay near zero, every sample
reads trend. The table is the share of trending days per quarter of 2025.

```{code-cell} ipython3
part = out.loc["2025"]
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(part.index, part["close"], color="0.2", lw=1, label="close")
ax.fill_between(part.index, 0, 1, where=part["mode"] == 0, color="darkorange",
                alpha=0.25, transform=ax.get_xaxis_transform(),
                label="TrendMode = 0 (cycling)")
ax.set_ylabel("BTC price (USD)")
ax.legend(loc="upper left", fontsize=8)
plt.tight_layout()
```

June has the most cycling days of 2025, while the price moved sideways
between about 101,000 and 110,000 USD.

## Go deeper

- [`TrendMode`](../../functions_signal/TrendMode.md): the three rules and why a large cycle reads trend near its peaks.
- [`DominantCycle`](../../functions_signal/DominantCycle.md): the cycle period that `TrendMode` builds on.
- [Cycle and spectral analysis notebook](../../notebooks/19-cycle-and-spectral-analysis.ipynb): Ehlers' cycle tools on one series.
