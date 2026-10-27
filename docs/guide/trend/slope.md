---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Measure a trend with a rolling slope

`RollingPoly1(derivative_order=1)` fits a straight line to each trailing
window and returns its slope; `Sign` of the slope gives the direction of the
trend, +1 up and -1 down.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import RollingPoly1, Sign

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
close = df[df["symbol"] == "BTC"].set_index("date")["close"]

out = pd.DataFrame({"close": close})
out["slope"] = RollingPoly1(50, derivative_order=1)(np.log(close))
out["trend"] = Sign()(out["slope"])
out.tail().round(4)
```

```{code-cell} ipython3
part = out.loc["2024":"2025"]
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(part.index, part["close"], color="0.2", lw=1, label="close")
ax.fill_between(part.index, 0, 1, where=part["trend"] > 0, color="seagreen",
                alpha=0.15, transform=ax.get_xaxis_transform(),
                label="slope of the 50-day fit > 0")
ax.set_ylabel("BTC price (USD)")
ax.legend(loc="upper left", fontsize=8)
plt.tight_layout()
```

The fit runs on the log price, so the slope is a growth rate: 0.002 means the
line rises about 0.2% per day. `RollingPoly1(50)` is `NaN` until it has 50
values. A longer window gives a steadier trend that turns later.

The shaded stretches are where the 50-day line slopes up. The trend turns
some days after the price does, since the line is fitted to the whole window.

## Go deeper

- [`RollingPoly1`](../../functions_rolling/RollingPoly1.md), [`RollingPoly2`](../../functions_rolling/RollingPoly2.md): straight-line and quadratic fits; `derivative_order=0` gives the fitted value.
- [`RollingLinearRegression`](../../functions_fin/RollingLinearRegression.md): slope, intercept and forecast together.
- [Financial indicators notebook](../../notebooks/03-financial-indicators.ipynb): more trend indicators on one price series.
