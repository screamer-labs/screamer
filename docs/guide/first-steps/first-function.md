---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Apply a function to an array, a column and one value at a time

Construct a function with its settings, such as `RollingMean(50)` for a 50-step
moving average, then call it on a NumPy array, a pandas column, or one value at
a time.

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
from screamer import RollingMean

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
btc = df[df["symbol"] == "BTC"].set_index("date")

# A NumPy array in, a NumPy array of the same length out.
ma_array = RollingMean(50)(btc["close"].to_numpy())

# A pandas column works the same way; assign the result to a new column.
btc["ma50"] = RollingMean(50)(btc["close"])

# One value at a time, as on a live feed; the object keeps its window.
live = RollingMean(50)
for price in btc["close"]:
    latest = live(price)

print(btc["ma50"].tail(3))
print("latest value from the loop:", latest)
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(btc.index, btc["close"], color="0.6", lw=0.8, label="BTC close")
ax.plot(btc.index, btc["ma50"], color="steelblue", label="RollingMean(50)")
ax.set_ylabel("price (USD)")
ax.legend(loc="upper left", fontsize=8)
plt.tight_layout()
```

The array call returns a NumPy array, the column call a pandas Series with the
dates as its index, and the loop the latest average after each price. The first
49 values are `NaN`, because the window holds fewer than 50 prices until then.

## Go deeper

- [Quickstart notebook](../../notebooks/01-quickstart-polymorphic-api.ipynb): every input shape a function accepts.
- [Using screamer](../../usage.md): construct, call, chain, and reset a function.
- [`RollingMean`](../../functions_rolling/RollingMean.md): parameters and warmup options.
