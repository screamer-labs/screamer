---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Remove spikes from a series

`Hampel` replaces a value with the median of its trailing window when it lies
more than `n_sigma` robust standard deviations from that median.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import Hampel

trades = pd.read_csv("../../notebooks/data/deribit_btc_perp_6h.csv")
price = trades["price"].copy()
bad = [500, 1500, 2600, 3300]                    # four bad prints, about 1% off
price.iloc[bad] *= [1.010, 0.990, 1.008, 0.992]

clean = Hampel(window_size=300, n_sigma=5.0)(price)
flag = Hampel(window_size=300, n_sigma=5.0, output="flag")(price)
print("flagged trades:", np.flatnonzero(flag == 1).tolist())
```

The robust standard deviation is 1.4826 times the median absolute deviation
(MAD) of the window, so the spikes themselves do not widen the threshold. Trade
prices repeat often, which makes the MAD of a short window close to zero and
flags ordinary ticks; a window of a few hundred trades gives a stable scale.
The output is `NaN` until the window is full.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(price.index, price, color="0.6", lw=0.8, label="trade price with spikes")
ax.plot(price.index, clean, color="steelblue",
        label="Hampel(window_size=300, n_sigma=5)")
ax.set_xlabel("trade number")
ax.set_ylabel("BTC perpetual price (USD)")
ax.legend(loc="lower left", fontsize=8)
plt.tight_layout()
```

## Go deeper

- [`Hampel`](../../functions_rolling/Hampel.md): the three `output` modes and the warmup.
- [`ImpulseClip`](../../functions_rolling/ImpulseClip.md): despike a trending series on its first difference.
- [`KalmanFilter`](../../functions_signal/KalmanFilter.md): its `huber` option limits the pull of outliers while smoothing.
