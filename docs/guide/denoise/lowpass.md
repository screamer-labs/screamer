---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Remove fast wiggles with a low-pass filter

`Butter` is a Butterworth low-pass filter: it keeps the slow movements of a
series and removes those faster than a cutoff you set.

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
from screamer import Butter

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
close = df[df["symbol"] == "ETH"].set_index("date")["close"]

out = pd.DataFrame({"close": close})
out["order2"] = Butter(order=2, cutoff_freq=0.1)(close)
out["order4"] = Butter(order=4, cutoff_freq=0.1)(close)
out.head().round(0)
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(out.index, out["close"], color="0.6", lw=0.8, label="close")
ax.plot(out.index, out["order2"], color="steelblue",
        label="Butter(order=2, cutoff_freq=0.1)")
ax.plot(out.index, out["order4"], color="crimson",
        label="Butter(order=4, cutoff_freq=0.1)")
ax.set_ylabel("ETH price (USD)")
ax.legend(loc="upper left", fontsize=8)
plt.tight_layout()
```

`cutoff_freq` is a fraction of the Nyquist frequency, half a cycle per sample.
A cycle of `p` samples has the frequency `2 / p` in these units, so
`cutoff_freq=0.1` keeps cycles longer than 20 days and removes shorter ones.
A higher `order` cuts more sharply at the cutoff and delays the output more.
Both filters start at the first close: the filter state starts as if the
series had been at that price before the first row.

## Go deeper

- [`Butter`](../../functions_signal/Butter.md): the design, and `start_policy="zero"` to match `scipy.signal.butter` with `lfilter`.
- [`ButterHighpass`](../../functions_signal/ButterHighpass.md) and [`ButterBandpass`](../../functions_signal/ButterBandpass.md): keep the fast part, or one band of cycle lengths.
- [Signal processing notebook](../../notebooks/04-signal-processing.ipynb): filters compared on the same series.
