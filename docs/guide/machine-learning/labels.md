---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Label each bar with its future move

`Sign` of `LogReturn(h)` labels each bar with the direction of the next `h`
bars, and `TripleBarrierOHLC` labels it by which of two price levels the next
`h` bars reach first.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import LogReturn, Sign, TripleBarrierOHLC

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
btc = df[df["symbol"] == "BTC"].set_index("date")
o, h, l, c = (btc[k].to_numpy() for k in ["open", "high", "low", "close"])

horizon = 10
direction = Sign()(LogReturn(horizon)(c))
barrier = TripleBarrierOHLC(horizon=horizon, upper=1.0, lower=1.0,
                            barrier_unit="vol", vol_span=20)(o, h, l, c)

# Row t of each output is the label of the bar at t - horizon
fixed, label = direction[horizon:], barrier[horizon:, 0]
for name, y in [("fixed horizon", fixed), ("triple barrier", label)]:
    values, counts = np.unique(y[~np.isnan(y)], return_counts=True)
    print(f"{name:15}", dict(zip(values.astype(int).tolist(), counts.tolist())))
```

A label describes the price after a bar, so it looks ahead by design; screamer
stamps it at the bar where it becomes known, `horizon` bars after the bar it
labels, and `[horizon:]` moves it back. The fixed-horizon label is `+1` when
daily BTC closes higher 10 days later and `-1` when it closes lower. The
triple-barrier label is `+1` when the price first reaches one expected
deviation above the close, `-1` when it first reaches one below, and `0` when
it reaches neither within 10 days. The expected deviation is the 20-day
exponential standard deviation of daily log returns, times the square root
of 10. The output's other two columns hold the return to the exit and the
days held.

```{code-cell} ipython3
dates = btc.index[:len(label)]
part = (dates >= "2024-01-01") & (dates < "2024-07-01")
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(dates[part], c[:len(label)][part], color="0.4", lw=0.8)
for value, color in [(1, "tab:green"), (-1, "tab:red"), (0, "0.7")]:
    at = part & (label == value)
    ax.scatter(dates[at], c[:len(label)][at], color=color, s=12, zorder=3,
               label=f"{value:+d}" if value else "0")
ax.set_ylabel("BTC close (USD)")
ax.legend(title="triple-barrier label", fontsize=8)
plt.tight_layout()
```

## Go deeper

- [Pair features with labels without look-ahead](training-set.md): put each label next to the features of its bar.
- [Labels and training sets](../../labels.md): targets for size, volatility, range, and cross-sectional rank.
- [Triple-barrier labels](../../notebooks/24-triple-barrier-labels.ipynb): set the barriers from the price excursions, holding times, and ambiguous bars.
- [Meta-labeling](../../notebooks/25-meta-labeling.ipynb): use the labels to decide which trades of a rule to take.
- [`TripleBarrierOHLC`](../../functions_fin/TripleBarrierOHLC.md): barrier units and the order of tests within a bar.
