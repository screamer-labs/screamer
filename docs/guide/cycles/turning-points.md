---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Mark cycle turning points

`CycleSine` returns John Ehlers' sinewave indicator, two lines called `sine`
and `leadsine`, as in TA-Lib's `HT_SINE`. While a series cycles, the two
lines cross just before each peak and trough of the cycle.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import CycleSine, Diff, Sign

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
df = df[df["symbol"] == "BTC"].set_index("date")

sine = CycleSine()(df["close"])
df["sine"], df["leadsine"] = sine[:, 0], sine[:, 1]
# -2 at a peak (leadsine falls below sine), +2 at a trough.
df["turn"] = Diff()(Sign()(df["leadsine"] - df["sine"]))

# After the fact: how often the 10 days after a mark moved the expected way.
ahead = np.log(df["close"]).shift(-10) - np.log(df["close"])
peak, trough = df["turn"] < 0, df["turn"] > 0
pd.DataFrame({"marks": [peak.sum(), trough.sum()],
              "moved as marked": [(ahead[peak] < 0).mean(), (ahead[trough] > 0).mean()]},
             index=["peak", "trough"]).round(2)
```

`sine` is the sine of the phase of the dominant cycle, and `leadsine` is the
same curve one eighth of a cycle ahead, so `leadsine` turns first. Both are
`NaN` for the first 40 days. The code after the second comment looks ahead to score the marks
and are not part of the indicator. Over 2022 to 2026 there are 54 marked
peaks and 54 troughs; 48% of the peaks were followed by a lower price ten
days later, and 52% of the troughs by a higher one.

```{code-cell} ipython3
part = df.loc["2025"]
peaks, troughs = part[part["turn"] < 0], part[part["turn"] > 0]
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(part.index, part["close"], color="0.2", lw=1, label="close")
ax.scatter(peaks.index, peaks["close"], marker="v", color="crimson",
           zorder=3, label="cycle peak")
ax.scatter(troughs.index, troughs["close"], marker="^", color="seagreen",
           zorder=3, label="cycle trough")
ax.set_ylabel("BTC price (USD)")
ax.legend(loc="lower left", fontsize=8)
plt.tight_layout()
```

Some marks sit at turns of the price, such as the peak on 5 October 2025, a
day before the highest close of the year, and others fall inside moves. The
marks are turns of the measured cycle, which match turns of the price only
while the price cycles. On BTC daily closes they forecast the next ten days
no better than a coin flip.

## Go deeper

- [`CycleSine`](../../functions_signal/CycleSine.md): the phase formula and the TA-Lib agreement.
- [Tell trending periods from cycling periods](../trend/trend-mode.md): `TrendMode` reads cycle near the crossings and trend away from them.
- [Cycle and spectral analysis notebook](../../notebooks/19-cycle-and-spectral-analysis.ipynb): the sinewave indicator on a clean cycle.
