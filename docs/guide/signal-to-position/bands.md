---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Enter and exit at different levels

`SchmittTrigger` switches a long-short position only when a score crosses an
outer level and keeps it in between; `BandSignal` does the same for a
mean-reversion position that can also be flat.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import LogReturn, Sign, SchmittTrigger

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
close = df[df["symbol"] == "BTC"].set_index("date")["close"]

out = pd.DataFrame({"close": close})
out["score"] = LogReturn(30)(close)
out["sign"] = Sign()(out["score"])
# 1 after the score rises above +5%, 0 after it falls below -5%
trigger = SchmittTrigger(lower=-0.05, upper=0.05, initial=np.nan)
out["bands"] = 2 * trigger(out["score"]) - 1     # 1 long, -1 short

out[["sign", "bands"]].diff().abs().gt(0).sum().rename("switches")
```

```{code-cell} ipython3
part = out.loc["2024":"2025"]
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 4.5), sharex=True)
ax0.plot(part.index, part["score"], color="0.3", lw=0.8)
ax0.axhline(0, color="k", lw=0.5)
ax0.hlines([-0.05, 0.05], part.index[0], part.index[-1], color="crimson",
           ls="--", lw=0.8)
ax0.set_ylabel("30-day log return")
ax1.step(part.index, part["sign"], where="post", color="0.6", lw=0.8,
         label="Sign")
ax1.step(part.index, part["bands"] * 0.9, where="post", color="crimson",
         label="SchmittTrigger, ±5% (drawn at ±0.9)")
ax1.set_ylabel("position")
ax1.set_ylim(-1.7, 1.2)
ax1.legend(loc="lower left", ncol=2, fontsize=8)
plt.tight_layout()
```

The score is the 30-day log return from [the previous page](sign.md).
`SchmittTrigger` outputs 1 once the score rises above `upper` and 0 once it
falls below `lower`, and holds its last value while the score is between the
two, so `2 * trigger - 1` is a long-short position. The levels of plus and
minus 5% are a choice: a wider band switches less often and later.

The band cuts the switches from 159 to 47 over the whole sample. The cost is a
later turn: the position waits until the score is 5% past zero.

## Go deeper

- [`SchmittTrigger`](../../functions_signal/SchmittTrigger.md): the thresholds and the `initial` state; `initial=np.nan` leaves the output undefined until the first crossing.
- [`BandSignal`](../../functions_signal/BandSignal.md): entry, exit and stop levels, and the lockout after a stop. `BandSignal(entry, exit)` reads a z-score the other way, for a reverting series: it goes short when the score reaches `entry`, long at `-entry`, and flat when it comes back to `exit`.
- [Trade the reversion with z-score bands](../mean-reversion/zscore-bands.md): `BandSignal` on a reverting series.
- [Hold a position for a fixed number of bars](hold.md): a time limit instead of a level.
