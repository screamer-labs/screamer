---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Hold a position for a fixed number of bars

`Hold(n)` turns a one-bar event into a position held for `n` bars, and
`cooldown=` ignores new events for a number of bars after each hold ends.

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
from screamer import LogReturn, CusumFilter, Hold

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
close = df[df["symbol"] == "BTC"].set_index("date")["close"]

out = pd.DataFrame({"close": close})
# +1 after a cumulative log return of +0.1, -1 after -0.1, 0 on other days
out["event"] = CusumFilter(threshold=0.1)(LogReturn()(close))
out["hold"] = Hold(10)(out["event"])
out["cooldown"] = Hold(10, cooldown=10)(out["event"])

out[["event", "hold", "cooldown"]].dropna().ne(0).sum().rename("days not zero")
```

The events come from [`CusumFilter`](../regimes/cusum.md): +1 on the day
the cumulative log return since the last event reaches 0.1, a rise of about
10.5%, and -1 on the day it reaches -0.1, a fall of about 9.5%.
`Hold(10)` keeps each event for 10 days, the event day included, and returns 0
when the 10 days are over. A new event during a hold replaces the held value
and starts the 10 days again. With `cooldown=10`, events in the 10 days after a
hold ends are ignored, so the position stays flat for at least that long.

```{code-cell} ipython3
part = out.loc["2024"]
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 4.5), sharex=True)
ax0.plot(part.index, part["close"], color="0.2", lw=1)
for side, color in [(1, "seagreen"), (-1, "crimson")]:
    days = part[part["event"] == side]
    ax0.plot(days.index, days["close"], "o", ms=4, color=color)
ax0.set_ylabel("BTC price (USD)")
ax1.step(part.index, part["hold"], where="post", color="0.6", label="Hold(10)")
ax1.step(part.index, part["cooldown"] * 0.9, where="post", color="crimson",
         label="Hold(10, cooldown=10), drawn at ±0.9")
ax1.set_ylabel("position")
ax1.set_ylim(-1.7, 1.2)
ax1.legend(loc="lower left", ncol=2, fontsize=8)
plt.tight_layout()
```

The dots mark the up (green) and down (red) events. Over the whole sample the
117 events give a position on 957 days with `Hold(10)` and on 672 days with
the cooldown.

## Go deeper

- [`Hold`](../../functions_signal/Hold.md): how a new event restarts a hold, the `release` value, and the cooldown rules.
- [Sample events at large cumulative moves](../regimes/cusum.md): the events used here.
- [Stops and exits](../stops/index.md): close a position on a price level instead of a count of bars.
