---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Go long or short on the sign of a score

`Sign` turns a score into a position: +1 (long) where the score is positive,
-1 (short) where it is negative, and 0 (flat) where it is exactly zero.

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
from screamer import LogReturn, Sign

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
close = df[df["symbol"] == "BTC"].set_index("date")["close"]

out = pd.DataFrame({"close": close})
out["score"] = LogReturn(30)(close)       # log return over the last 30 days
out["position"] = Sign()(out["score"])

switches = (out["position"].diff().abs() > 0).sum()
print(f"{switches} switches in {len(out)} days")
out.tail().round(3)
```

The score is the 30-day log return of BTC, so the position is long while the
price is above its level 30 days ago and short while it is below. Both are
`NaN` for the first 30 days, until the score has a value. The position on a
day uses that day's close; [`BacktestOHLCTarget`](../../functions_fin/BacktestOHLCTarget.md)
trades it at the next day's open, so it needs no extra lag.

```{code-cell} ipython3
part = out.loc["2024":"2025"]
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 4.5), sharex=True)
ax0.plot(part.index, part["close"], color="0.2", lw=1)
for side, color in [(1, "seagreen"), (-1, "crimson")]:
    ax0.fill_between(part.index, 0, 1, where=part["position"] == side,
                     color=color, alpha=0.15, transform=ax0.get_xaxis_transform(),
                     label="long" if side > 0 else "short")
ax0.set_ylabel("BTC price (USD)")
ax0.legend(loc="upper left", fontsize=8)
ax1.plot(part.index, part["score"], color="0.3", lw=0.8)
ax1.axhline(0, color="k", lw=0.5)
ax1.set_ylabel("30-day log return")
plt.tight_layout()
```

The position switches side 159 times in 4.7 years, about once every 11 days.
Many switches follow the previous one within a few days, while the score
hovers near zero, as in April 2024 and January 2025.

## Go deeper

- [`Sign`](../../functions_math/Sign.md): the output for zero and `NaN`.
- [Enter and exit at different levels](bands.md): a dead band around zero that stops these runs of switches.
- [Detect a trend](../trend/index.md): other scores for the sign, such as a fitted slope or a moving-average crossover.
- [Backtest a signal on OHLCV bars](../backtest/ohlcv.md): trade the position with fees and slippage.
