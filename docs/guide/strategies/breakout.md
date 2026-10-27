---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Trade a breakout of the recent high or low

A breakout rule goes long when the close rises above the highest high of the
previous 20 days and short when it falls below the lowest low; the two levels
come from `DonchianChannels`, `BacktestOHLCTarget` trades the position, and
`backtest_report` summarizes the result.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import (DonchianChannels, Lag, BacktestOHLCTarget,
                      backtest_report, MaxDrawdown)

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
btc = df[df["symbol"] == "BTC"].set_index("date")
o, h, l, c = (btc[k] for k in ["open", "high", "low", "close"])

lower, _, upper = Lag(1)(DonchianChannels(20)(h, l)).T   # previous 20 days
weight = np.where(c > upper, 1.0, np.where(c < lower, -1.0, np.nan))

engine = BacktestOHLCTarget(taker_fee=1e-3, slippage=5e-4,
                            size="weight", initial_capital=1.0)
out = engine(weight, o, h, l, c)
running, summary = backtest_report(out, periods_per_year=365)
```

```{code-cell} ipython3
part = btc.index.year == 2024
t = btc.index[part]
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(t, c[part], color="0.3", lw=0.8, label="close")
ax.plot(t, upper[part], color="seagreen", lw=0.8, label="20-day high")
ax.plot(t, lower[part], color="crimson", lw=0.8, label="20-day low")
ax.fill_between(t, 0, 1, where=out[part, 2] > 0, color="seagreen", alpha=0.12,
                transform=ax.get_xaxis_transform(), label="long")
ax.set_ylabel("BTC price (USD)")
ax.legend(loc="upper left", fontsize=8)
plt.tight_layout()
```

`DonchianChannels(20)(high, low)` returns the lowest low, the midline, and the
highest high of the last 20 days, today included. A close can never be above
a high that includes its own day, so `Lag(1)` compares each close with the
channel up to the day before. On a day with no breakout the weight is NaN,
which the engine reads as "keep the current position", so the rule stays long
after an upside breakout until a downside one turns it short. The data is daily
Binance BTC/USDT bars from January 2022 to September 2026; each target trades
at the next day's open as a fraction of equity, and every fill pays a 0.1%
taker fee and 0.05% slippage. The 20-day window is the usual default, not
tuned on this data.

```{code-cell} ipython3
dd = MaxDrawdown()(1 + out[:, 0])[-1]           # fraction of the peak
pd.Series({"sharpe": summary["sharpe"], "max_drawdown_pct": 100 * dd,
           "num_trades": summary["num_trades"]}).round(2)
```

The plot shows 2024, with the days held long shaded and the unshaded days
held short. Over the full sample the rule loses money, as its negative Sharpe
ratio shows.

## Go deeper

- [`DonchianChannels`](../../functions_rolling/DonchianChannels.md): the channel and its warmup.
- [Follow a trend with a moving-average crossover](trend-following.md): the same summary for a slower trend rule.
- [Add stops and exits](../stops/stops-and-exits.md): close a position before the opposite breakout.
- [Backtest a signal on OHLCV bars](../backtest/ohlcv.md): what a NaN target and the four output columns mean.
