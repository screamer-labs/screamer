---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Track drawdown and distance to liquidation live

`Drawdown` gives the loss from the highest equity so far on every bar, and
`LiquidationDistance` gives how far each bar's worst price was from the price
at which a leveraged position would be liquidated.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import (EwMean, Sign, BacktestOHLCTarget, Drawdown,
                      LiquidationDistance)

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
btc = df[df["symbol"] == "BTC"].set_index("date")
o, h, l, c = (btc[k].to_numpy() for k in ["open", "high", "low", "close"])
direction = Sign()(EwMean(span=20)(c) - EwMean(span=100)(c))
direction[:100] = 0.0
out = BacktestOHLCTarget(taker_fee=1e-3, slippage=5e-4, size="weight",
                         initial_capital=1.0)(direction, o, h, l, c)

drawdown = Drawdown()(1 + out[:, 0])                  # fraction of the peak
liq = {lev: LiquidationDistance(leverage=lev)(out[:, 2], o, h, l)
       for lev in [3, 5]}                             # distance, liq_price, breached
pd.DataFrame({f"{lev}x": {"closest distance": np.nanmin(r[:, 0]),
                          "liquidations": np.sum(np.diff(np.nan_to_num(r[:, 2])) == 1)}
              for lev, r in liq.items()}).round(3)
```

```{code-cell} ipython3
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
ax0.fill_between(btc.index, 100 * drawdown, color="crimson", alpha=0.4)
ax0.set_ylabel("drawdown (%)")
for (lev, r), color in zip(liq.items(), ["steelblue", "darkorange"]):
    ax1.plot(btc.index, 100 * r[:, 0], color=color, lw=0.8, label=f"{lev}x")
ax1.axhline(0, color="k", lw=0.5)
ax1.set_ylabel("distance to liquidation (%)")
ax1.legend(loc="upper left", fontsize=8)
plt.tight_layout()
```

The strategy is the 20/100-day trend rule of [Backtest a signal on OHLCV
bars](ohlcv.md) on daily BTC bars, long or short one times equity
(`size="weight"`), with a 0.1% fee and 0.05% slippage. `Drawdown` divides the
equity `1 + out[:, 0]` by its highest value so far and subtracts 1, so -0.3
means 30% below the peak.

`LiquidationDistance` takes the position the engine reports, column 2 of its
output, which already holds each fill at the bar's open, and the bar's open,
high and low. It treats the position as held on isolated margin at the given
leverage: the margin is the position's value at entry divided by the leverage,
and the default maintenance rate is 0.5%. A long is liquidated at
`entry * (1 - 1 / leverage + maintenance)` and a short at
`entry * (1 + 1 / leverage - maintenance)`. The `distance` output is the move
still needed from the bar's low (for a long) or high (for a short) to that
price, as a fraction of it; 0 or less means the bar reached it, and `breached`
turns 1. All three outputs are NaN while the position is flat. Adding to a
position blends its entry price, and reducing it keeps the entry.

`LiquidationDistance` only measures: it does not close the position. At 3x the
book came within about 7% of liquidation and never reached it. At 5x it
would have been liquidated twice, so 5x was too much leverage for this rule.
Both functions use only the bars up to the current one, so a [live
loop](../live/live-loop.md) can watch them bar by bar and cut the position
before the margin runs out.

## Go deeper

- [`LiquidationDistance`](../../functions_fin/LiquidationDistance.md): the accounting of entries, increases, and reductions.
- [`Drawdown`](../../functions_fin/Drawdown.md), [`MaxDrawdown`](../../functions_fin/MaxDrawdown.md), [`RollingMaxDrawdown`](../../functions_fin/RollingMaxDrawdown.md): the drawdown now, the worst so far, and the worst in a window.
- [Measure tail risk with rolling CVaR and downside deviation (expected shortfall)](../performance/tail-risk.md): risk measured on returns.
- [Run functions in a live event loop](../live/live-loop.md): update a strategy one bar at a time.
