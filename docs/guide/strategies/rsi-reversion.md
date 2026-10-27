---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Buy oversold and sell overbought with RSI

`RollingRSI` scores recent gains against recent losses on a scale of 0 to 100,
and `BandSignal` turns it into a position: short above 70 (overbought), long
below 30 (oversold), and flat again when the RSI crosses 50.

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
from screamer import RollingRSI, BandSignal, BacktestOHLCTarget, backtest_report

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
btc = df[df["symbol"] == "BTC"].set_index("date")
o, h, l, c = (btc[k] for k in ["open", "high", "low", "close"])

rsi = RollingRSI(14)(c)
weight = BandSignal(entry=2.0, exit=0.0)((rsi - 50) / 10)

engine = BacktestOHLCTarget(taker_fee=1e-3, slippage=5e-4,
                            size="weight", initial_capital=1.0)
out = engine(weight, o, h, l, c)
running, summary = backtest_report(out, periods_per_year=365)
```

`BandSignal` reads a score centred on 0, so `(rsi - 50) / 10` maps RSI 70 to
+2 and RSI 30 to -2. With `entry=2.0` it goes short when the score reaches +2
and long at -2; with `exit=0.0` it closes the position when the score crosses
0, RSI 50. The 14-day window and the 70 and 30 levels are Wilder's defaults,
not tuned on this data. The data is daily Binance BTC/USDT bars from January
2022 to September 2026; each target trades at the next day's open as a fraction
of equity, and every fill pays a 0.1% taker fee and 0.05% slippage.

```{code-cell} ipython3
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 5), sharex=True,
                               gridspec_kw={"height_ratios": [1, 2]})
ax0.plot(btc.index, rsi, color="0.3", lw=0.6)
ax0.axhline(70, color="crimson", ls="--", lw=0.8)
ax0.axhline(30, color="seagreen", ls="--", lw=0.8)
ax0.set_ylabel("RollingRSI(14)")
ax1.plot(btc.index, 1 + out[:, 0], color="steelblue")
ax1.axhline(1, color="k", lw=0.5)
ax1.set_ylabel("equity (starting capital 1)")
plt.tight_layout()
```

```{code-cell} ipython3
pd.Series(summary)[["sharpe", "max_drawdown", "num_trades"]].round(2)
```

On daily BTC the rule loses money over the sample. Its long positions made a
little, and its short positions lost about three times as much.

## Go deeper

- [`RollingRSI`](../../functions_rolling/RollingRSI.md): Wilder's and Cutler's smoothing, and the warm-up.
- [`BandSignal`](../../functions_signal/BandSignal.md): entry, exit and stop levels.
- [Fade moves outside the Bollinger Bands](bollinger-reversion.md): the same kind of rule on a z-score of the price.
- [Detect mean reversion](../mean-reversion/index.md): test whether a series reverts before trading the reversion.
