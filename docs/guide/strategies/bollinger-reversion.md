---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Fade moves outside the Bollinger Bands

`BollingerBands` draws bands two rolling standard deviations above and below a
20-day mean; a fade rule goes short when the close ends above the upper band,
long when it ends below the lower band, and flat at the middle band, with
`BandSignal` keeping track of the position.

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
from screamer import (BollingerBands, BandSignal, BacktestOHLCTarget,
                      backtest_report, MaxDrawdown)

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
btc = df[df["symbol"] == "BTC"].set_index("date")
o, h, l, c = (btc[k] for k in ["open", "high", "low", "close"])

lower, middle, upper = BollingerBands(20, num_std=2.0)(c).T
z = 2 * (c - middle) / (upper - middle)   # +2 at the upper band
weight = BandSignal(entry=2.0, exit=0.0)(z)

engine = BacktestOHLCTarget(taker_fee=1e-3, slippage=5e-4,
                            size="weight", initial_capital=1.0)
out = engine(weight, o, h, l, c)
running, summary = backtest_report(out, periods_per_year=365)
```

`z` is the distance of the close from the middle band in rolling standard
deviations, so it reaches +2 at the upper band and -2 at the lower one.
`BandSignal(entry=2.0, exit=0.0)` goes short at +2, long at -2, and flat when
`z` crosses 0, the middle band. A window of 20 days and a width of 2 standard
deviations are Bollinger's defaults, not tuned on this data. The data is daily
Binance BTC/USDT bars from January 2022 to September 2026; each target trades
at the next day's open as a fraction of equity, and every fill pays a 0.1%
taker fee and 0.05% slippage.

```{code-cell} ipython3
part = btc.index.year == 2024
t = btc.index[part]
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(t, c[part], color="0.3", lw=0.8, label="close")
ax.fill_between(t, lower[part], upper[part], color="steelblue", alpha=0.15,
                label="Bollinger Bands")
ax.plot(t, middle[part], color="steelblue", lw=0.8, label="20-day mean")
for side, color in [(1, "seagreen"), (-1, "crimson")]:
    ax.fill_between(t, 0, 1, where=out[part, 2] * side > 0, color=color,
                    alpha=0.1, transform=ax.get_xaxis_transform())
ax.set_ylabel("BTC price (USD)")
ax.legend(loc="upper left", fontsize=8)
plt.tight_layout()
```

```{code-cell} ipython3
dd = MaxDrawdown()(1 + out[:, 0])[-1]           # fraction of the peak
pd.Series({"sharpe": summary["sharpe"], "max_drawdown_pct": 100 * dd,
           "num_trades": summary["num_trades"]}).round(2)
```

The plot shows 2024, with long days shaded green and short days red. Over the
sample about seven in ten round trips made money, but an average loss was
about three times an average win.

## Go deeper

- [`BollingerBands`](../../functions_rolling/BollingerBands.md): the bands and the sample standard deviation they use.
- [Trade the reversion with z-score bands](../mean-reversion/zscore-bands.md): the same rule with `RollingZscore`, on a series that reverts.
- [Buy oversold and sell overbought with RSI](rsi-reversion.md): a reversion rule on a different score.
- [Detect mean reversion](../mean-reversion/index.md): test whether a series reverts before fading it.
