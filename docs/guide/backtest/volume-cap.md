---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Cap trades at a share of each bar's volume (participation limit)

`BacktestOHLCVTarget` fills at most `max_participation` times a bar's volume
on that bar and carries the rest of the order to the next bars, so a large
order fills over several days instead of all at one open.

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
from screamer import EwMean, Sign, BacktestOHLCVTarget, backtest_report

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
xlm = df[df["symbol"] == "XLM"].set_index("date")      # least traded coin
o, h, l, c, v = (xlm[k] for k in ["open", "high", "low", "close", "volume"])
direction = Sign()(EwMean(span=20)(c) - EwMean(span=100)(c))
direction[:100] = 0.0
target = 1_000_000 * direction                          # USD long or short

out, summary = {}, {}
for cap in [1.0, 0.05, 0.01]:
    engine = BacktestOHLCVTarget(max_participation=cap, taker_fee=1e-3,
                                 slippage=5e-4, size="notional")
    out[cap] = engine(target, o, h, l, c, v)
    summary[f"{cap:.0%} of volume"] = backtest_report(
        out[cap], periods_per_year=365)[1]
pd.DataFrame(summary).loc[["total_pnl", "total_cost", "num_trades"]].round(0)
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(9, 3.5))
for (cap, res), color in zip(out.items(), ["0.6", "steelblue", "darkorange"]):
    ax.plot(xlm.index, res[:, 0] / 1e6, color=color, label=f"{cap:.0%} of volume")
ax.axhline(0, color="k", lw=0.5)
ax.set_ylabel("profit (million USD)")
ax.legend(loc="upper left", fontsize=8)
plt.tight_layout()
```

The coin is XLM, the one of the 20 Binance pairs with the smallest median
daily volume in USD, about 9 million. The rule is the 20/100-day trend rule
of [Backtest a signal on OHLCV bars](ohlcv.md), with a target of 1 million USD
long or short; `size="notional"` converts it to XLM at the fill price. The
volume column counts XLM, the same unit as the position.

A target decided at a day's close starts to fill at the next day's open, with
a 0.1% taker fee and 0.05% slippage, as in `BacktestOHLCTarget`. On each day
the order trades at most `max_participation` times that day's volume, and the
rest waits for the following days. A new target replaces the order still
working. `max_participation=1.0`, the default, lets an order take up to the
whole day's volume. At 1%, a day of median volume takes 90,000 USD, so a
reversal of 2 million USD needs about 22 days. Each partial fill counts as a trade.

The slower fills enter each trend later and give up part of its gain, so the
capped runs make less. A cap of 5% to 20% of volume is common in practice. A
stop-loss or other risk exit is not capped: it closes the whole position at
once.

## Go deeper

- [`BacktestOHLCVTarget`](../../functions_fin/BacktestOHLCVTarget.md): every parameter, and how a NaN target continues an order.
- [`BacktestTradesTarget`](../../functions_fin/BacktestTradesTarget.md): `max_participation` of each print on the trade tape.
- [See how fees and slippage change a result](costs.md): the other costs of trading.
- [Choosing a backtest engine](../../functions_fin/choosing_a_backtest_engine.md): sizing and volume limits across the engines.
