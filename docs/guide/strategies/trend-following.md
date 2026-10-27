---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Follow a trend with a moving-average crossover

A trend-following rule holds a long position while a fast `EwMean` of the
close is above a slow one and a short position while it is below;
`BacktestOHLCTarget` trades it and `backtest_report` summarizes the result.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import (EwMean, Sign, BacktestOHLCTarget,
                      backtest_report, MaxDrawdown)

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
btc = df[df["symbol"] == "BTC"].set_index("date")
o, h, l, c = (btc[k] for k in ["open", "high", "low", "close"])

weight = Sign()(EwMean(span=20)(c) - EwMean(span=100)(c))
weight[:100] = 0.0                          # slow average still warming up

engine = BacktestOHLCTarget(taker_fee=1e-3, slippage=5e-4,
                            size="weight", initial_capital=1.0)
out = engine(weight, o, h, l, c)
hold = engine(np.ones_like(weight), o, h, l, c)
running, summary = backtest_report(out, periods_per_year=365)
```

The data is daily Binance BTC/USDT bars from January 2022 to September 2026.
The weight is +1 or -1 times the account's equity, decided at each close and
traded at the next day's open. Every fill pays a 0.1% taker fee and 0.05%
slippage. The first 100 days stay flat, because the 100-day average rests on
few prices there. The spans are the ones of the [crossover
page](../trend/crossover.md), not tuned on this data. `hold` buys BTC on the
first day and keeps it, as a reference.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(btc.index, 1 + out[:, 0], color="steelblue", label="trend rule")
ax.plot(btc.index, 1 + hold[:, 0], color="0.6", lw=0.8, label="buy and hold")
ax.axhline(1, color="k", lw=0.5)
ax.set_ylabel("equity (starting capital 1)")
ax.legend(loc="upper left", fontsize=8)
plt.tight_layout()
```

```{code-cell} ipython3
dd = MaxDrawdown()(1 + out[:, 0])[-1]           # fraction of the peak
pd.Series({"sharpe": summary["sharpe"], "max_drawdown_pct": 100 * dd,
           "num_trades": summary["num_trades"]}).round(2)
```

The Sharpe ratio is annualized, `max_drawdown_pct` is the largest fall of
equity from a peak in percent of that peak (`MaxDrawdown`), and `num_trades`
counts the days on which the position changed. This is a template to start from; one
run on one sample does not show that the rule has an edge.

## Go deeper

- [Detect a trend with a moving-average crossover](../trend/crossover.md): the signal on its own.
- [Add stops and exits](../stops/stops-and-exits.md): this rule with a stop-loss and a trailing stop.
- [Backtest a signal on OHLCV bars](../backtest/ohlcv.md): the engine's timing and output columns.
- [Check that a backtest result is not luck](../not-luck/index.md): compare the rule with randomly timed copies.
