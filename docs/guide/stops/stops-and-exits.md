---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Add stops and exits

`BacktestOHLCTarget` closes a position on its own when the price moves too
far: `stop_loss` against the entry, `take_profit` in favor of it, and
`trailing_stop` back from the best price since entry, each a fraction of the
price (0.10 is 10%).

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
from screamer import (EwMean, Sign, BacktestOHLCTarget, backtest_report,
                      MaxDrawdown)

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
btc = df[df["symbol"] == "BTC"].set_index("date")
o, h, l, c = (btc[k] for k in ["open", "high", "low", "close"])
weight = Sign()(EwMean(span=20)(c) - EwMean(span=100)(c))
weight[:100] = 0.0                          # slow average still warming up

base = dict(taker_fee=1e-3, slippage=5e-4, size="weight", initial_capital=1.0)
exits = {"no exits": {}, "10% stop-loss": {"stop_loss": 0.10},
         "15% trailing stop": {"trailing_stop": 0.15}}
out = {name: BacktestOHLCTarget(**base, **kw)(weight, o, h, l, c)
       for name, kw in exits.items()}
summary = pd.DataFrame({name: backtest_report(res, periods_per_year=365)[1]
                        for name, res in out.items()})
```

The rule is the moving-average trend rule of [Follow a trend with a
moving-average crossover](../strategies/trend-following.md), long or short one
times equity on daily BTC bars. Each target fills at the next day's open, with
a 0.1% taker fee and 0.05% slippage on every fill. The stop-loss compares the
price with the average entry fill. The trailing stop compares it with the
highest high since entry for a long, and the lowest low for a short. When a
bar opens beyond the exit level the position closes at that open; otherwise it
closes at the level, on the bar whose low or high reaches it. An exit is a
market order and pays the same fee and slippage as any other trade. After an
exit the engine ignores further targets on the same side, so the position
stays flat until the signal turns flat or reverses.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(9, 3.5))
for (name, res), color in zip(out.items(), ["0.6", "crimson", "darkorange"]):
    ax.plot(btc.index, 1 + res[:, 0], color=color, label=name)
ax.axhline(1, color="k", lw=0.5)
ax.set_ylabel("equity (starting capital 1)")
ax.legend(loc="upper left", fontsize=8)
plt.tight_layout()
```

```{code-cell} ipython3
summary.loc["max_drawdown_pct"] = [100 * MaxDrawdown()(1 + res[:, 0])[-1]
                                   for res in out.values()]   # % of the peak
summary.loc[["sharpe", "max_drawdown_pct", "num_trades", "exposure"]].round(2)
```

`exposure` is the share of days with a position. On this sample both exits
reduce the largest drawdown, from 53% of the peak to 43% with the stop-loss
and 38% with the trailing stop, which holds a position on fewer than half the
days. The two levels were picked as round numbers and compared on the same
data; choosing among variants this way is a form of selection, which a
parameter grid accounts for.

## Go deeper

- [`BacktestOHLCTarget`](../../functions_fin/BacktestOHLCTarget.md): every exit parameter, fill prices and delays.
- [A parameter grid and its robustness](../../recipes/parameter-grid.md): account for picking the best of several exit levels.
- [Choosing a backtest engine](../../functions_fin/choosing_a_backtest_engine.md): engines with limit and stop orders.
