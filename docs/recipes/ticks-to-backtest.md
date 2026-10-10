---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# From ticks to a costed backtest

This recipe takes a raw trade tape to a backtest with fees and slippage in four
steps: bucket the trades into bars, compute a signal on each bar's close, hand
the signal to `BacktestOHLCTarget`, and summarize the result with
`backtest_report`. The data is six hours of Deribit BTC-perpetual trades.

## Bars

`Resample` with `agg="ohlc"` turns the trade prices into one-minute bars. Each
row of `bars` is `[open, high, low, close]`.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import Resample, EwMean, BacktestOHLCTarget, backtest_report

trades = pd.read_csv("../notebooks/data/deribit_btc_perp_6h.csv")
bars, ts = Resample(freq=60_000, agg="ohlc")(
    trades["price"].to_numpy(), trades["timestamp"].to_numpy())
o, h, l, c = bars.T
t = pd.to_datetime(ts, unit="ms")
print(f"{len(c)} one-minute bars")
```

## Signal

A trend rule: long when the fast exponential average of the close is above the
slow one, short when it is below. The signal at bar `t` uses that bar's close.

```{code-cell} ipython3
signal = np.sign(np.nan_to_num(EwMean(span=5)(c) - EwMean(span=30)(c)))
```

## Backtest with costs

`BacktestOHLCTarget` executes the target decided on bar `t` as a market order on
bar `t + 1`, so the raw signal goes in without a manual lag. `fill_at` picks the
reference price of the execution bar and `slippage` moves each fill against the
trade; here one tick of $0.50 on Deribit's price grid, plus a 5 bps taker fee.
The run without costs shows how much of the result the costs take.

```{code-cell} ipython3
frictionless = BacktestOHLCTarget()(signal, o, h, l, c)
costed = BacktestOHLCTarget(fill_at="open", taker_fee=5e-4,
                            tick_size=0.5, slippage=1, slippage_unit="ticks")(
    signal, o, h, l, c)

running, summary = backtest_report(costed)
{k: round(v, 2) for k, v in summary.items()}
```

The output columns are `[equity, pnl, position, cost]`, one row per bar. The
plot shows both equity curves and the position the costed run held.

```{code-cell} ipython3
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 5), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
ax0.plot(t, frictionless[:, 0], color="0.6", ls="--", label="no costs")
ax0.plot(t, costed[:, 0], color="steelblue", label="5 bps fee + 1 tick slippage")
ax0.axhline(0, color="k", lw=0.5)
ax0.set_ylabel("equity (USD, 1 BTC position)")
ax0.legend(loc="lower left", fontsize=8)
ax1.step(t, costed[:, 2], where="post", color="indigo")
ax1.set_ylabel("position")
plt.tight_layout()
```

On this tape the rule loses money once costs are included. The
[significance recipe](significance-test.md) shows how to test whether a rule's
timing beats chance.

Related pages: [`BacktestOHLCTarget`](../functions_fin/BacktestOHLCTarget.md),
[`backtest_report`](../functions_fin/backtest_report.md),
[Choosing a backtest engine](../functions_fin/choosing_a_backtest_engine.md),
and the [backtesting notebook](../notebooks/14-backtesting-a-signal.ipynb).
