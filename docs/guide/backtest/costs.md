---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# See how fees and slippage change a result

`BacktestOHLCTarget` charges `taker_fee` on the traded value and moves each
fill against the trade by `slippage`; running the same signal with and
without them shows how much of the result the costs take.

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from screamer import Resample, EwMean, Sign, BacktestOHLCTarget, backtest_report

trades = pd.read_csv("../../notebooks/data/deribit_btc_perp_6h.csv")
bars, ts = Resample(freq=60_000, agg="ohlc")(     # one-minute bars
    trades["price"].to_numpy(), trades["timestamp"].to_numpy())
o, h, l, c = bars.T
signal = Sign()(EwMean(span=5)(c) - EwMean(span=30)(c))

costs = {"no costs": {},
         "0.05% fee": {"taker_fee": 5e-4},
         "0.05% fee + 0.02% slippage": {"taker_fee": 5e-4, "slippage": 2e-4}}
equity, summary = {}, {}
for name, kwargs in costs.items():
    out = BacktestOHLCTarget(**kwargs)(signal, o, h, l, c)
    equity[name] = out[:, 0]
    summary[name] = backtest_report(out)[1]
pd.DataFrame(summary).loc[["total_pnl", "total_cost", "num_trades"]].round(1)
```

The bars are one-minute bars built from six hours of Deribit BTC-perpetual
trades, as on [Build bars from trades](../get-data-in/bars-from-ticks.md), and
the signal is the moving-average rule of [the previous page](ohlcv.md) with
spans of 5 and 30 minutes. Each run holds the same positions and fills at the
next bar's open, so the gap between two curves is the cost alone. `slippage`
is a fraction of the price by default; `slippage_unit="ticks"` with
`tick_size` sets it in ticks of the price grid instead.

```{code-cell} ipython3
t = pd.to_datetime(ts, unit="ms")
fig, ax = plt.subplots(figsize=(9, 3.5))
for (name, curve), color in zip(equity.items(), ["0.6", "steelblue", "crimson"]):
    ax.plot(t, curve, color=color, label=name)
ax.axhline(0, color="k", lw=0.5)
ax.set_ylabel("profit (USD, 1 BTC)")
ax.legend(loc="lower left", fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
plt.tight_layout()
```

The rule loses 72 USD before costs. With BTC near 29,000 USD, a 0.05% fee is
about 15 USD a contract, and the 17 trades, most of them reversals of two
contracts, pay 480 USD in fees. That is more than three times the price range
of the whole six hours, about 140 USD. On the daily bars of the previous page
the costs took 3,500 USD of a 56,000 USD profit: the slower the rule, the
smaller the share of its moves that the costs take.

## Go deeper

- [Backtesting a signal notebook](../../notebooks/14-backtesting-a-signal.ipynb): the cost of each trade, running statistics, and a test of the timing on the same tape.
- [Choosing a backtest engine](../../functions_fin/choosing_a_backtest_engine.md): how each engine sets the fill price, slippage, and costs.
- [`BacktestOHLCTarget`](../../functions_fin/BacktestOHLCTarget.md): `fill_at`, `delay`, `tick_size`, `slippage_unit`, and `fee_per_contract`.
- [Event-driven backtests notebook](../../notebooks/15-event-driven-backtests.ipynb): limit orders that fill on the trade tape or against quotes.
