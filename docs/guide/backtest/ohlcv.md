---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Backtest a signal on OHLCV bars

`BacktestOHLCTarget` takes a target position for each bar, trades it at the
next bar's open with fees and slippage, and returns the equity curve;
`backtest_report` summarizes that curve.

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
from screamer import EwMean, Sign, BacktestOHLCTarget, backtest_report

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
btc = df[df["symbol"] == "BTC"].set_index("date")
o, h, l, c = (btc[k] for k in ["open", "high", "low", "close"])

# Signal: long 1 BTC while the 20-day average is above
# the 100-day one, short while it is below.
signal = Sign()(EwMean(span=20)(c) - EwMean(span=100)(c))

# Backtest: a 0.1% taker fee and 0.05% slippage on every fill.
engine = BacktestOHLCTarget(taker_fee=1e-3, slippage=5e-4)
out = engine(signal, o, h, l, c)

running, summary = backtest_report(out, periods_per_year=365)
pd.Series(summary)[["total_pnl", "total_cost", "max_drawdown", "sharpe",
                    "round_trips", "win_rate"]].round(2)
```

The data is daily Binance BTC/USDT bars from January 2022 to September 2026.
The signal on each day uses that day's close. The engine holds it and trades
it at the next day's open, so the signal goes in as it is, without a lag. A
target is a position in units, here BTC, so +1 is long one BTC. `EwMean`
gives a value from the first day, so the first trades rest on averages of
only a few prices.

`out` has one row per bar and four columns: `equity` (the cumulative profit,
in USD), `pnl` (the profit of the bar), `position` (BTC held at the close),
and `cost` (fees and slippage paid on the bar). `backtest_report` returns
`running`, the statistics as series over time, and `summary`, their final
values. `periods_per_year=365` annualizes the Sharpe ratio for a market that
trades every day.

```{code-cell} ipython3
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 5), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
ax0.plot(btc.index, out[:, 0], color="steelblue", label="strategy")
ax0.plot(btc.index, c - c.iloc[0], color="0.6", lw=0.8, label="buy and hold")
ax0.axhline(0, color="k", lw=0.5)
ax0.set_ylabel("profit (USD, 1 BTC)")
ax0.legend(loc="upper left", fontsize=8)
ax1.step(btc.index, out[:, 2], where="post", color="indigo")
ax1.set_ylabel("position (BTC)")
plt.tight_layout()
```

The rule made about 56,000 USD over 21 round trips, after 3,500 USD of costs,
with a Sharpe ratio of 0.40. Its worst drawdown was 45,000 USD, and a third of
its round trips made money. Holding one BTC over the same period made about
36,000 USD. One run on one sample does not show that the timing has an edge.

## Go deeper

- [See how fees and slippage change a result](costs.md): the same kind of rule at three cost levels.
- [Test whether a strategy's timing beats chance](../not-luck/timing-test.md): compare the rule with randomly timed copies of its own trades.
- [Size a position by volatility](../signal-to-position/volatility-sizing.md): trade a fraction of equity with `size="weight"`.
- [`BacktestOHLCTarget`](../../functions_fin/BacktestOHLCTarget.md), [`backtest_report`](../../functions_fin/backtest_report.md): fill prices, delays, sizing, stops, and every statistic in the report.
- [Choosing a backtest engine](../../functions_fin/choosing_a_backtest_engine.md): engines for prices, trades, and quotes, and for limit orders.
- [Backtesting best practices](../../backtesting_best_practices.md): look-ahead, fills, costs, and luck.
