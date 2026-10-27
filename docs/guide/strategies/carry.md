---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Earn the funding carry

A carry trade holds BTC spot long and the BTC perpetual future short in equal
value, so price moves cancel and the book collects the funding that longs pay
shorts; `BacktestPortfolioOHLCTarget` runs the two legs and `BacktestCarry`
adds the funding payments to the short leg.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import (Resample, BacktestPortfolioOHLCTarget, BacktestCarry,
                      PortfolioReport)

bars = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                   parse_dates=["date"])
btc = bars[bars["symbol"] == "BTC"]
o, h, l, c = (np.column_stack([btc[k]] * 2)    # (T, 2): spot, perpetual
              for k in ["open", "high", "low", "close"])

funding = pd.read_csv("../../notebooks/data/binance_btc_funding.csv")
DAY, HOUR = 86_400_000, 3_600_000
per_day, edge = Resample(freq=DAY, origin=HOUR, agg="sum", label="right")(
    funding["funding_rate"].to_numpy(), funding["timestamp"].to_numpy())
day = pd.to_datetime(edge - HOUR, unit="ms")
rate = pd.Series(per_day, index=day).reindex(btc["date"], fill_value=0.0)
rate = rate.to_numpy()

target = np.full(o.shape, np.nan)
target[0] = [0.5, -0.5]              # long spot, short perpetual; NaN holds
out = BacktestPortfolioOHLCTarget(fee=1e-3, slippage=5e-4)(target, o, h, l, c)
out[:, 1] = BacktestCarry()(*out[:, 1].T, o[:, 1], rate)
```

The funding file holds the BTCUSDT perpetual's payments on Binance, one row
every eight hours, as a fraction of the position's value; a positive rate means
longs pay shorts. Over this sample it averaged 6.6% a year. `Resample` sums each day's payments into a daily rate, using
one-day buckets shifted by one hour (`origin`) so that a day's sum covers the
three payments made while a position bought at that day's open is held.
`BacktestCarry` charges the short leg's position with that rate, which pays
the short when the rate is positive.

The book puts half the capital in each leg on the first day and holds it. Both
legs are priced with the daily spot bars, so their price gains and losses
cancel exactly; a real perpetual trades at a small, moving difference to spot
(the basis), and its short needs margin. Each leg pays a 0.1% fee and 0.05%
slippage on its one trade.

```{code-cell} ipython3
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
ax0.plot(btc["date"], rate * 365, color="0.4", lw=0.6)
ax0.axhline(0, color="k", lw=0.5)
ax0.set_ylabel("funding (annualized)")
ax1.plot(btc["date"], 1 + out[:, :, 0].sum(axis=1), color="steelblue")
ax1.set_ylabel("equity (starting capital 1)")
plt.tight_layout()
```

```{code-cell} ipython3
*_, trades, max_drawdown, sharpe = PortfolioReport()(out)[-1]
pd.Series({"sharpe": sharpe * np.sqrt(365), "max_drawdown": max_drawdown,
           "num_trades": trades}).round(3)
```

`PortfolioReport` is the portfolio version of `backtest_report`: its Sharpe
ratio is per day, and `np.sqrt(365)` annualizes it. The book gains whenever the
funding rate is positive and gives some back when it is negative. Its Sharpe
ratio is far above that of a directional strategy because the backtest has no
basis risk; take it as an upper bound.

## Go deeper

- [`BacktestCarry`](../../functions_fin/BacktestCarry.md): funding, borrow fees and financing on any engine's output.
- [`Resample`](../../functions_streams/Resample.md): `origin` and `label` for time buckets.
- [Backtest a portfolio of many assets](../portfolios/multi-asset.md): the portfolio engine and its output.
- [Portfolios and pairs](../../portfolio_backtesting.md): sizing and reporting a book of several assets.
