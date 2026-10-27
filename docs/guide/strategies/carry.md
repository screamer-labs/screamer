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
                      PortfolioReport, MaxDrawdown)

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

```{code-cell} ipython3
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
ax0.plot(btc["date"], rate * 365, color="0.4", lw=0.6)
ax0.axhline(0, color="k", lw=0.5)
ax0.set_ylabel("funding (annualized)")
ax1.plot(btc["date"], 1 + out[:, :, 0].sum(axis=1), color="steelblue")
ax1.set_ylabel("equity (starting capital 1)")
plt.tight_layout()
```

The funding file holds the BTCUSDT perpetual's payments on Binance, one row
every eight hours, as a fraction of the position's value; a positive rate means
longs pay shorts. Over this sample it averaged 6.6% a year. The book sets a
target of half the capital in each leg on the first day, which fills at the
second day's open, and holds it. Each leg pays a 0.1% fee and 0.05% slippage
on its one trade.

```{code-cell} ipython3
*_, trades, _, sharpe = PortfolioReport()(out)[-1]
dd = MaxDrawdown()(1 + out[:, :, 0].sum(axis=1))[-1]   # fraction of the peak
pd.Series({"sharpe": sharpe * np.sqrt(365), "max_drawdown_pct": 100 * dd,
           "num_trades": trades}).round(2)
```

The book gains whenever the funding rate is positive and gives some back when
it is negative, and ends at 1.20 times its capital, about 3.8% a year. Both
legs are priced with the daily spot bars, so their price gains and losses
cancel exactly; a real perpetual trades at a small, moving difference to spot
(the basis), and its short needs margin. The Sharpe ratio is therefore not
comparable with that of a directional strategy: the backtest has no basis risk
and no margin cost, so take it as an upper bound.

## Go deeper

- [`BacktestCarry`](../../functions_fin/BacktestCarry.md): funding, borrow fees and financing on any engine's output. Here it charges the daily rate on day t to the position held through day t-1, the position that earned those payments, and pays the short when the rate is positive.
- [`Resample`](../../functions_streams/Resample.md): `origin` and `label` for time buckets. Here one-day buckets shifted by one hour (`origin`) make the rate on day t the sum of the three payments from 08:00 on day t-1 to 00:00 on day t.
- [`PortfolioReport`](../../functions_fin/PortfolioReport.md): the portfolio version of `backtest_report`. Its Sharpe ratio is per day, and `np.sqrt(365)` annualizes it.
- [Backtest a portfolio of many assets](../portfolios/multi-asset.md): the portfolio engine and its output.
- [Portfolios and pairs](../../portfolio_backtesting.md): sizing and reporting a book of several assets.
