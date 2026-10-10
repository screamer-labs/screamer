---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Funding costs on a perpetual

A perpetual future has no expiry; instead, longs and shorts exchange a funding
payment at fixed times, every eight hours on Binance. `BacktestCarry` adds
those payments to any backtest engine's output. This recipe charges the
recorded BTCUSDT perpetual funding rates from January 2022 to September 2026
on two daily strategies, buy-and-hold and a trend rule, and compares their
equity with and without funding.

## Funding per day

The funding file has one row per payment: a timestamp in milliseconds and the
rate, a fraction of the position's value. A positive rate means longs pay
shorts; a negative rate means shorts pay longs.

`BacktestCarry` charges the position held into step `t` with the rate on step
`t`. On daily bars that fill at the open, the position decided on day `t - 1`
is held from day `t - 1`'s open to day `t`'s open, which covers the payments at
08:00 and 16:00 UTC on day `t - 1` and at 00:00 UTC on day `t`. `Resample` with
one-day buckets shifted by one hour (`origin`) sums exactly those three
payments, and `label="right"` stamps each sum at the bucket's right edge, one
hour past midnight of day `t`.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import (EwMean, Sign, FillNa, Resample, BacktestOHLCTarget,
                      BacktestCarry, backtest_report)

bars = pd.read_csv("../notebooks/data/binance_daily_20.csv", parse_dates=["date"])
btc = bars[bars["symbol"] == "BTC"]
t = btc["date"].to_numpy()
o, h, l, c = (btc[k].to_numpy() for k in ["open", "high", "low", "close"])

funding = pd.read_csv("../notebooks/data/binance_btc_funding.csv")
DAY, HOUR = 86_400_000, 3_600_000
per_day, edge = Resample(freq=DAY, origin=HOUR, agg="sum", label="right")(
    funding["funding_rate"].to_numpy(), funding["timestamp"].to_numpy())
day = pd.to_datetime(edge - HOUR, unit="ms")
rate = pd.Series(per_day, index=day).reindex(btc["date"], fill_value=0.0).to_numpy()
print(f"mean funding {rate.mean():.5f} per day, {rate.mean() * 365:.1%} per year")
```

Funding averaged 6.6% of the position's value per year over this sample, paid
by longs.

## Equity with and without funding

The price data are Binance spot bars, which stand in for the perpetual's price.
`BacktestCarry` takes the engine's four columns, a price to value the position
at, and the rate. The open of day `t` is the price at the 00:00 payment.

```{code-cell} ipython3
signal = Sign()(FillNa(0.0)(EwMean(span=20)(c) - EwMean(span=100)(c)))
engine = BacktestOHLCTarget(taker_fee=1e-3, slippage=5e-4)

runs = {}
for name, target in [("buy and hold", np.ones_like(c)), ("trend", signal)]:
    out = engine(target, o, h, l, c)
    funded = BacktestCarry()(*out.T, o, rate)
    runs[name] = out, funded
    s0 = backtest_report(out, periods_per_year=365)[1]
    s1 = backtest_report(funded, periods_per_year=365)[1]
    print(f"{name:13s} pnl {s0['total_pnl']:7.0f} -> {s1['total_pnl']:7.0f}   "
          f"Sharpe {s0['sharpe']:.2f} -> {s1['sharpe']:.2f}")
```

Holding one BTC long paid about $18,900 of funding, more than half of its pnl
before funding. The
trend rule paid about $11,600: it pays funding on its long days and receives it
on its short days, and because the rate is positive most of the time the short
days partly offset the long ones.

```{code-cell} ipython3
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                               gridspec_kw={"height_ratios": [1, 2]})
ax0.plot(t, rate * 365, color="0.4", lw=0.8)
ax0.axhline(0, color="k", lw=0.5)
ax0.set_ylabel("funding (annualized)")
for (name, (out, funded)), color in zip(runs.items(), ["0.5", "steelblue"]):
    ax1.plot(t, out[:, 0], color=color, ls="--", lw=0.8, label=f"{name}, no funding")
    ax1.plot(t, funded[:, 0], color=color, label=f"{name}, with funding")
ax1.axhline(0, color="k", lw=0.5)
ax1.set_ylabel("equity (USD, one BTC)")
ax1.legend(loc="upper left", fontsize=8)
plt.tight_layout()
```

Related pages: [`BacktestCarry`](../functions_fin/BacktestCarry.md),
[`Resample`](../functions_streams/Resample.md),
[`BacktestOHLCTarget`](../functions_fin/BacktestOHLCTarget.md), and
[`backtest_report`](../functions_fin/backtest_report.md).
