---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Estimate the bid-ask spread from OHLC bars (EDGE, Corwin-Schultz, Abdi-Ranaldo, Roll)

`EdgeSpread`, `CorwinSchultzSpread` and `AbdiRanaldoSpread` estimate the
bid-ask spread from the open, high, low and close of each bar, and `RollSpread`
estimates it from a price series alone, for markets where you have prices but
no quotes.

```{code-cell} ipython3
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from screamer import (AbdiRanaldoSpread, CorwinSchultzSpread, EdgeSpread,
                      Resample, RollSpread)

trades = pd.read_csv("../../notebooks/data/deribit_btc_perp_6h.csv")
price, t = trades["price"].to_numpy(), trades["timestamp"].to_numpy()
values, index = Resample(freq=60_000, agg="ohlc")(price, t)
bars = pd.DataFrame(values, index=pd.to_datetime(index, unit="ms"),
                    columns=["open", "high", "low", "close"])

ohlc = bars["open"], bars["high"], bars["low"], bars["close"]
bars["edge"] = EdgeSpread(60, sign=True)(*ohlc) * bars["close"]
bars["corwin_schultz"] = CorwinSchultzSpread(60)(*ohlc[1:]) * bars["close"]
bars["abdi_ranaldo"] = AbdiRanaldoSpread(60)(*ohlc[1:]) * bars["close"]
roll = pd.Series(RollSpread(1000)(price), index=pd.to_datetime(t, unit="ms"))
pd.DataFrame({"EDGE": bars["edge"].describe(),
              "Corwin-Schultz": bars["corwin_schultz"].describe(),
              "Abdi-Ranaldo": bars["abdi_ranaldo"].describe(),
              "Roll": roll.describe()}).round(2)
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(bars.index, bars["corwin_schultz"], label="CorwinSchultzSpread(60), bars")
ax.plot(bars.index, bars["abdi_ranaldo"], label="AbdiRanaldoSpread(60), bars")
ax.plot(roll.index, roll, lw=1, label="RollSpread(1000), trades")
ax.axhline(0.5, color="k", ls="--", lw=0.8, label="one tick (0.5 USD)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.set(xlabel="time (UTC), 4 August 2023", ylabel="spread (USD)")
ax.legend(loc="upper left", fontsize=8)
```

The data is six hours of Deribit BTC perpetual trades, where the quoted spread
is usually one tick of 0.5 USD, a known value to hold the estimates against.
`Resample(agg="ohlc")` turns the trades into one-minute bars. The bar
estimators return a fraction of the price, which the code multiplies by the
close to give USD.
`CorwinSchultzSpread` compares the high-low range of one bar with the range of
two bars together: price variance grows with the length of the period and the
spread does not, which separates the two. `AbdiRanaldoSpread` measures how far
each close sits from the midpoint of its own bar's high and low and from the
next bar's. `EdgeSpread` combines the returns between the open, the high-low
midpoint and the previous close into an estimate of the squared spread, and
with `sign=True` it reports a negative value where that estimate is negative.
The three average over the last 60 bars. `RollSpread` reads the bounce between
bid and ask in consecutive price changes, here over the last 1,000 trades, and
gives `NaN` when that covariance is not negative.

Corwin-Schultz gives a median of 0.28 USD, about half a tick, and Roll a
median of 0.72 USD, with `NaN` in about a third of its windows. Roll estimates
the effective spread, what trades paid against the mid, which includes orders
that walk through the book and is about 2 USD here (see [Measure the effective
and realized spread of your fills](fill-quality.md)), so 0.72 USD lies between
the quoted and the effective spread. Abdi-Ranaldo gives 0 in most windows,
because the average of its two-bar products is negative there: at one minute
the spread is small next to the price moves, and the estimator cannot separate
the two. EDGE is negative in every window, with a median of -3.74 USD, a
squared spread below zero that no spread produces. It assumes that the open,
high, low and close are trades on a continuous price, and these bars are a few
trades on a 0.5 USD tick, with the open at the bar's high or low in 95% of
them. With the default `sign=False` the same windows report 3.74 USD, so ask
for the sign where the spread may be small.

All four were built for daily bars of stocks, where the spread is a larger
share of a bar's range. The daily bars of 20 coins in `binance_daily_20.csv`
test the high-low estimators in the regime where the spread is small next to
the daily range: these coins trade with spreads well below 0.1%.

```{code-cell} ipython3
daily = pd.read_csv("../../notebooks/data/binance_daily_20.csv")
est = {}
for symbol, g in daily.groupby("symbol"):
    o, h, l, c = (g[k].to_numpy() for k in ("open", "high", "low", "close"))
    est[symbol] = {"EDGE, all days": EdgeSpread(len(g), sign=True)(o, h, l, c)[-1],
                   "Corwin-Schultz, 30-day median":
                       pd.Series(CorwinSchultzSpread(30)(h, l, c)).median()}
(pd.DataFrame(est).T * 100).describe().loc[["min", "50%", "max"]].round(2)
```

Over the 1,734 days of each coin, EDGE gives between -0.13% and +0.26% of the
price, scattered around 0 with a median of -0.05%, while the median
Corwin-Schultz estimate is between 0.72% and 1.95% for every coin, the upward
bias its page describes. The EDGE estimates are not precise enough to read a
spread this small, but they do not report a large one where there is none. The
high-low estimators measure a spread best on daily bars of assets whose spread
is a visible share of the daily range, such as small-cap stocks and thinly
traded coins, and serve best to compare liquidity across assets or over time
rather than to read an absolute level.

## Go deeper

- [`EdgeSpread`](../../functions_micro/EdgeSpread.md), [`CorwinSchultzSpread`](../../functions_micro/CorwinSchultzSpread.md), [`AbdiRanaldoSpread`](../../functions_micro/AbdiRanaldoSpread.md), [`RollSpread`](../../functions_micro/RollSpread.md): the formulas, the overnight adjustment and the limitations of each estimator.
- [Measure the effective and realized spread of your fills](fill-quality.md): the spread from quotes instead of bars.
- [Price impact notebook](../../notebooks/12-microstructure-price-impact.ipynb): the Roll spread on BTC and ETH.
