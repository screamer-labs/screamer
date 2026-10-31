---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Estimate the bid-ask spread from OHLC bars (Corwin-Schultz, Abdi-Ranaldo, Roll)

`CorwinSchultzSpread` and `AbdiRanaldoSpread` estimate the bid-ask spread from
the high, low and close of each bar, and `RollSpread` estimates it from a price
series alone, for markets where you have prices but no quotes.

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from screamer import (AbdiRanaldoSpread, CorwinSchultzSpread, Resample,
                      RollSpread)

trades = pd.read_csv("../../notebooks/data/deribit_btc_perp_6h.csv")
price, t = trades["price"].to_numpy(), trades["timestamp"].to_numpy()
values, index = Resample(freq=60_000, agg="ohlc")(price, t)
bars = pd.DataFrame(values, index=pd.to_datetime(index, unit="ms"),
                    columns=["open", "high", "low", "close"])

# Both bar estimators return a fraction of the price; times close gives USD.
hlc = bars["high"], bars["low"], bars["close"]
bars["corwin_schultz"] = CorwinSchultzSpread(60)(*hlc) * bars["close"]
bars["abdi_ranaldo"] = AbdiRanaldoSpread(60)(*hlc) * bars["close"]
roll = pd.Series(RollSpread(1000)(price), index=pd.to_datetime(t, unit="ms"))
pd.DataFrame({"Corwin-Schultz": bars["corwin_schultz"].describe(),
              "Abdi-Ranaldo": bars["abdi_ranaldo"].describe(),
              "Roll": roll.describe()}).round(2)
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(bars.index, bars["corwin_schultz"], color="steelblue",
        label="CorwinSchultzSpread(60), minute bars")
ax.plot(bars.index, bars["abdi_ranaldo"], color="seagreen",
        label="AbdiRanaldoSpread(60), minute bars")
ax.plot(roll.index, roll, color="crimson", lw=1, label="RollSpread(1000), trades")
ax.axhline(0.5, color="k", ls="--", lw=0.8, label="one tick (0.5 USD)")
ax.set_ylabel("spread (USD)")
ax.legend(loc="upper left", fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
plt.tight_layout()
```

The data is six hours of Deribit BTC perpetual trades, where the quoted spread
is usually one tick of 0.5 USD, so the estimates can be checked against a
known value. `Resample(agg="ohlc")` turns the trades into one-minute bars.
`CorwinSchultzSpread` compares the high-low range of one bar with the range of
two bars together: price variance grows with the length of the period and the
spread does not, which separates the two. `AbdiRanaldoSpread` measures how far
each close sits from the midpoint of its own bar's high and low and from the
next bar's. Both average over the last 60 bars. `RollSpread` reads the bounce
between bid and ask in consecutive price changes, here over the last 1,000
trades, and gives `NaN` when that covariance is not negative.

Corwin-Schultz gives a median of 0.28 USD, about half a tick, and Roll a
median of 0.72 USD, with `NaN` in about a third of its windows. Abdi-Ranaldo
gives 0 in most windows, because the average of its two-bar products is
negative there: at one minute the spread is small next to the price moves, and
the estimator cannot separate the two. All three were built for stocks, where
the spread is a larger share of a bar's range. On the daily BTC bars of
`binance_daily_20.csv`, `CorwinSchultzSpread(30)` gives about 0.9% of the price,
while one tick here is under 0.002%: on a market this liquid, daily bars hold
too little of the spread for these estimators to measure it.

## Go deeper

- [`CorwinSchultzSpread`](../../functions_micro/CorwinSchultzSpread.md), [`AbdiRanaldoSpread`](../../functions_micro/AbdiRanaldoSpread.md), [`RollSpread`](../../functions_micro/RollSpread.md): the formulas and the overnight adjustment.
- [Measure the effective and realized spread of your fills](fill-quality.md): the spread from quotes instead of bars.
- [Price impact notebook](../../notebooks/12-microstructure-price-impact.ipynb): the Roll spread on BTC and ETH.
