---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Measure price impact

`RollingKyleLambda` measures how far the price moves per unit of net order
flow, as the slope of returns on signed flow over a trailing window.
`AmihudIlliquidity` measures how far it moves per unit of traded value, in
either direction, and needs only returns and volume.

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from screamer import AmihudIlliquidity, LogReturn, Resample, RollingKyleLambda

data = "../../notebooks/data/"
# Kyle's lambda needs signed flow: one-minute bars of Deribit BTC trades.
trades = pd.read_csv(data + "deribit_btc_perp_6h.csv")
values, index = Resample(freq=60_000, agg="ohlcv2")(
    trades[["price", "volume"]].to_numpy(), trades["timestamp"].to_numpy())
close, buy, sell = values[:, 3], values[:, 4], values[:, 5]
kyle = pd.Series(RollingKyleLambda(30)(buy - sell, LogReturn()(close)),
                 index=pd.to_datetime(index, unit="ms"))

# Amihud needs no signs: daily bars of 20 coins, traded value in USDT.
daily = pd.read_csv(data + "binance_daily_20.csv", parse_dates=["date"])
close_d = daily.pivot(index="date", columns="symbol", values="close")
value_d = daily.pivot(index="date", columns="symbol", values="quote_volume")
amihud = AmihudIlliquidity(30)(LogReturn()(close_d), value_d)
latest = pd.Series(amihud[-1], index=close_d.columns).sort_values()

k, a = 1e10 * kyle, 1e10 * latest
print(f"Kyle: median {k.median():.1f}, {k.min():.1f} to {k.max():.1f}")
print(f"Amihud: {a.index[0]} {a.iloc[0]:.2f} to {a.index[-1]} {a.iloc[-1]:.2f}")
```

The trades have `volume` in US dollars, positive for a buyer-initiated trade
and negative for a seller-initiated one, and `Resample(agg="ohlcv2")` splits
each minute into buy and sell volume. `RollingKyleLambda` takes the net flow
first and the return second, and regresses the return on the flow over the
last 30 minutes. `AmihudIlliquidity` takes the return first and the traded
value second, and averages `|return| / value` over the last 30 days. The
traded value is the `quote_volume` column, in USDT, a dollar stablecoin. Each
column of the daily tables is one coin, and `latest` holds the last day, 30
September 2026. The factor `1e10` turns both into basis points of return
per million USD.

```{code-cell} ipython3
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4),
                               gridspec_kw={"width_ratios": [3, 2]})
ax0.plot(k.index, k, color="steelblue", lw=1)
ax0.set_ylabel("bp per million USD of net flow")
ax0.set_title("RollingKyleLambda(30), BTC perpetual, 1-minute", fontsize=9)
ax0.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax1.barh(a.index, a, color="darkorange")
ax1.set_xscale("log")
ax1.set_xlabel("bp per million USD traded")
ax1.set_title("AmihudIlliquidity(30), daily, 30 Sep 2026", fontsize=9)
ax1.tick_params(axis="y", labelsize=7)
plt.tight_layout()
```

On BTC a million USD of net buying in a minute went with a rise of about 19
basis points, between 5 and 49 depending on the 30 minutes in the window.
Across coins the Amihud ratio spans three orders of magnitude, from 0.09 for
BTC to 109 for ATOM. It suits bars with steady volume, such as daily bars: on
one-minute bars, minutes with a single small trade dominate the average.

## Go deeper

- [`RollingKyleLambda`](../../functions_micro/RollingKyleLambda.md), [`EwKyleLambda`](../../functions_micro/EwKyleLambda.md): the regression and an exponentially weighted version.
- [`AmihudIlliquidity`](../../functions_micro/AmihudIlliquidity.md): the ratio and how zero volume is handled.
- [Price impact notebook](../../notebooks/12-microstructure-price-impact.ipynb): BTC against ETH, the Roll spread, and the propagator model.
