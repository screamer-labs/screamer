---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Measure price impact

`RollingKyleLambda` measures how far the price moves per unit of net order
flow: the slope of returns on signed flow over a trailing window.

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from screamer import LogReturn, Resample, RollingKyleLambda

trades = pd.read_csv("../../notebooks/data/deribit_btc_perp_6h.csv")
values, index = Resample(freq=60_000, agg="ohlcv2")(
    trades[["price", "volume"]].to_numpy(), trades["timestamp"].to_numpy())
bars = pd.DataFrame(values, columns=["open", "high", "low", "close", "buy", "sell"],
                    index=pd.to_datetime(index, unit="ms"))

bars["ret"] = LogReturn()(bars["close"])
bars["kyle"] = RollingKyleLambda(30)(bars["buy"] - bars["sell"], bars["ret"])
# In basis points of return per million USD of net flow.
impact = 1e10 * bars["kyle"]
impact.describe().round(1)
```

The bars are the one-minute bars of the [imbalance page](imbalance.md), with
buy and sell volume in US dollars. `RollingKyleLambda` takes the net flow
first and the return second, and regresses the return on the flow over the
last 30 minutes. It is `NaN` until the window is full. The factor `1e10`
turns return per USD into basis points per million USD.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(impact.index, impact, color="steelblue", lw=1,
        label="RollingKyleLambda(30)")
ax.set_ylabel("bp per million USD of net flow")
ax.legend(loc="upper left", fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
plt.tight_layout()
```

On BTC a million USD of net buying in a minute went with a rise of about 19
basis points, between 5 and 49 depending on the 30 minutes in the window.

## Go deeper

- [`RollingKyleLambda`](../../functions_micro/RollingKyleLambda.md), [`EwKyleLambda`](../../functions_micro/EwKyleLambda.md): the regression and an exponentially weighted version.
- [Measure illiquidity from daily bars](illiquidity.md): price impact without trade signs.
- [Price impact notebook](../../notebooks/12-microstructure-price-impact.ipynb): BTC against ETH, the Roll spread, and the propagator model.
