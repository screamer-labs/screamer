---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Measure order-flow toxicity (VPIN)

`VPIN` measures how one-sided trading is on a volume clock: it packs trades
into buckets of equal volume and averages the buy-sell imbalance of the last
buckets, from 0 (balanced) to 1 (all buys or all sells).

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from screamer import NegPart, PosPart, VPIN

trades = pd.read_csv("../../notebooks/data/deribit_btc_perp_6h.csv")
trades.index = pd.to_datetime(trades.pop("timestamp"), unit="ms")
buy = PosPart()(trades["volume"])      # USD bought by aggressors
sell = NegPart()(trades["volume"])     # USD sold by aggressors

trades["vpin"] = VPIN(bucket_volume=250_000, n_buckets=10)(buy, sell)
n_buckets = (buy + sell).sum() / 250_000
print(f"{n_buckets:.0f} buckets of 250,000 USD in six hours")
trades["vpin"].describe().round(2)
```

```{code-cell} ipython3
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 4.5), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
ax0.plot(trades.index, trades["price"], color="0.2", lw=1)
ax0.set_ylabel("BTC price (USD)")
ax1.plot(trades.index, trades["vpin"], color="crimson", lw=1)
ax1.set_ylim(0, 1)
ax1.set_ylabel("VPIN")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
plt.tight_layout()
```

The trades are six hours of Deribit BTC perpetual trades, with `volume` in US
dollars, positive for a buyer-initiated trade and negative for a
seller-initiated one. `PosPart` and `NegPart` split it into buy and sell volume,
which `VPIN` takes in that order. Each bucket holds 250,000 USD of volume, and a
trade that crosses a bucket boundary is split between the two buckets. Each
closed bucket contributes `|buy - sell|`, and `VPIN` is the mean of that over
the last 10 buckets divided by the bucket volume. It is `NaN` until 10 buckets
have closed, and changes only when a bucket closes, so a quiet hour moves it
less often than a busy one. Here it ranges from 0.20 to 0.74 and peaks at 0.74
at 05:50, after buyers lifted the price 84 USD from the low near 04:26.

Pick the bucket volume from the instrument's typical volume: here the six hours
hold 61 buckets, so 10 buckets span about an hour of trading. With trades
signed by the venue the imbalance is exact; without the side, sign them first
with `TickRuleSign` or `BulkVolumeClassifier`.

## Go deeper

- [`VPIN`](../../functions_micro/VPIN.md): bucketing, the split of large trades, and missing values.
- [Sign trades as buys or sells](sign-trades.md): buy and sell volume when the venue gives no side.
- [`BulkVolumeClassifier`](../../functions_micro/BulkVolumeClassifier.md): buy and sell volume from bars.
- [Toxicity and book notebook](../../notebooks/13-microstructure-toxicity-and-book.ipynb): VPIN, book pressure and spreads together.
