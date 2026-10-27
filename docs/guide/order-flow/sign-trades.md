---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Sign trades as buys or sells

`TickRuleSign` labels each trade as a buy (+1) or a sell (-1) from prices
alone: a trade above the previous price is a buy, one below it a sell, and a
trade at the same price keeps the previous sign. `SignedVolume` multiplies the
sign by the trade size to give signed order flow.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from screamer import SignedVolume, TickRuleSign

trades = pd.read_csv("../../notebooks/data/deribit_btc_perp_6h.csv")
trades.index = pd.to_datetime(trades.pop("timestamp"), unit="ms")
size = trades["volume"].abs()            # US dollars
venue_sign = np.sign(trades["volume"])   # the exchange's own label

trades["tick_sign"] = TickRuleSign()(trades["price"])
trades["tick_flow"] = SignedVolume()(trades["tick_sign"], size)

signed = trades["tick_sign"].notna()
agree = (trades["tick_sign"] == venue_sign)[signed].mean()
print(f"{len(trades):,} trades, tick rule agrees with the venue on {agree:.0%}")
```

The data is six hours of Deribit BTC perpetual trades. Its `volume` column
is the trade size in US dollars, positive when the buyer took the offer and
negative when the seller hit the bid, so it carries the true sign to check
against. Most public trade feeds do not. The tick rule gives `NaN` until the
first price change; here that is the first 40 trades.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(trades.index, trades["volume"].cumsum() / 1e6, color="0.2", lw=1,
        label="venue sign")
ax.plot(trades.index, trades["tick_flow"].fillna(0).cumsum() / 1e6,
        color="steelblue", lw=1, label="TickRuleSign")
ax.set_ylabel("cumulative net flow (million USD)")
ax.legend(loc="upper left", fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
plt.tight_layout()
```

The two lines stay within about 0.3 million USD of each other for the first
two hours. Between 02:00 and 02:30 the tick rule labels a run of large sells
as buys, and since each wrong label moves the line by twice the trade's size,
the lines then stay about 0.5 million apart until 05:00. Use the exchange's
sign when the feed has one, and the tick rule when it does not.

## Go deeper

- [`TickRuleSign`](../../functions_micro/TickRuleSign.md), [`SignedVolume`](../../functions_micro/SignedVolume.md): the rule and the product.
- [`LeeReadySign`](../../functions_micro/LeeReadySign.md), [`BulkVolumeClassifier`](../../functions_micro/BulkVolumeClassifier.md): signs from quotes, or a buy share per bar from returns.
- [Order flow notebook](../../notebooks/11-microstructure-order-flow.ipynb): signing, imbalance, and trade clustering on the same trades.
