---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Measure order-flow imbalance

`OFI` measures the balance between buying and selling in a bar as
`(buy - sell) / (buy + sell)`, from -1 (only sells) to +1 (only buys).
`RollingOrderImbalance` sums net flow over a trailing window instead.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import LogReturn, OFI, Resample

trades = pd.read_csv("../../notebooks/data/deribit_btc_perp_6h.csv")
# ohlcv2 splits the signed volume of each minute into buy and sell volume.
values, index = Resample(freq=60_000, agg="ohlcv2")(
    trades[["price", "volume"]].to_numpy(), trades["timestamp"].to_numpy())
bars = pd.DataFrame(values, columns=["open", "high", "low", "close", "buy", "sell"],
                    index=pd.to_datetime(index, unit="ms"))

bars["ofi"] = OFI()(bars["buy"], bars["sell"])
bars["ret"] = LogReturn()(bars["close"])
print(f"{len(bars)} one-minute bars, "
      f"correlation of OFI with the bar's return: {bars['ofi'].corr(bars['ret']):.2f}")
```

The trades are six hours of Deribit BTC perpetual trades, with `volume` in US
dollars, positive for a buyer-initiated trade and negative for a
seller-initiated one. `Resample(agg="ohlcv2")` turns them into one-minute
bars with separate buy and sell volume; a minute without trades gives no
bar. The return of each bar is the log change of its close from the previous
bar's close.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(bars["ofi"], 1e4 * bars["ret"], s=12, color="steelblue", alpha=0.6)
ax.axhline(0, color="0.5", lw=0.6)
ax.axvline(0, color="0.5", lw=0.6)
ax.set_xlabel("OFI of the bar")
ax.set_ylabel("return of the bar (bp)")
plt.tight_layout()
```

Most minutes, 62% of them, hold only buys or only sells, so their OFI is -1
or +1. The
correlation of 0.46 says that minutes with more buying than selling tend to
end higher. Both values describe the same minute, so here OFI explains the
move rather than forecasting it.

## Go deeper

- [`OFI`](../../functions_micro/OFI.md), [`RollingOrderImbalance`](../../functions_micro/RollingOrderImbalance.md): the normalized imbalance and the windowed sum.
- [`ContOFI`](../../functions_micro/ContOFI.md): imbalance from order book changes instead of trades.
- [Order flow notebook](../../notebooks/11-microstructure-order-flow.ipynb): imbalance, its persistence, and bulk volume classification.
