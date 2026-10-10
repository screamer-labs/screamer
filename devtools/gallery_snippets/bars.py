# Thumbnail for the "Bars from ticks" capability: BTC trades and 10-minute bars.
# Run by devtools/build_gallery.py with docs/notebooks as the working directory.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import Resample

trades = pd.read_csv("data/deribit_btc_perp_6h.csv").iloc[:1400]
ts = trades["timestamp"].to_numpy()
price = trades["price"].to_numpy()
bars, bt = Resample(freq=600_000, agg="ohlc")(price, ts)

fig, ax = plt.subplots(figsize=(6.4, 4))
ax.plot(ts, price, ".", ms=2, color="0.65")
w = 600_000 * 0.6
for t0, (o, h, l, c) in zip(bt, bars):
    x = t0 + 300_000
    color = "#2ca02c" if c >= o else "#d62728"
    ax.vlines(x, l, h, color="0.25", lw=1)
    ax.add_patch(plt.Rectangle((x - w / 2, min(o, c)), w, max(abs(c - o), 0.5),
                               color=color, alpha=0.75))
