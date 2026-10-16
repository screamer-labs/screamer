# Thumbnail for the "Labels and classification" capability: one-minute BTC
# closes, each marked by its triple-barrier label, with one entry's barriers.
# Run by devtools/build_gallery.py with docs/notebooks as the working directory.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import Resample, TripleBarrier

h, u = 10, 0.0004
trades = pd.read_csv("data/deribit_btc_perp_6h.csv")
close, _ = Resample(freq=60_000, agg="last")(
    trades["price"].to_numpy(), trades["timestamp"].to_numpy())
label = TripleBarrier(horizon=h, upper=u, lower=u)(close)[:, 0]
label = np.r_[label[h:], np.full(h, np.nan)]  # move each label to its entry

c, lab = close[:110], label[:110]
x = np.arange(len(c))

fig, ax = plt.subplots(figsize=(6.4, 4))
e = 52  # one entry, with its upper, lower, and time barriers shaded
ax.fill_between([e, e + h], c[e] * (1 - u), c[e] * (1 + u), color="#1f77b4", alpha=0.15)
ax.hlines([c[e] * (1 + u), c[e] * (1 - u)], e, e + h, colors=["#2ca02c", "#d62728"], lw=2)
ax.plot(x, c, color="0.45", lw=1)
for value, color in ((0, "0.75"), (1, "#2ca02c"), (-1, "#d62728")):
    m = lab == value
    ax.plot(x[m], c[m], "o", ms=4.5, color=color)
