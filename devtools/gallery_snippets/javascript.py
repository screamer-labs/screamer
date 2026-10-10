# Thumbnail for the "JavaScript in the browser" capability: a code panel above a
# trade trace with its moving average. Run by devtools/build_gallery.py with
# docs/notebooks as the working directory.
import pandas as pd
import matplotlib.pyplot as plt
from screamer import RollingMean

price = pd.read_csv("data/deribit_btc_perp_6h.csv")["price"].to_numpy()[:900]

fig = plt.figure(figsize=(6.4, 4), facecolor="white")
code = fig.add_axes([0.0, 0.48, 1.0, 0.52])
code.set_facecolor("#1e1e2e")
mono = dict(family="monospace", size=13)
fig.text(0.04, 0.87, 'import { ready, RollingMean }', color="#c3e88d", **mono)
fig.text(0.04, 0.78, '  from "@screamer-labs/screamer";', color="#e0e0e0", **mono)
fig.text(0.04, 0.66, 'await ready();', color="#e0e0e0", **mono)
fig.text(0.04, 0.57, 'const sma = RollingMean(50);', color="#82aaff", **mono)

ax = fig.add_axes([0.02, 0.03, 0.96, 0.42])
ax.set_facecolor("#2b2b3b")
ax.plot(price, ".", ms=1.5, color="0.6")
ax.plot(RollingMean(50)(price), color="#f7df1e", lw=2)
