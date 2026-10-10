# Thumbnail for the "Microstructure" capability: BTC price, and order flow from
# the venue's aggressor side next to the flow inferred by the tick rule.
# Run by devtools/build_gallery.py with docs/notebooks as the working directory.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import TickRuleSign, EwMean

trades = pd.read_csv("data/deribit_btc_perp_6h.csv")
price = trades["price"].to_numpy()
signed = trades["volume"].to_numpy()
inferred = TickRuleSign()(price) * np.abs(signed)
true_flow = EwMean(span=100)(signed)
tick_flow = EwMean(span=100)(inferred)

skip = 300  # leave out the start, where the averages are still settling
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(6.4, 4), sharex=True)
ax0.plot(price[skip:], color="0.3", lw=0.8)
ax1.plot(true_flow[skip:], color="seagreen", lw=1)
ax1.plot(tick_flow[skip:], color="darkorange", lw=1)
ax1.axhline(0, color="k", lw=0.5)
