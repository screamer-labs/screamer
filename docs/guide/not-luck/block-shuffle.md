---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Shuffle trades in blocks to test a strategy's timing (block randomization test)

`position_test(..., method="segments")` replays copies of a strategy in which
each trade and each flat stretch moves as one block to a random place, and
`segment_shuffle` returns those copies so you can look at them.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import (EwMean, Sign, BacktestOHLCTarget,
                      position_test, segment_shuffle)

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
btc = df[df["symbol"] == "BTC"].set_index("date")
o, h, l, c = (btc[k].to_numpy() for k in ["open", "high", "low", "close"])

target = np.maximum(Sign()(EwMean(span=20)(c) - EwMean(span=100)(c)), 0.0)
target[:100] = 0.0
engine = BacktestOHLCTarget(taker_fee=1e-3, slippage=5e-4)

for method in ["segments", "shift"]:
    test = position_test(engine, target, o, h, l, c, n=999, method=method,
                         periods_per_year=365)
    print(f"{method:8} Sharpe {test.observed:.2f}, p-value {test.p_value:.3f}")
copies = segment_shuffle(target, 2, seed=1)
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(9, 3))
for row, y in enumerate([target, copies[:, 0], copies[:, 1]]):
    ax.fill_between(btc.index, -row, -row + 0.8 * y, step="post",
                    color="crimson" if row == 0 else "steelblue")
ax.set_yticks([0.4, -0.6, -1.6], ["actual", "copy 1", "copy 2"])
ax.set_title("days long BTC", fontsize=10)
plt.tight_layout()
```

The rule is long one BTC while the 20-day exponential average of the close is
above the 100-day one and flat otherwise, flat during the first 100 days of
warmup, on daily Binance BTC/USDT bars from January 2022 to September 2026. Every
fill is at the next open and pays a 0.1% taker fee and 0.05% slippage. Each
copy holds the same nine trades with the same holding periods and the same flat
stretches between them, in a random order, so it pays the same costs and is
long on the same number of days.

The rule's Sharpe ratio of 0.57 is matched or beaten by 12% of the 999
shuffled copies, so the timing is not clearly better than chance. The copies
are long as often as the rule, so a rule that does well only because it is long
in a rising market does not pass. `method="shift"`, the default of
[Test whether a strategy's timing beats chance](timing-test.md), rotates the
whole series instead: it keeps the order of the trades and the gaps between
them and moves them together, and gives a p-value of 0.07 here. The shuffle
also breaks the order, so it removes any timing that lies in the sequence of
trades.

## Go deeper

- [Test whether a strategy's timing beats chance](timing-test.md): the same test with rotated copies.
- [Run a Monte Carlo permutation test on a whole strategy](permutation-test.md): randomize the market instead of the positions.
- [Is my strategy's timing real?](../../notebooks/20-is-the-timing-real.ipynb): shifting versus shuffling segments, and every report statistic of the copies.
- [`segment_shuffle`](../../functions_fin/segment_shuffle.md), [`position_test`](../../functions_fin/position_test.md): how runs are reordered, and the other metrics.
