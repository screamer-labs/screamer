---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Measure how predictable a series is (permutation entropy)

`RollingPermutationEntropy` counts how often each up-and-down pattern occurs in
a trailing window and returns their entropy: 1 when every pattern is equally
common, as in noise, and lower when some patterns repeat.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import LogReturn, RollingPermutationEntropy, permute_returns

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
df = df[df["symbol"] == "BTC"].set_index("date")
entropy = RollingPermutationEntropy(window_size=100, order=3)

df["entropy"] = entropy(LogReturn()(df["close"]))
# The same returns in 200 random orders: the values that chance alone gives
shuffled = permute_returns(df["close"].to_numpy(), n=200, seed=1)
noise = entropy(LogReturn()(shuffled))
df["noise_5%"], df["noise_95%"] = np.percentile(noise, [5, 95], axis=1)

below = (df["entropy"] < df["noise_5%"])[df["entropy"].notna()]
print(f"days below the 5% noise level: {below.mean():.0%}")
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.fill_between(df.index, df["noise_5%"], df["noise_95%"], color="0.85",
                label="shuffled returns, 5% to 95%")
ax.plot(df.index, df["entropy"], color="steelblue",
        label="RollingPermutationEntropy(100, order=3)")
ax.set_ylabel("BTC return entropy")
ax.legend(loc="lower left", fontsize=8)
plt.tight_layout()
```

With `order=3`, each pattern is the ranking of three consecutive daily returns,
one of six. Over a 100-day window even random returns give a value a little
below 1, so the shaded band shows the range of the same returns shuffled into
random orders. BTC's entropy lies below that band on 14% of the days, more than
the 5% chance gives, in stretches such as early 2024, late 2025 and mid-2026,
when some orderings of returns were more common than others. A low value says
that the order of moves has a pattern a model could use; it says nothing about
the direction.

`order` sets the pattern length, from 2 to 7, and `delay` the steps between
its values. Use returns, not prices: on a random walk, three rising or three
falling prices are each twice as common as each of the other four patterns, so
its prices score about 0.97 in a long window.

## Go deeper

- [`RollingPermutationEntropy`](../../functions_rolling/RollingPermutationEntropy.md): the definition, ties, and the `order` and `delay` parameters.
- [`permute_returns`](../../functions_fin/permute_returns.md): the shuffled copies used as the noise level.
- [Test whether a series reverts with the variance ratio](../mean-reversion/variance-ratio.md): another test of whether returns are independent.
