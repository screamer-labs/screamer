---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Test whether a strategy's timing beats chance

`position_test` replays randomly timed copies of a strategy's trades through
the same backtest engine and returns the share that do at least as well as the
real timing, a p-value; `probabilistic_sharpe` gives the probability that the
true Sharpe ratio is above zero.

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
from screamer import (EwMean, Sign, BacktestOHLCTarget,
                      position_test, probabilistic_sharpe)

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
btc = df[df["symbol"] == "BTC"].set_index("date")
o, h, l, c = (btc[k] for k in ["open", "high", "low", "close"])

signal = Sign()(EwMean(span=20)(c) - EwMean(span=100)(c))
engine = BacktestOHLCTarget(taker_fee=1e-3, slippage=5e-4)

test = position_test(engine, signal, o, h, l, c, n=999, periods_per_year=365)
prob = probabilistic_sharpe(engine(signal, o, h, l, c))
print(f"Sharpe {test.observed:.2f}, p-value {test.p_value:.3f}, "
      f"P(true Sharpe > 0) {prob:.2f}")
```

The rule is the one of [Backtest a signal on OHLCV bars](../backtest/ohlcv.md):
long one BTC while the 20-day average of the close is above the 100-day one,
short while it is below, on daily Binance BTC/USDT bars from January 2022 to
September 2026, with a 0.1% taker fee and 0.05% slippage. Each of the 999
copies rotates the whole signal by a random offset, so it keeps the number,
sides, and lengths of the trades and changes only when they happen.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.hist(test.null, bins=40, color="lightsteelblue", label="randomly timed copies")
ax.axvline(test.observed, color="crimson", lw=2,
           label=f"actual timing, p = {test.p_value:.3f}")
ax.set_xlabel("annualized Sharpe ratio")
ax.legend(fontsize=8)
plt.tight_layout()
```

The rule's Sharpe ratio of 0.40 sits inside the spread of the copies: 15% of
them do at least as well, so this sample does not separate the timing from
luck. The probability of 0.81 that the true Sharpe ratio is above zero answers
a different question, whether the rule makes money after costs. It uses the
returns of this one run and their skewness and kurtosis, with no randomized
copies.

## Go deeper

- [Check that a result holds across nearby parameters](parameter-grid.md): a p-value for the best of many variants.
- [Is my strategy's timing real?](../../notebooks/20-is-the-timing-real.ipynb): every report statistic of the copies, and shifting versus shuffling segments.
- [I tuned my parameters. Is the result real?](../../notebooks/21-is-the-tuned-result-real.ipynb): rerun a whole parameter search on permuted markets with `strategy_test`.
- [`position_test`](../../functions_fin/position_test.md), [`probabilistic_sharpe`](../../functions_fin/probabilistic_sharpe.md): methods, metrics, and the null hypothesis.
