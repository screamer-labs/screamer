---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Measure tail risk with rolling CVaR and downside deviation (expected shortfall)

`RollingCVaR` gives the average loss on the worst days of a trailing window
(expected shortfall), `RollingDownsideDeviation` the spread of the returns
below zero, and `RollingMaxDrawdown` the worst fall from a peak inside the
window.

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
from screamer import (Return, RollingCVaR, RollingDownsideDeviation,
                      RollingMaxDrawdown)

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
close = df.pivot(index="date", columns="symbol")["close"][["BTC", "ETH"]]
r = Return()(close)                                  # daily simple returns

risk = {"CVaR 5% (daily)": RollingCVaR(window_size=365, alpha=0.05)(r),
        "downside deviation (daily)": RollingDownsideDeviation(window_size=365)(r),
        "max drawdown": RollingMaxDrawdown(window_size=365)(close)}
risk = {name: pd.DataFrame(x, index=close.index, columns=close.columns)
        for name, x in risk.items()}
pd.DataFrame({name: x.iloc[-1] for name, x in risk.items()}).round(3)
```

```{code-cell} ipython3
fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
for ax, (name, x) in zip(axes, risk.items()):
    x.plot(ax=ax, lw=0.9, color=["darkorange", "steelblue"], legend=False)
    ax.set_ylabel(name, fontsize=8)
axes[0].legend(loc="upper right", fontsize=8)
axes[-1].set_xlabel("")
plt.tight_layout()
```

The data is daily Binance closes of BTC and ETH from January 2022 to
September 2026. Each function treats each column as its own series and returns
an array with one column per coin, which the page puts back in a DataFrame.
The window is the last 365 days, and every measure is NaN until the window is
full.

`RollingCVaR` sorts the window's returns and averages the worst `alpha` share
of them, here the worst 5%, or 18 of 365 days, and reports that average loss
as a positive number. It is also called the expected shortfall. The
value-at-risk is the return at the edge of that tail; CVaR is the average
beyond it, so it also counts how bad the worst days were.
`RollingDownsideDeviation` is the root mean square of the returns below a
minimum acceptable return `mar`, 0 by default, with the other returns counted
as 0; it is the denominator of the Sortino ratio. `RollingMaxDrawdown` takes
prices, not returns, and gives the worst peak-to-trough fall inside the
window as a negative fraction.

Over the last year of the sample, ETH lost on average 7.6% on its worst 5% of
days and BTC 5.3%. All three measures were at their worst in the first full
window, which holds 2022, the year of the largest falls, and ETH's rose again
in 2025.

## Go deeper

- [`RollingCVaR`](../../functions_fin/RollingCVaR.md): the tail size and the link to value-at-risk.
- [`RollingDownsideDeviation`](../../functions_fin/RollingDownsideDeviation.md): `mar` and the output before the window is full.
- [`RollingMaxDrawdown`](../../functions_fin/RollingMaxDrawdown.md), [`MaxDrawdown`](../../functions_fin/MaxDrawdown.md): the worst fall in a window or since the start.
- [Compute a rolling Sharpe, Sortino and Calmar ratio](rolling-ratios.md): return per unit of these risks.
- [Track drawdown and distance to liquidation live](../backtest/drawdown-liquidation.md): drawdown of a strategy's equity, bar by bar.
