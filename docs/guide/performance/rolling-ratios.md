---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Compute a rolling Sharpe, Sortino and Calmar ratio

`RollingSharpe`, `RollingSortino` and `RollingCalmar` compute the
risk-adjusted return over a trailing window of returns, so you see how a
strategy's quality changes over time; `RollingOmega` and `RollingInfoRatio`
add the Omega ratio and the information ratio against a benchmark.

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
from screamer import (EwMean, Sign, BacktestOHLCTarget, Return, RollingSharpe,
                      RollingSortino, RollingCalmar, RollingOmega, RollingInfoRatio)

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
btc = df[df["symbol"] == "BTC"].set_index("date")
o, h, l, c = (btc[k].to_numpy() for k in ["open", "high", "low", "close"])
direction = Sign()(EwMean(span=20)(c) - EwMean(span=100)(c))
direction[:100] = 0.0
out = BacktestOHLCTarget(taker_fee=1e-3, slippage=5e-4, size="weight",
                         initial_capital=1.0)(direction, o, h, l, c)
r, benchmark = Return()(1 + out[:, 0]), Return()(c)    # daily simple returns

year = dict(window_size=365, periods_per_year=365)
ratios = pd.DataFrame(
    {"Sharpe": RollingSharpe(**year)(r), "Sortino": RollingSortino(**year)(r),
     "Calmar": RollingCalmar(**year)(r), "Omega": RollingOmega(window_size=365)(r),
     "information ratio": RollingInfoRatio(**year)(r, benchmark)}, index=btc.index)
ratios.iloc[-1].round(2)
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(9, 3.5))
ratios[["Sharpe", "Sortino", "Calmar"]].plot(
    ax=ax, lw=0.9, color=["steelblue", "seagreen", "darkorange"])
ax.axhline(0, color="k", lw=0.5)
ax.set_ylabel("ratio over the last 365 days")
ax.set_xlabel("")
plt.tight_layout()
```

The returns are the daily returns of the 20/100-day trend rule of [Backtest
a signal on OHLCV bars](../backtest/ohlcv.md), long or short one times equity
on BTC, after a 0.1% fee and 0.05% slippage; `Return` turns the equity
`1 + out[:, 0]` into simple returns. Each ratio uses the last 365 daily
returns, and `periods_per_year=365` annualizes it for a market that trades
every day. The first value comes after a full window.

The Sharpe ratio divides the mean return by its standard deviation. The
Sortino ratio divides it by the downside deviation, the spread of the returns
below 0 only, so a strategy whose large moves are gains scores higher on it.
The Calmar ratio divides the annualized mean return by the worst drawdown in
the window. The Omega ratio is the sum of the gains divided by the sum of the
losses, so a value above 1 means the gains outweigh the losses. The
information ratio is the Sharpe ratio of the return minus the benchmark's, here
holding BTC. None of them subtracts a risk-free rate; subtract it from the
returns first if you want one.

The three ratios move together. They were high in the year to early 2023,
which held the trend down through 2022, and fell below zero in mid 2023 and
again in 2025: one year can look excellent and the next poor, for the same
rule. A single Sharpe ratio over the whole sample hides this.

## Go deeper

- [`RollingSharpe`](../../functions_fin/RollingSharpe.md), [`RollingSortino`](../../functions_fin/RollingSortino.md), [`RollingCalmar`](../../functions_fin/RollingCalmar.md): the formulas and the warmup.
- [`RollingOmega`](../../functions_fin/RollingOmega.md), [`RollingInfoRatio`](../../functions_fin/RollingInfoRatio.md): a threshold for Omega, and active returns.
- [Measure tail risk with rolling CVaR and downside deviation](tail-risk.md): the losses behind the ratios.
- [Correct a Sharpe ratio for the number of strategies tried (deflated Sharpe ratio)](../not-luck/deflated-sharpe.md): when the best of many is not luck.
