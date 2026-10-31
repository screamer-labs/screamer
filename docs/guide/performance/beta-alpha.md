---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Estimate beta and alpha against a benchmark

`RollingBeta` and `RollingAlpha` regress an asset's returns on a benchmark's
over a trailing window and give the slope (beta) and the intercept (alpha);
`EwBeta` weights recent days more instead of using a window, and
`RollingTLSBeta` gives a slope that treats both series alike.

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
from screamer import Return, RollingBeta, RollingAlpha, EwBeta, RollingTLSBeta

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
close = df.pivot(index="date", columns="symbol")["close"]
sol, btc = Return()(close["SOL"]), Return()(close["BTC"])   # daily returns

fit = pd.DataFrame({
    "beta, 90-day window": RollingBeta(window_size=90)(sol, btc),
    "beta, exponential (span 90)": EwBeta(span=90)(sol, btc),
    "total least squares, 90 days": RollingTLSBeta(window_size=90)(sol, btc),
    "alpha, annualized": 365 * RollingAlpha(window_size=90)(sol, btc),
}, index=close.index)
fit.iloc[-1].round(2)
```

```{code-cell} ipython3
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 5), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
fit.iloc[:, :3].plot(ax=ax0, lw=0.9, color=["steelblue", "darkorange", "0.5"])
ax0.axhline(1, color="k", lw=0.5)
ax0.set_ylabel("beta of SOL to BTC")
ax0.legend(fontsize=8, loc="upper left")
fit.iloc[:, 3].plot(ax=ax1, lw=0.9, color="seagreen")
ax1.axhline(0, color="k", lw=0.5)
ax1.set_ylabel("alpha (per year)")
ax1.set_xlabel("")
plt.tight_layout()
```

The asset is SOL and the benchmark is BTC, with daily simple returns of
Binance closes. The first input is the asset and the second the benchmark, so
`RollingBeta(90)(sol, btc)` is the least-squares slope of SOL's returns on
BTC's over the last 90 days. A beta of 1.2 means SOL moved 1.2% for each 1%
move of BTC, on average. `RollingAlpha` is the intercept of the same fit, the
average daily return SOL had beyond what its beta explains; the page
multiplies it by 365 to read it per year.

`EwBeta(span=90)` fits the same slope with weights that fall off with age, so
the estimate moves smoothly and has no window edge, where a large day drops
out all at once. `RollingTLSBeta` minimizes the distance of the points to the
line instead of the vertical distance, so the slope of SOL on BTC is exactly
one over the slope of BTC on SOL. That symmetry suits a hedge ratio between
two legs of a pair; for a beta to a benchmark, ordinary least squares is the
standard. Total least squares gives a larger slope here: when the two series
are not perfectly correlated, its slope lies between the least-squares slope
of SOL on BTC and one over the slope of BTC on SOL.

The beta of SOL to BTC was above 1 almost throughout the sample. The alpha swings
far above and below zero: 90 days hold too little data to estimate an
average return, so a large alpha in one window says little about the next.

## Go deeper

- [`RollingBeta`](../../functions_fin/RollingBeta.md), [`RollingAlpha`](../../functions_fin/RollingAlpha.md): the formulas and the order of the inputs.
- [`EwBeta`](../../functions_ew/EwBeta.md): `span`, `halflife`, and when the estimate is NaN.
- [`RollingTLSBeta`](../../functions_fin/RollingTLSBeta.md): total least squares for hedge ratios.
- [Regress returns on several factors in a moving window (rolling multi-factor regression)](factor-regression.md): more than one benchmark at once.
- [Trade a pair](../strategies/pairs.md): a hedge ratio in use.
