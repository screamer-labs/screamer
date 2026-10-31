---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Regress returns on several factors in a moving window (rolling multi-factor regression)

`RollingMultiRegression` fits an asset's returns on several factor returns by
least squares over a trailing window and gives, on every row, the
coefficients, the R-squared, and a one-step-ahead prediction;
`EwMultiRegression` weights recent rows more, and `ExpandingMultiRegression`
uses all rows so far.

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
from screamer import (Return, RollingMultiRegression, EwMultiRegression,
                      ExpandingMultiRegression)

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
close = df.pivot(index="date", columns="symbol")["close"]
sol, btc, eth = (Return()(close[k]) for k in ["SOL", "BTC", "ETH"])

fits = {"180-day window": RollingMultiRegression(window_size=180, n_regressors=2),
        "half-life 60 days": EwMultiRegression(n_regressors=2, halflife=60),
        "expanding": ExpandingMultiRegression(n_regressors=2)}
cols = ["pred", "resid", "r2", "alpha", "beta BTC", "beta ETH"]
out = {name: pd.DataFrame(f(sol, btc, eth), index=close.index, columns=cols)
       for name, f in fits.items()}
pd.DataFrame({name: o.iloc[-1] for name, o in out.items()}).loc[cols[2:]].round(3)
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(9, 3.5))
for (name, o), ls in zip(out.items(), ["-", "--", ":"]):
    for factor, color in [("BTC", "darkorange"), ("ETH", "steelblue")]:
        ax.plot(o.index, o[f"beta {factor}"], color=color, ls=ls, lw=0.9,
                label=f"{factor}, {name}")
ax.set_ylabel("coefficient of SOL on the factor")
ax.legend(fontsize=7, loc="upper right", ncol=3)
plt.tight_layout()
```

The model is `sol = b0 + b1 btc + b2 eth + e`, on daily simple returns of
Binance closes. Each function takes the target first and then the
`n_regressors` factors, and returns `n_regressors + 4` columns in this order:
`pred`, `resid`, `r2`, the intercept `b0` and the coefficients. The page names
them `alpha`, `beta BTC` and `beta ETH`. `pred` is the fit of the rows before
the current one applied to the current factors, so it is a forecast made
without the current return; `resid` is the return minus `pred`. The
coefficients and `r2` include the current row.

The 180-day window weighs each of its rows equally and drops the oldest at
once. The exponential fit gives a row 60 days old half the weight of today's,
and the expanding fit uses the whole history, so it barely moves by the end.
All three are NaN until they have enough rows.

BTC and ETH returns are strongly correlated, about 0.84 over the sample, so the
regression has trouble telling their effects apart: the two coefficients swing
in opposite directions in the moving fits while their sum moves less. At the
end of the sample the three fits explain 61% to 75% of the variance of SOL's
daily returns (`r2`).

## Go deeper

- [`RollingMultiRegression`](../../functions_fin/RollingMultiRegression.md): the outputs, collinear factors, and a forecast several rows ahead.
- [`EwMultiRegression`](../../functions_ew/EwMultiRegression.md), [`ExpandingMultiRegression`](../../functions_expanding/ExpandingMultiRegression.md): the weighted and whole-history fits.
- [Estimate beta and alpha against a benchmark](beta-alpha.md): one factor at a time.
- [Build a feature matrix for a model](../machine-learning/features.md): factors as model inputs.
