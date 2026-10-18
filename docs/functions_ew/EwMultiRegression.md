---
name: EwMultiRegression
title: Exponentially-weighted multiple regression
implementation_family: ew
topics:
- regression
- supervised
tags:
- ew
- exponential
- regression
- least squares
- weighted least squares
- multiple regression
- factor model
- hedge ratio
- forecast
- r-squared
short: Exponentially weighted least-squares fit of y on k regressors, with the one-step-ahead
  prediction made before each row enters, R-squared, and the coefficients.
inputs: 4
outputs: 7
parameters:
- name: n_regressors
  type: int
  default: 3
  required: true
  min: 1
  max: 16
  description: Number of regressors k. The operator takes k + 1 inputs, y first, and returns
    k + 3 outputs, plus 1 with an intercept.
- name: intercept
  type: bool
  default: true
  description: Fit an intercept b0. Without one, the line goes through the origin and r2 is the
    uncentered R-squared.
- name: com
  type: float
  default: null
  description: Center of mass c, giving alpha = 1 / (1 + c). Must be positive. Give exactly
    one of com, span, halflife and alpha.
- name: span
  type: float
  default: 100.0
  description: Span s in rows, giving alpha = 2 / (s + 1). Must be greater than 1. Give
    exactly one of com, span, halflife and alpha. The default shown is an example value; the
    constructor has no default.
- name: halflife
  type: float
  default: null
  description: Half-life h in rows, giving alpha = 1 - exp(-ln(2) / h). Must be positive.
    Give exactly one of com, span, halflife and alpha.
- name: alpha
  type: float
  default: null
  description: Smoothing factor, with 0 < alpha < 1; a larger alpha gives recent rows more
    weight. Give exactly one of com, span, halflife and alpha.
nan_policy: ignore
output_names:
- pred
- resid
- r2
- b0
- b1
- b2
- b3
see_also:
- RollingMultiRegression
- ExpandingMultiRegression
- EwBeta
- RollingLinearRegression
- BayesianRegression
- KalmanRegression
---

# `EwMultiRegression`

## Description

`EwMultiRegression` fits `y = b0 + b1 x1 + ... + bk xk + e` by weighted least squares over every
row seen so far, with weights that decay exponentially with age: the row `i` rows back has weight
`(1 - alpha)^i`. The fit follows coefficients that change over time without the hard cutoff of a
window. It is the exponentially weighted counterpart of
[`RollingMultiRegression`](../functions_fin/RollingMultiRegression.md) and
[`ExpandingMultiRegression`](../functions_expanding/ExpandingMultiRegression.md), with the same
inputs and outputs. It takes `y` and then the `k = n_regressors` regressors, and returns per row:

| Output | Meaning |
|---|---|
| `pred` | `b0 + b1 x1 + ... + bk xk` at this row, with the coefficients fitted on the rows before it |
| `resid` | `y - pred` |
| `r2` | weighted R-squared of the fit after this row enters |
| `b0` | the intercept after this row enters (only with `intercept=True`) |
| `b1` ... `bk` | the coefficients after this row enters |

The smoothing factor `alpha` is set by exactly one of `com`, `span`, `halflife` or `alpha`, as in
[`EwBeta`](EwBeta.md). The weights are those of pandas `ewm(adjust=True)`, so with one regressor
and an intercept `b1` equals `EwBeta(y, x)`.

`pred` is an out-of-sample forecast one row ahead, since the row's own `y` has not entered the fit
yet. There are `k + 4` outputs with an intercept and `k + 3` without; the names above are those of
the default `n_regressors=3`.

The first fit needs `p + 1` rows, with `p = k + 1` coefficients with an intercept and `p = k`
without; `r2` and the coefficients are `NaN` before that, and `pred` until the row after. With an
intercept `r2` is `1 - SSE / sum(w (y - mean_w(y))^2)`, with `SSE = sum(w e^2)` and `mean_w` the
weighted mean; without one it is the uncentered `1 - SSE / sum(w y^2)`. Collinear regressors give a
`NaN` fit. A regressor that stops varying makes the fit `NaN` once the weight of its earlier,
varying rows has decayed, after roughly `45 / alpha` rows (with an intercept, or
without one when the regressor stays at 0). The test compares the regressor's variance with a
peak of it that decays eight times slower than the weights, so a single extreme value can also give
`NaN` for a while: a tick of 1e10 in returns of spread 0.01 does for up to about 1000 rows at
`alpha = 0.1`, a tick of 1e8 does not. The accuracy is that of `RollingMultiRegression`: with an intercept the level of
the data costs no digits, and without one the coefficients keep about `16 - 2 log10(cond)`
significant digits, `cond` being the condition number of the weighted regressor matrix.

Each row scales the total weight and the centered cross-products by `1 - alpha`, adds the row to
the weighted means and cross-products in `O(k^2)`, and solves the normal equations by Cholesky in
`O(k^3)`. Memory does not grow with the number of rows. An infinite input stays in the sums with
a positive weight, so the fit is `NaN` from then on, as in `ExpandingMultiRegression`.

The example regresses an asset's returns on two factor returns whose loadings change halfway.
The fit with a half-life of 60 rows moves to the new loadings within a few half-lives.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input gives a `NaN` output at that step. The step is skipped: it does not enter the window or update the internal state. See [NaN and warmup](../nan_and_warmup.md).
<!-- NAN_FOOTNOTE_END -->

## Examples

### Factor loadings that change

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from screamer import EwMultiRegression

    rng = np.random.default_rng(2)
    n = 2000
    f = 0.01 * rng.standard_normal((n, 2))          # two factor returns
    b1 = np.where(np.arange(n) < n // 2, 1.2, 0.6)
    b2 = np.where(np.arange(n) < n // 2, -0.3, 0.5)
    r = 0.0002 + b1 * f[:, 0] + b2 * f[:, 1] + 0.005 * rng.standard_normal(n)

    out = EwMultiRegression(n_regressors=2, halflife=60)(r, f[:, 0], f[:, 1])

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=out[:, 4], name="b1, fitted"))
    fig.add_trace(go.Scatter(y=out[:, 5], name="b2, fitted"))
    fig.add_trace(go.Scatter(y=b1, name="b1, true", line=dict(color="gray", dash="dot")))
    fig.add_trace(go.Scatter(y=b2, name="b2, true", line=dict(color="gray", dash="dash")))
    fig.update_layout(title="EwMultiRegression loadings with a half-life of 60 rows",
                      xaxis_title="row", yaxis_title="loading",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.show()
```

<!-- HELP_END -->
