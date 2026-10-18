---
name: ExpandingOLS
title: Expanding multiple regression
implementation_family: expanding
topics:
- regression
- supervised
tags:
- regression
- ols
- least squares
- multiple regression
- expanding
- forecast
- r-squared
short: Least-squares fit of y on k regressors over every row so far, with the one-step-ahead
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
- RollingOLS
- ExpandingSlope
- BayesianRegression
- KalmanRegression
---

# `ExpandingOLS`

## Description

`ExpandingOLS` fits `y = b0 + b1 x1 + ... + bk xk + e` by ordinary least squares over every row
seen so far. It is the whole-history counterpart of [`RollingOLS`](../functions_fin/RollingOLS.md),
with the same inputs and outputs. It takes `y` and then the `k = n_regressors` regressors, and
returns per row:

| Output | Meaning |
|---|---|
| `pred` | `b0 + b1 x1 + ... + bk xk` at this row, with the coefficients fitted on the rows before it |
| `resid` | `y - pred` |
| `r2` | R-squared of the fit after this row enters |
| `b0` | the intercept after this row enters (only with `intercept=True`) |
| `b1` ... `bk` | the coefficients after this row enters |

`pred` is an out-of-sample forecast one row ahead, since the row's own `y` has not entered the
fit yet. There are `k + 4` outputs with an intercept and `k + 3` without; the names above are those
of the default `n_regressors=3`.

The first fit needs `p + 1` rows, with `p = k + 1` coefficients with an intercept and `p = k`
without, so that one residual degree of freedom is left; `r2` and the coefficients are `NaN` before
that, and `pred` until the row after. `r2`, collinear regressors (a `NaN` fit), the uncentered
`r2` without an intercept, and the accuracy follow `RollingOLS`: with an intercept the level of the data costs no digits, and without one they keep
about `16 - 2 log10(cond)` significant digits, `cond` being the condition number of the regressor
matrix. Each row costs `O(k^2)` to update the means and centered cross-products (by Welford's
method) and `O(k^3)` to solve, so `O(k^3)` in all, and memory does not grow with the number of rows.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input gives a `NaN` output at that step. The step is skipped: it does not enter the window or update the internal state. See [NaN and warmup](../nan_and_warmup.md).
<!-- NAN_FOOTNOTE_END -->

## Examples

### Converging coefficients

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import ExpandingOLS

    rng = np.random.default_rng(0)
    n = 400
    x = rng.standard_normal((n, 2))
    y = 0.5 + 1.5 * x[:, 0] - 0.8 * x[:, 1] + 0.5 * rng.standard_normal(n)

    out = ExpandingOLS(n_regressors=2)(y, x[:, 0], x[:, 1])
    resid, b = out[:, 1], out[:, 3:]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    for j, (name, true) in enumerate([("b0", 0.5), ("b1", 1.5), ("b2", -0.8)]):
        fig.add_trace(go.Scatter(y=b[:, j], name=name), row=1, col=1)
        fig.add_hline(y=true, line=dict(color="gray", dash="dot", width=1), row=1, col=1)
    fig.add_trace(go.Scatter(y=resid, name="resid", mode="markers",
                             marker=dict(size=3, color="gray")), row=2, col=1)
    fig.update_yaxes(title_text="coefficient", row=1, col=1)
    fig.update_yaxes(title_text="residual", row=2, col=1)
    fig.update_layout(title="ExpandingOLS coefficients settle on the true values",
                      xaxis2_title="row", margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.show()
```

<!-- HELP_END -->
