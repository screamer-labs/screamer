---
name: RollingMultiRegression
title: Rolling multiple regression
implementation_family: fin
topics:
- regression
- supervised
tags:
- regression
- ols
- least squares
- multiple regression
- har
- realized volatility
- factor model
- hedge ratio
- forecast
- r-squared
short: Rolling least-squares fit of y on k regressors, with the one-step-ahead prediction made
  before each row enters, R-squared, and the coefficients.
inputs: 4
outputs: 7
parameters:
- name: window_size
  type: int
  default: 250
  required: true
  min: 2
  description: Number of most recent rows in the fit. At least n_regressors + 1, plus 1 with an
    intercept, so the fit keeps a residual degree of freedom.
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
- ExpandingMultiRegression
- EwMultiRegression
- RollingLinearRegression
- RollingBeta
- BayesianRegression
- KalmanRegression
references:
- 'Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of
  Financial Econometrics*, 7(2), 174-196.'
---

# `RollingMultiRegression`

## Description

`RollingMultiRegression` fits `y = b0 + b1 x1 + ... + bk xk + e` by ordinary least squares over
the last `window_size` rows. It takes `k + 1` inputs, `y` first and then the `k = n_regressors`
regressors, and returns per row:

| Output | Meaning |
|---|---|
| `pred` | `b0 + b1 x1 + ... + bk xk` at this row, with the coefficients fitted on the rows before it |
| `resid` | `y - pred` |
| `r2` | R-squared of the fit after this row enters the window |
| `b0` | the intercept after this row enters (only with `intercept=True`) |
| `b1` ... `bk` | the coefficients after this row enters |

`pred` is computed before the row updates the fit, so it uses no information from the row's own
`y`: it is an out-of-sample forecast one row ahead. The coefficients and `r2` include the row. The
number of outputs depends on the settings, `k + 4` with an intercept and `k + 3` without; the
names above are those of the default `n_regressors=3`. A batch call returns them along the last
axis, in this order.

The coefficients and `r2` are `NaN` until the window is full, and `pred` until the row after.
Without an intercept, `r2` is the uncentered `1 - SSE / sum(y^2)`, as statsmodels reports it for a
model without a constant; with one it is `1 - SSE / sum((y - mean(y))^2)`. A window whose
regressors are collinear, for example one that is constant in the window when the fit has an
intercept, has no unique fit: `r2` and the coefficients are `NaN` for that row, and so is the next
`pred`. More precisely, a regressor counts as collinear when the others explain all but 1e-12 of
its variation in the window.

The operator keeps the window means and the centered cross-products of the columns, updated in
`O(k^2)` per row, and solves the normal equations by Cholesky in `O(k^3)`, so a row costs
`O(k^3)` in all. The cross-products are recomputed exactly from the stored rows every
`window_size` rows, and at once when the row leaving the window was far out of the others, so
rounding does not build up. With an intercept the fit uses the centered normal equations, so the
level of the data costs no digits: on well-conditioned regressors `b1` to `bk` match
`numpy.linalg.lstsq` on the same window to about 1e-14 at any level. Without an intercept it
solves the raw normal equations `X'X b = X'y`, formed from the centered sums at each row.
Normal equations square the condition number of the regressors, so the coefficients keep about
`16 - 2 log10(cond)` significant digits, where `cond` is the condition number of the window's
regressor matrix. On data far from zero, such as prices near 1e5 with unit moves, `cond` is large
and the coefficients keep only a few digits; prefer an intercept there, or regress on differences.
For the fit over the whole history so far, use
[`ExpandingMultiRegression`](../functions_expanding/ExpandingMultiRegression.md), and for weights
that decay with age instead of a window,
[`EwMultiRegression`](../functions_ew/EwMultiRegression.md).

For a forecast `h` rows ahead, regress `y` on the regressors lagged by `h` rows, as in the example
below. The forecast made at a row is then `b0 + b1 x1 + ... + bk xk` with that row's coefficients
and its unlagged regressors. `pred` of such a fit is not that forecast: its coefficients use rows
up to the one before, whose targets lie less than `h` rows in the past.

The example fits the HAR model (Corsi, 2009), which predicts realized variance from its averages
over the last day, week, and month. The target is the log of the average daily variance over the
next 30 days, so each row pairs the 30-day log variance up to the row with the regressors of 30
days earlier.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input gives a `NaN` output at that step. The step is skipped: it does not enter the window or update the internal state. See [NaN and warmup](../nan_and_warmup.md).
<!-- NAN_FOOTNOTE_END -->

## Examples

### HAR forecast of realized volatility

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import Lag, RollingMean, RollingMultiRegression

    # Daily realized variance with HAR dynamics in its log.
    rng = np.random.default_rng(1)
    n = 2000
    lv = np.full(n, -10.0)
    for t in range(30, n):
        lv[t] = (-1.0 + 0.35 * lv[t - 1] + 0.3 * lv[t - 7:t].mean()
                 + 0.25 * lv[t - 30:t].mean() + 0.5 * rng.standard_normal())
    rv = np.exp(lv)

    rv1, rv7, rv30 = np.log(rv), np.log(RollingMean(7)(rv)), np.log(RollingMean(30)(rv))
    h = 30
    y = rv30                                        # log RV over the 30 days up to t
    x = [Lag(h)(v) for v in (rv1, rv7, rv30)]       # the regressors 30 days earlier
    out = RollingMultiRegression(window_size=500, n_regressors=3)(y, *x)
    b0, b = out[:, 3], out[:, 4:]

    # Made at t for the 30 days after t, from today's coefficients and regressors.
    forecast = b0 + (b * np.c_[rv1, rv7, rv30]).sum(axis=1)
    forecast_due = np.r_[np.full(h, np.nan), forecast[:-h]]   # lined up with the realized value

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        row_heights=[0.6, 0.4])
    fig.add_trace(go.Scatter(y=y, name="realized log RV, 30 days",
                             line=dict(color="gray")), row=1, col=1)
    fig.add_trace(go.Scatter(y=forecast_due, name="forecast made 30 days before",
                             line=dict(color="crimson")), row=1, col=1)
    for j, name in enumerate(["day", "week", "month"]):
        fig.add_trace(go.Scatter(y=b[:, j], name=f"b, {name}"), row=2, col=1)
    fig.update_yaxes(title_text="log variance", row=1, col=1)
    fig.update_yaxes(title_text="coefficient", row=2, col=1)
    fig.update_layout(title="HAR forecast with RollingMultiRegression over 500 days",
                      xaxis2_title="day",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.show()
```

<!-- HELP_END -->
