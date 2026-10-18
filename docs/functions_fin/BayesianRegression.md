---
name: BayesianRegression
title: Online Bayesian regression
implementation_family: fin
topics:
- trend
- regression
- supervised
tags:
- bayesian
- regression
- online
- pairs
- hedge ratio
- normal-inverse-gamma
- predictive interval
- forgetting factor
short: 'Online Bayesian linear regression with a forgetting factor: the current slope and
  intercept, and a one-step-ahead predictive mean and standard deviation.'
inputs: 2
outputs: 4
parameters:
- name: com
  type: float
  default: null
  description: Center of mass c, giving alpha = 1 / (1 + c). Must be positive. Give exactly
    one of com, span, halflife and alpha.
- name: span
  type: float
  default: 20.0
  description: Span s in samples, giving alpha = 2 / (s + 1). Must be greater than 1. Give
    exactly one of com, span, halflife and alpha. The default shown is an example value; the
    constructor has no default.
- name: halflife
  type: float
  default: null
  description: Half-life h in samples, giving alpha = 1 - exp(-ln(2) / h). Must be positive.
    Give exactly one of com, span, halflife and alpha.
- name: alpha
  type: float
  default: null
  description: Smoothing factor, with 0 < alpha < 1; a larger alpha gives recent samples more
    weight. Give exactly one of com, span, halflife and alpha.
- name: prior_precision
  type: float
  default: 1.0
  description: Prior precision of the intercept and slope, which have prior mean 0. Larger
    values pull the coefficients toward zero more strongly. Must be positive.
- name: prior_sigma
  type: float
  default: 1.0
  description: 'Prior guess of the noise standard deviation: the prior mean of the noise variance
    is prior_sigma^2. Must be positive.'
nan_policy: ignore
output_names:
- slope
- intercept
- pred_mean
- pred_std
see_also:
- KalmanRegression
- EwBeta
- RollingLinearRegression
- RollingBeta
- RollingOLS
---

# `BayesianRegression`

## Description

`BayesianRegression` fits the linear model `y = intercept + slope * x + noise`
online, with a Normal-Inverse-Gamma posterior over the intercept, the slope, and
the noise variance. The inputs are `(y, x)`.

Each step first predicts `y_t` from `x_t` with the posterior built from the data
before `t`, then updates the posterior with `(x_t, y_t)`. The prediction never
uses `y_t`, so it can be compared with `y_t` as an out-of-sample forecast. The
four output columns are:

| Column | Name | Value |
|---|---|---|
| 0 | `slope` | Posterior mean of the slope, after the update at `t` |
| 1 | `intercept` | Posterior mean of the intercept, after the update at `t` |
| 2 | `pred_mean` | Predictive mean of `y_t`, from the data before `t` |
| 3 | `pred_std` | Predictive standard deviation of `y_t` (Student-t), from the data before `t` |

### Posterior update

The prior has coefficient mean `0`, coefficient precision
`Lambda0 = prior_precision * I`, and inverse-gamma shape `a0 = 2` and scale
`b0 = prior_sigma^2 * (a0 - 1)`, so the prior mean of the noise variance is
`prior_sigma^2`. The posterior is kept as the precision `Lambda`, the vector
`eta = Lambda * mu` (with `mu` the coefficient mean), the sum of squares `s`, and
the shape `a`. With `phi = [1, x_t]` and `lambda = 1 - alpha`, each step does:

1. **Forget.** Every statistic moves toward its prior value:
   `Lambda = lambda * Lambda + (1 - lambda) * Lambda0`, and the same for `eta`,
   `s`, and `a`.
2. **Predict.** `pred_mean = mu' phi` and
   `pred_std^2 = (b / a) * (1 + phi' Lambda^-1 phi) * a / (a - 1)`, with
   `b = (s - eta' Lambda^-1 eta) / 2`. This is the variance of the Student-t
   predictive with `2a` degrees of freedom.
3. **Update.** `Lambda += phi phi'`, `eta += phi * y_t`, `s += y_t^2`, and
   `a += 1/2`. The reported `slope` and `intercept` are the new
   `mu = Lambda^-1 eta`.

### Choosing the forgetting rate

`com`, `span`, `halflife`, or `alpha` sets how fast old data fades. With a long
`span` (small `alpha`) the posterior accumulates evidence slowly and the
predictive band narrows gradually. With a short `span` it forgets old data
quickly, so the slope and intercept track a drifting relationship.

### Warmup

Because the prior is proper (`a0 = 2`, positive `prior_precision`), every
output is finite from the first sample. On the first step `pred_mean` is 0, the
prior mean.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input gives a `NaN` output at that step. The step is skipped: it does not enter the window or update the internal state. See [NaN and warmup](../nan_and_warmup.md).
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage example

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from screamer import BayesianRegression

    rng = np.random.default_rng(42)
    n = 300
    # Drifting synthetic relationship: true slope starts at 0.5, shifts to 1.5 at t=150
    x = rng.standard_normal(n)
    true_slope = np.where(np.arange(n) < 150, 0.5, 1.5)
    y = true_slope * x + rng.standard_normal(n) * 0.4

    out = BayesianRegression(span=40)(y, x)
    pred_mean = out[:, 2]
    pred_std  = out[:, 3]

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y, mode="lines", name="y (observed)",
                             line=dict(color="steelblue", width=1), opacity=0.6))
    fig.add_trace(go.Scatter(y=pred_mean, mode="lines", name="pred_mean",
                             line=dict(color="darkorange", width=2)))
    fig.add_trace(go.Scatter(
        y=pred_mean + pred_std,
        mode="lines", name="pred_mean + pred_std",
        line=dict(color="darkorange", width=0),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        y=pred_mean - pred_std,
        mode="lines", name="pred_mean +/- pred_std",
        line=dict(color="darkorange", width=0),
        fill="tonexty",
        fillcolor="rgba(255,140,0,0.15)",
    ))
    fig.update_layout(
        title="BayesianRegression: causal prediction with confidence band (span=40)",
        xaxis_title="time",
        yaxis_title="y",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.show()
```

<!-- HELP_END -->
