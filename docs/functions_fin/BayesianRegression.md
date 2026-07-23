---
name: BayesianRegression
title: Online Bayesian univariate regression (2->4 posterior fit)
implementation_family: fin
topics:
- trend
- regression
tags:
- bayesian
- regression
- online
- pair
short: Current slope and intercept via Normal-Inverse-Gamma posterior plus causal one-step-ahead predictive mean and std.
inputs: 2
outputs: 4
parameters:
- name: com
  type: float
  default: null
  description: Center of mass (alpha = 1 / (1 + com)). Exclusive with span/halflife/alpha.
- name: span
  type: float
  default: 20.0
  description: Span (alpha = 2 / (span + 1)). Default smoothing parameter. Exclusive with com/halflife/alpha.
- name: halflife
  type: float
  default: null
  description: Halflife (alpha = 1 - 0.5^(1/halflife)). Exclusive with com/span/alpha.
- name: alpha
  type: float
  default: null
  description: Smoothing parameter directly (forgetting factor lambda = 1 - alpha). Exclusive with com/span/halflife.
- name: prior_precision
  type: float
  default: 1.0
  description: Prior precision on the regression coefficients (slope and intercept). Larger values pull coefficients toward zero more strongly.
- name: prior_sigma
  type: float
  default: 1.0
  description: Prior scale for the noise variance. Sets b0 = prior_sigma^2 * (a0 - 1) where a0 = 2.
nan_policy: ignore
---

# `BayesianRegression`

## Description

`BayesianRegression` fits a univariate linear model `y ≈ slope * x + intercept + noise`
online using a Normal-Inverse-Gamma (NIG) conjugate posterior. It is a **2-input,
4-output** operator (`FunctorBase<_, 2, 4>`). Inputs are `(y, x)`; outputs per step are
`(slope, intercept, pred_mean, pred_std)`.

At every step the operator emits a **causal, one-step-ahead prediction** for the current
`y` using the posterior mean from all previous data, then updates the posterior with the
new observation. The prediction is formed before the update, so no future information
enters the output.

| Output | Column | Description |
|---|---|---|
| `slope` | 0 | Current posterior mean slope |
| `intercept` | 1 | Current posterior mean intercept |
| `pred_mean` | 2 | Predictive mean of `y` given `x` and the prior posterior |
| `pred_std` | 3 | Predictive standard deviation (Student-t predictive, finite from step 1) |

### Bayesian update

The posterior over `(slope, intercept, sigma^2)` is NIG with parameters
`(mu, Lambda, a, b)`. The prior is set at construction:

- `a0 = 2` (shape, ensures finite prior variance)
- `b0 = prior_sigma^2 * (a0 - 1)` (scale)
- `Lambda0 = prior_precision * I_2` (precision matrix on the coefficient vector)
- `mu0 = [0, 0]` (zero prior mean)

At step `t`, given feature vector `phi = [x_t, 1]` and observation `y_t`, the
**predict-then-update** loop runs as follows.

**Predict** (before the update):

The predictive distribution is Student-t with `2 * a` degrees of freedom, mean
`mu^T phi`, and variance `(b / a) * (1 + phi^T Lambda^{-1} phi)`. For a 2x2
diagonal prior this reduces to an analytic formula with no matrix library.

**Update** (after emitting the prediction):

    Lambda_new  = lambda_ * Lambda + phi phi^T
    mu_new      = Lambda_new^{-1} (lambda_ * Lambda mu + phi y_t)
    a_new       = lambda_ * a + 0.5
    b_new       = lambda_ * b + 0.5 * (y_t - mu^T phi)^2 * lambda_det / lambda_det_new

where `lambda_det = det(Lambda)` and `lambda_det_new = det(Lambda_new)`.

### Forgetting factor

Exponential forgetting is applied by discounting the sufficient statistics by
`lambda_ = 1 - alpha` at each step. Specify exactly one of `com`, `span`,
`halflife`, or `alpha` to set the forgetting rate. With a long `span` (or small
`alpha`) the posterior accumulates evidence slowly and the confidence intervals
narrow gradually. With a short `span` the posterior forgets old data quickly,
letting the slope and intercept track a drifting relationship.

### Prior and warmup

The weak proper prior (`a0 = 2`, `prior_precision = 1.0`) ensures `pred_std` is
finite and `pred_mean` is defined from the very first sample. There is no NaN
warmup period: outputs are finite for all `t >= 0`. As evidence accumulates the
predictive interval shrinks and the posterior mean converges toward the true
coefficients.


<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
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
