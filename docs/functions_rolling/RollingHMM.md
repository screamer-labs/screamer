---
name: RollingHMM
title: Rolling two-state hidden Markov model
implementation_family: rolling
topics:
- regime
tags:
- hidden markov model
- hmm
- regime switching
- markov switching
- baum-welch
- volatility regime
- hamilton
short: Two-state hidden Markov model of a series, refit on a trailing window and filtered forward
  each step; the probability of the high-variance (or more persistent) state and a one-step predictive
  mean and standard deviation.
inputs: 1
outputs: 3
parameters:
- name: window_size
  type: int
  default: 720
  min: 10
  description: Number of most recent values the model is fit on.
- name: refit_every
  type: int
  default: 24
  min: 1
  description: Steps between fits. Between fits the model filters each new value with fixed
    parameters.
- name: emission
  type: str
  default: gaussian
  enum:
  - gaussian
  - ar1
  description: '"gaussian": each state emits a normal value with its own mean and variance;
    "ar1": each state emits an AR(1) step x_t = c + phi x_{t-1} + e with its own c, phi, and
    variance (Hamilton''s Markov-switching AR(1)).'
- name: max_iter
  type: int
  default: 50
  min: 1
  description: Largest number of EM iterations per fit.
- name: tol
  type: float
  default: 1.0e-06
  min: 0.0
  description: A fit stops when an EM iteration raises the log-likelihood by less than this.
nan_policy: ignore
see_also:
- BayesianChangepoint
- RollingVarianceRatio
- BayesianRegression
- EwStd
output_names:
- prob
- pred_mean
- pred_std
references:
- 'Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series
  and the business cycle. *Econometrica* 57(2), 357-384.'
- 'Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech
  recognition. *Proceedings of the IEEE* 77(2), 257-286.'
---

# `RollingHMM`

## Description

`RollingHMM` is a two-state hidden Markov model of a series, normally
returns. The series is in one of two states, which persist and switch with
the probabilities of a 2 x 2 transition matrix, and each state emits values
from its own distribution:

- `emission="gaussian"`: state `k` emits `N(mu_k, sigma_k^2)`, for example a
  calm and a volatile regime of returns;
- `emission="ar1"`: state `k` emits `x_t = c_k + phi_k x_{t-1} + sigma_k e_t`,
  Hamilton's (1989) Markov-switching AR(1), for example a trending and a
  reverting regime.

The model is fit by Baum-Welch EM on the last `window_size` values, first on
the step the window fills and then every `refit_every` steps, each fit
starting from the previous one. Between fits each new value updates the state
probability with the fitted parameters (the forward filter). After each fit
the states are ordered so that state 1 is the high-variance state
(`"gaussian"`) or the state with the larger `phi` (`"ar1"`), so the
meaning of the output does not change between fits. Each output at `t` uses
values up to `t`.

It returns three columns:

0. `prob`: the filtered probability of state 1 at `t`.
1. `pred_mean`: the mean of the one-step-ahead predictive distribution, a
   mixture of the two states weighted by their probability at `t + 1`.
2. `pred_std`: the standard deviation of that mixture.

The first `window_size - 1` rows are `NaN`. The first fit starts from a
deterministic split (values below and above the median absolute value for
`"gaussian"`; `phi` at the window's lag-1 autocorrelation minus and plus 0.5
for `"ar1"`), so the result is reproducible. Variances are floored at
`1e-6` times the window variance, which keeps a fit finite on a constant
window. A fit costs `O(max_iter * window_size)`; a step between fits is
`O(1)`.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input gives a `NaN` output at that step. The step is skipped: it does not enter the window or update the internal state. See [NaN and warmup](../nan_and_warmup.md).
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import RollingHMM

    rng = np.random.default_rng(8)
    T = 3000
    state = np.zeros(T, dtype=int)
    for t in range(1, T):
        stay = 0.99 if state[t - 1] == 0 else 0.98
        state[t] = state[t - 1] if rng.random() < stay else 1 - state[t - 1]
    r = rng.standard_normal(T) * np.where(state == 1, 2.0, 0.5)
    out = RollingHMM(window_size=1000, refit_every=100)(r)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06)
    fig.add_trace(go.Scatter(y=r, name='returns', line=dict(color='lightgray')), row=1, col=1)
    fig.add_trace(go.Scatter(y=out[:, 0], name='P(volatile state)', line=dict(color='crimson')), row=2, col=1)
    fig.add_trace(go.Scatter(y=state, name='true state', line=dict(color='gray', dash='dot')), row=2, col=1)
    fig.update_layout(title='RollingHMM: the filtered probability of the volatile regime',
                      xaxis2_title='step', margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```

<!-- HELP_END -->
