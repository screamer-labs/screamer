---
name: KalmanFilter
title: Scalar Kalman filter
implementation_family: signal
topics:
- smoothing
- filtering
tags:
- kalman
- state-space
- adaptive
short: Scalar 1-D Kalman filter for a noisy random-walk model.
inputs: 1
outputs: 1
parameters:
- name: process_var
  type: float
  default: 0.01
  min: 0.0
  description: Variance of the random-walk innovation (larger = more responsive).
- name: observation_var
  type: float
  default: 1.0
  min: 0.0
  description: Variance of the measurement noise.
- name: initial_state
  type: float
  default: 0.0
  description: Initial state estimate.
- name: initial_variance
  type: float
  default: 1.0
  min: 0.0
  description: Initial state variance. Set to a large value to forget the initial
    state quickly.
nan_policy: ignore
---

# `KalmanFilter`

## Description

Scalar 1-D Kalman filter for the classic "noisy random walk" model: state and observation are scalars with unit transition/observation matrices. Useful as an adaptive smoother whose responsiveness is governed by the ratio of process to observation noise.

$$
\begin{aligned}
\text{predict: } & x_\text{pred} = x_{t-1},\quad P_\text{pred} = P_{t-1} + \sigma^2_p \\
\text{gain: }    & K = P_\text{pred} / (P_\text{pred} + \sigma^2_o) \\
\text{update: }  & x_t = x_\text{pred} + K \cdot (z_t - x_\text{pred}) \\
                  & P_t = (1 - K) \cdot P_\text{pred}
\end{aligned}
$$

In the steady state $K$ converges to a constant determined by $\sigma^2_p / \sigma^2_o$, at which point the filter behaves like an exponential smoother with that effective $\alpha$.

## Parameters

- `process_var` ($\ge 0$): variance of the random-walk innovations. Larger values make the filter trust new measurements more (responsive but noisier).
- `observation_var` ($> 0$): variance of the measurement noise.
- `initial_state` (default `0.0`): seed for $x_0$.
- `initial_variance` (default `1.0`): seed for $P_0$. Pass a large value (e.g. `1e9`) to make the filter forget the initial state quickly.

## Implementation Details

Constant-time O(1) per step; no buffer, two scalar state variables (`x`, `P`).


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
    from screamer import KalmanFilter

    rng = np.random.default_rng(0)
    N = 300
    t = np.linspace(0, 6 * np.pi, N)

    # Smooth true signal: slow sine wave
    true_signal = np.sin(t)
    # Noisy observations
    obs = true_signal + rng.standard_normal(N)

    # Kalman smoother: trust model more than measurements
    kf_smooth = KalmanFilter(process_var=0.01, observation_var=1.0)
    smoothed = kf_smooth(obs)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=obs, mode='lines', name='Noisy observations',
                             line=dict(color='lightblue'), opacity=0.7))
    fig.add_trace(go.Scatter(y=true_signal, mode='lines', name='True signal',
                             line=dict(color='green', dash='dash')))
    fig.add_trace(go.Scatter(y=smoothed, mode='lines',
                             name='KalmanFilter(process_var=0.01, observation_var=1.0)',
                             line=dict(color='red')))
    fig.update_layout(
        title="Scalar Kalman filter: recovering a sine from noisy observations",
        xaxis_title="Index", yaxis_title="Value",
        margin=dict(l=20, r=20, t=80, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.show()
```

<!-- HELP_END -->

## Reference

Validated in `tests/test_signal.py` against the limiting cases (extreme noise ratios) plus a steady-state convergence test (zero process variance + constant input). No canonical third-party reference for a scalar Kalman filter with this exact API.
