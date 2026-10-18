---
name: KalmanFilter
title: Scalar Kalman filter
implementation_family: signal
topics:
- smoothing
- filtering
- supervised
tags:
- kalman
- state-space
- adaptive
- state space
- kalman filter
- adaptive smoother
short: Scalar 1-D Kalman filter for a noisy random-walk model.
inputs: 1
outputs: 1
parameters:
- name: process_var
  type: float
  default: 0.01
  min: 0.0
  description: Variance of the random-walk innovation. Larger values make the
    filter follow new measurements more closely, with a noisier output.
- name: observation_var
  type: float
  default: 1.0
  min: 0.0
  description: Variance of the measurement noise. Must be positive.
- name: initial_state
  type: float
  default: 0.0
  description: Starting state estimate x_0, used before the first observation.
- name: initial_variance
  type: float
  default: 1.0
  min: 0.0
  description: Variance P_0 of the starting state. A large value such as 1e9 makes
    the filter forget the initial state quickly. inf is a diffuse prior, so the first observation sets the state, with the observation variance.
- name: huber
  type: float
  default: .inf
  min: 0.0
  description: Outlier threshold, in standard deviations of the prediction error. An
    observation further out than this updates the state with a weight that falls with its
    distance, so it cannot pull the state far. 2 to 3 is typical. inf, the default, is the
    ordinary Kalman update. Must be positive.
nan_policy: ignore
see_also:
- KalmanRegression
- EwMean
- SuperSmoother
references:
- Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. *Journal
  of Basic Engineering*, 82(1), 35-45. doi:10.1115/1.3662552
- 'Masreliez, C. J., and Martin, R. D. (1977). Robust Bayesian estimation for the linear model
  and robustifying the Kalman filter. *IEEE Transactions on Automatic Control*, 22(3), 361-371.
  doi:10.1109/TAC.1977.1101538'
- 'Huber, P. J. (1964). Robust estimation of a location parameter. *The Annals of Mathematical
  Statistics*, 35(1), 73-101. doi:10.1214/aoms/1177703732'
---

# `KalmanFilter`

## Description

Scalar 1-D Kalman filter for the "noisy random walk" model: state and observation are scalars with unit transition and observation matrices. It works as an adaptive smoother whose responsiveness is set by the ratio of process to observation noise.

$$
\begin{aligned}
\text{predict: } & x_\text{pred} = x_{t-1},\quad P_\text{pred} = P_{t-1} + \sigma^2_p \\
\text{gain: }    & K = P_\text{pred} / (P_\text{pred} + \sigma^2_o) \\
\text{update: }  & x_t = x_\text{pred} + K \cdot (z_t - x_\text{pred}) \\
                  & P_t = (1 - K) \cdot P_\text{pred}
\end{aligned}
$$

In the steady state $K$ converges to a constant determined by $\sigma^2_p / \sigma^2_o$, at which point the filter behaves like an exponential smoother with that effective $\alpha$.

### Robust update

The filter assumes Gaussian measurement noise, so a single bad observation moves the state in proportion to its size, and the error carries forward. Setting `huber` caps that influence. With the prediction error $e = z_t - x_\text{pred}$, its predicted variance $Q = P_\text{pred} + \sigma^2_o$ and the standardized error $|e| / \sqrt{Q}$, an observation beyond `huber` standard deviations gets the weight

$$
w = \frac{\text{huber}}{|e| / \sqrt{Q}} < 1,
$$

and the gain and variance update use the observation variance $\sigma^2_o / w$ in place of $\sigma^2_o$. This is the Huber M-estimate step of the Masreliez and Martin (1977) robust filter. Observations within `huber` standard deviations update exactly as without it. A value of 2 to 3 is typical.

The protection has a cost: a genuine level shift is also treated as an outlier at first, so the filter takes longer to follow it.

## Implementation Details

Constant-time O(1) per step; no buffer, two scalar state variables (`x`, `P`).


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
