---
name: KalmanFilter
title: Scalar Kalman filter
implementation_family: signal
topics:
- signal-processing
- smoothing
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

## Examples

### Usage example

```python
import numpy as np
from screamer import KalmanFilter

# Noisy observations of a roughly-constant value.
truth = 5.0
obs = truth + np.random.normal(0, 1.0, 200)

# Trust the model more than the measurements (slow / smooth).
kf = KalmanFilter(process_var=0.001, observation_var=1.0)
smoothed = kf(obs)

# Edge cases:
# - process_var -> 0 collapses to the running mean
# - process_var -> infinity collapses to "output the latest measurement"
```

<!-- HELP_END -->

## Reference

Validated in `tests/test_signal.py` against the limiting cases (extreme noise ratios) plus a steady-state convergence test (zero process variance + constant input). No canonical third-party reference for a scalar Kalman filter with this exact API.
