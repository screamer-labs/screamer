---
name: RollingOU
title: Rolling Ornstein-Uhlenbeck fit
implementation_family: rolling
topics:
- regression
tags:
- ou
- ornstein-uhlenbeck
- mean-reversion
- rolling
short: Rolling MLE fit of a mean-reverting Ornstein-Uhlenbeck process.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Trailing-window length.
- name: output
  type: int|null
  default: null
  enum:
  - 0
  - 1
  - 2
  - null
  description: 'Which fitted parameter to return: 0=mu, 1=theta, 2=sigma. None returns
    all three (1->3).'
- name: start_policy
  type: str
  default: strict
  enum:
  - strict
  - expanding
  - zero
  description: Warmup behaviour.
---

# `RollingOU`

## Description

`RollingOU` applies a rolling discrete-time Ornstein-Uhlenbeck (OU) model to a data series within a specified window. The OU model, commonly used to model mean-reverting processes, estimates the rate of reversion to the mean and the variance of the noise in each window. As a rolling estimator, `RollingOU` recalculates model parameters at each step based on a sliding window, making it suitable for non-stationary time series where parameters may evolve over time.

The discretized form of the OU model is:

$$
x[i] = \mu - (x[i-1] - \mu) \cdot \text{mrr} + \sigma \cdot \epsilon
$$

where $ x[i] $ is the current value, $ x[i-1] $ is the previous value, $ \mu $ is the mean, $ \text{mrr} $ is the mean reversion rate, $ \sigma $ is the standard deviation of the noise, and $ \epsilon \sim \mathcal{N}(0,1) $ is a standard normal random variable. The parameter `mrr` controls the rate and nature of mean reversion:

- **0 < mrr < 1**: Values tend to revert to the mean at a rate determined by mrr, with a higher mrr closer to 1 indicating faster mean reversion.
- **mrr < 0**: Values display mean repulsion, where deviations from the mean amplify rather than attenuate.
- **mrr > 1**: Oscillatory behavior or instability arises, as the model overshoots the mean.

### Parameters

**`window_size`** *(int)*: The size of the rolling window used to fit the OU model. A larger window provides a smoother estimate but may be less responsive to rapid changes.

**`output`** *(int)*: Specifies the output type:
- **0**: Returns the mean reversion rate (mrr).
- **1**: Returns the estimated mean ($\mu$).
- **2**: Returns the relative mean ($\mu^\prime = \mu - x[i]$), the mean relative to the last value.
- **3**: Returns the estimated standard deviation of the noise ($\sigma$).

- **`start_policy`**: Defines how the function handles the initial phase when fewer than `window_size` data points are available. This parameter accepts one of the following three values:
  - `"strict"`: Returns `NaN` for all calculations until `window_size` elements have been processed.
  - `"expanding"`: Adapts the computation by dynamically reducing the window size to include all available data, starting from a single point and growing until `window_size` is reached.
  - `"zero"`: Simulates a full initial window of zeros, effectively pre-filling the data stream with `window_size` zeros before processing the actual input.

## Examples

### Usage example

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import RollingOU

    # Generate synthetic data with cumulative noise
    np.random.seed(0)
    data = np.cumsum(np.random.normal(0, 1, 500))

    # Define a rolling OU model with window size of 50
    rolling_ou_mrr = RollingOU(window_size=50, output=0)
    rolling_ou_mean = RollingOU(window_size=50, output=1)
    rolling_ou_std = RollingOU(window_size=50, output=3)

    # Apply RollingOU on the data for each output type
    mrr_estimates = [rolling_ou_mrr(d) for d in data]
    mean_estimates = [rolling_ou_mean(d) for d in data]
    std_estimates = [rolling_ou_std(d) for d in data]

    # Create subplots with original and estimated parameters
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                        row_heights=[1/4, 1/4, 1/4, 1/4], vertical_spacing=0.05)

    fig.add_trace(go.Scatter(y=data, mode='lines', name='Original Data'), row=1, col=1)
    fig.add_trace(go.Scatter(y=mrr_estimates, mode='lines', name='Mean Reversion Rate (mrr)'), row=2, col=1)
    fig.add_trace(go.Scatter(y=mean_estimates, mode='lines', name='Estimated Mean (μ)', line=dict(color='green')), row=3, col=1)
    fig.add_trace(go.Scatter(y=std_estimates, mode='lines', name='Noise Std Dev (σ)', line=dict(color='purple')), row=4, col=1)

    fig.update_layout(
        title="Rolling Discrete-Time Ornstein-Uhlenbeck Model Parameters",
        xaxis_title="Index",
        yaxis=dict(title="Original Data"),
        yaxis2=dict(title="mrr"),
        yaxis3=dict(title="mean"),
        yaxis4=dict(title="noise"),
        margin=dict(l=20, r=20, t=120, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)        
    )

    fig.show()
```

<!-- HELP_END -->

