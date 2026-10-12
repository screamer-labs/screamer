---
name: RollingOU
title: Rolling Ornstein-Uhlenbeck fit
implementation_family: rolling
topics:
- regression
- pairs
tags:
- ou
- ornstein-uhlenbeck
- mean-reversion
- rolling
- half-life
- mean reversion
- ornstein uhlenbeck
short: Rolling MLE fit of a mean-reverting Ornstein-Uhlenbeck process.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 3
  description: Number of most recent samples in the window.
- name: output
  type: str
  default: mrr
  enum:
  - mrr
  - mean
  - relmean
  - std
  description: 'Which fitted parameter to return: "mrr" gives the mean-reversion rate per
    step, "mean" the long-run mean, "relmean" the long-run mean minus the current value, and
    "std" the noise standard deviation.'
- name: start_policy
  type: str
  default: strict
  enum:
  - strict
  - expanding
  - zero
  description: 'Output before the window is full: "strict" gives NaN, "expanding" computes
    from the samples seen so far, and "zero" treats the window as pre-filled with zeros.'
nan_policy: ignore
see_also:
- RollingADF
- RollingHurst
- RollingZscore
- RollingLinearRegression
references:
- Uhlenbeck, G. E. and Ornstein, L. S. (1930). On the theory of the Brownian motion. *Physical
  Review*, 36(5), 823-841. doi:10.1103/PhysRev.36.823
---

# `RollingOU`

## Description

`RollingOU` fits a discrete-time Ornstein-Uhlenbeck (OU) model, a mean-reverting process, to the `window_size` most recent values and returns one of the fitted parameters. The fit is refreshed at every step, so the parameters can drift over time.

The discretized form of the OU model is:

$$
x[i] = (1 - \text{mrr}) \cdot x[i-1] + \text{mrr} \cdot \mu + \sigma \cdot \epsilon
$$

where $ x[i] $ is the current value, $ x[i-1] $ is the previous value, $ \mu $ is the mean, $ \text{mrr} $ is the mean reversion rate, $ \sigma $ is the standard deviation of the noise, and $ \epsilon \sim \mathcal{N}(0,1) $ is a standard normal random variable. The parameter `mrr` controls the rate and nature of mean reversion:

- **0 < mrr < 1**: Values tend to revert to the mean at a rate determined by mrr, with a higher mrr closer to 1 indicating faster mean reversion.
- **mrr < 0**: Values display mean repulsion, where deviations from the mean amplify rather than attenuate.
- **mrr > 1**: Oscillatory behavior or instability arises, as the model overshoots the mean.

The `output` parameter selects $\text{mrr}$, $\mu$ (`"mean"`), $\mu - x[i]$ (`"relmean"`, the distance from the current value to the mean) or $\sigma$ (`"std"`). A larger `window_size` gives smoother estimates that respond more slowly to changes. The half-life of a deviation from the mean, in steps, is $\ln 2 / -\ln(1 - \text{mrr})$ for $0 < \text{mrr} < 1$.

The fit is ordinary least squares of $x[i] = a + b \, x[i-1] + e[i]$ over the $m = $ `window_size` $- 1$ consecutive pairs in the window, with $\text{mrr} = 1 - b$, $\mu = a / (1 - b)$ and $\sigma$ the residual standard deviation with $m - 2$ degrees of freedom. Adding a constant to the input adds it to `"mean"` and leaves the other outputs unchanged, and the pair sums are taken about an origin near the data, so the fit stays accurate on price levels (see [Accuracy at large levels](../conventions.md#accuracy-at-large-levels)). `window_size` must be at least 3, since one pair has no slope. `"std"` needs at least 3 pairs, so it is `NaN` for `window_size=3`, where the residual has no degrees of freedom. With `start_policy="expanding"`, `"mrr"`, `"mean"` and `"relmean"` start at the third value (two pairs) and `"std"` at the fourth.


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
    from plotly.subplots import make_subplots
    from screamer import RollingOU

    # Generate synthetic data with cumulative noise
    np.random.seed(0)
    data = np.cumsum(np.random.normal(0, 1, 500))

    # Define a rolling OU model with window size of 50
    rolling_ou_mrr = RollingOU(window_size=50, output="mrr")
    rolling_ou_mean = RollingOU(window_size=50, output="mean")
    rolling_ou_std = RollingOU(window_size=50, output="std")

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

