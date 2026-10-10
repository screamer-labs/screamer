---
name: RollingHurst
title: Rolling Hurst Exponent
implementation_family: rolling
topics:
- trend
- regression
tags:
- hurst
- long-memory
- fractal
- r/s
- anis-lloyd
- regime
- hurst exponent
- long memory
- rescaled range
- mean reversion
short: Rolling-window Hurst exponent via Anis-Lloyd corrected rescaled-range analysis.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 256
  min: 16
  description: Trailing-window length W in samples. Must be at least 4 * min_scale so the
    regression has at least three scales.
- name: min_scale
  type: int
  default: 4
  min: 4
  description: Smallest R/S block size; the scales are min_scale, 2*min_scale, ... up to window_size/2.
    Larger values reduce small-sample bias but leave fewer scales for the regression.
- name: method
  type: str
  default: rs
  enum:
  - rs
  description: Estimator family. Only "rs" (Anis-Lloyd corrected rescaled range) is available.
references:
- Hurst, H. E. (1951). Long-term storage capacity of reservoirs. *Transactions of the American
  Society of Civil Engineers*, 116, 770-799.
- Anis, A. A. and Lloyd, E. H. (1976). The expected value of the adjusted rescaled Hurst range
  of independent normal summands. *Biometrika*, 63(1), 111-116. doi:10.1093/biomet/63.1.111
- 'Peters, E. E. (1994). *Fractal Market Analysis: Applying Chaos Theory to Investment and
  Economics*. Wiley.'
- 'Weron, R. (2002). Estimating long-range dependence: finite sample properties and confidence
  intervals. *Physica A*, 312(1-2), 285-299. doi:10.1016/S0378-4371(02)00961-5'
- Estimating the Hurst exponent. arXiv:1805.08931.
see_also:
- RollingOU
- RollingADF
- FracDiff
- TrendMode
nan_policy: ignore
---

# `RollingHurst`

## Description

`RollingHurst` estimates the Hurst exponent $H$ over a trailing window. $H$ measures
long-range dependence in a time series:

- $H \approx 0.5$: uncorrelated (for example white noise, or the log-returns of a
  geometric Brownian motion).
- $H > 0.5$: persistent (moves tend to continue).
- $H < 0.5$: anti-persistent (moves tend to reverse).

The estimate uses rescaled-range (R/S) analysis with the Anis-Lloyd small-sample
correction. The trailing window of `window_size` samples is split into
non-overlapping blocks of length
$n_k = \texttt{min\_scale}, 2\cdot\texttt{min\_scale}, \dots, W/2$ (dyadic scales).
For each scale the average rescaled range across blocks is corrected for small-sample
bias, and $\hat H$ is the slope of its logarithm against $\log n$:

$$
\begin{aligned}
\text{R/S}(n) &= \frac{1}{\lfloor W/n \rfloor}
  \sum_{p=0}^{\lfloor W/n \rfloor-1}
  \frac{\max_j Y_{p,j} - \min_j Y_{p,j}}{\sigma_p},
  \quad Y_{p,j} = \sum_{i=1}^{j}(x_{p,i} - \bar x_p) \\[2pt]
\text{rsal}(n) &= \text{R/S}(n) - \text{ers}(n) + \sqrt{\tfrac{\pi n}{2}} \\[2pt]
\hat H &= \text{slope of } \log \text{rsal}(n_k) \text{ on } \log n_k
\end{aligned}
$$

where $\text{ers}(n)$ is the Anis-Lloyd expected R/S for $H = 1/2$:

$$
\text{ers}(n) = \frac{\Gamma\!\left(\tfrac{n-1}{2}\right)}{\sqrt{\pi}\,\Gamma\!\left(\tfrac{n}{2}\right)}
\sum_{i=1}^{n-1} \sqrt{\tfrac{n-i}{i}} \qquad (n \le 340; \text{asymptotic form otherwise}).
$$

The first valid output is at sample index `window_size - 1`. The output is also NaN
when any block in the window has zero variance.

## Notes on usage

- For asset prices, apply `RollingHurst` to returns (or log-returns), not to the price
  itself. Prices are close to a random walk, so the estimate on prices is near
  $H \approx 1$ whatever the memory in the returns.
- With `window_size=256` and `min_scale=4` the regression has 6 scales
  (4, 8, 16, 32, 64, 128). Smaller windows give noisier estimates; larger windows
  respond more slowly.

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
    from screamer import RollingHurst

    rng = np.random.default_rng(0)
    n = 1500

    # Stitch three regimes: white noise, trend (persistent), oscillating (anti-persistent).
    wn   = rng.normal(0, 1, n // 3)
    drift = 0.05 * np.arange(n // 3) + rng.normal(0, 1, n // 3)
    osc  = rng.normal(0, 1, n - 2 * (n // 3))
    osc[1::2] *= -1  # zig-zag → strong mean reversion
    x = np.concatenate([wn, drift, osc])

    H = RollingHurst(window_size=512)(x)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=x, mode='lines', name='signal',
                             line=dict(color='lightsteelblue'), yaxis='y1'))
    fig.add_trace(go.Scatter(y=H, mode='lines', name='RollingHurst(W=512)',
                             line=dict(color='crimson'), yaxis='y2'))
    fig.add_hline(y=0.5, line_dash="dot", line_color="gray",
                  annotation_text="H = 0.5 (uncorrelated)", yref='y2')
    fig.update_layout(
        title="Hurst exponent across regime changes",
        xaxis_title="Index",
        yaxis=dict(title="signal", side='left'),
        yaxis2=dict(title="Hurst H", side='right', overlaying='y', range=[0, 1]),
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.show()
```

<!-- HELP_END -->

## Implementation Details

* Circular buffer of size `W`.
* The dyadic scale list and the `ers(n_k)`, $\sqrt{\pi n_k / 2}$ and $\log n_k$
  tables are computed once in the constructor.
* Per step: the average R/S at each of $\sim \log_2 W$ scales, then a closed-form OLS
  slope of $\log \text{rsal}$ on $\log n$.
* Time complexity: `O(W log W)` per step.
* Space complexity: `O(W)`.

## Validation

Validated against a reference Python implementation of the algorithm in
`tests/test_rolling_hurst.py`.
