---
name: MovingAverage
title: FIR moving average (arbitrary taps)
implementation_family: signal
topics:
- smoothing
- filtering
tags:
- fir
- convolution
- moving-average
- fir filter
- weighted moving average
- kernel
short: Finite-impulse-response filter with user-supplied taps.
inputs: 1
outputs: 1
parameters:
- name: taps
  type: float[]
  default:
  - 0.25
  - 0.5
  - 0.25
  description: FIR coefficients; taps[0] multiplies the current sample and taps[-1] the oldest.
    Must be non-empty. The default is a 3-tap triangular kernel.
nan_policy: ignore
see_also:
- WMA
- TRIMA
- RollingMean
- Butter
---

# `MovingAverage`

## Description

Finite-impulse-response (FIR) filter with arbitrary user-supplied tap coefficients:

$$
y[t] = \sum_{k=0}^{L-1} \text{taps}[k] \cdot x[t - k]
$$

`taps[0]` is the coefficient on the *current* sample; `taps[L-1]` is on the oldest. Pre-compute the coefficient vector with `numpy` (`np.hamming(n)`, `np.bartlett(n)`, `np.blackman(n)`, `np.kaiser(n, beta)`) or `scipy.signal.firwin` and pass it in. The taps are used as given, so normalize them yourself if you need unity gain (for example, divide by `taps.sum()` for a low-pass smoother).

The first `len(taps) - 1` outputs are `NaN`.

## Implementation Details

Circular buffer of the last `L = len(taps)` samples plus an in-order convolution sweep. **O(L) per step**.


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
    from screamer import MovingAverage

    rng = np.random.default_rng(0)
    N = 300
    data = np.cumsum(rng.standard_normal(N))

    # Hamming-windowed low-pass smoother (unity gain)
    taps = np.hamming(21)
    taps /= taps.sum()

    smoothed = MovingAverage(list(taps))(data)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data, mode='lines', name='Input',
                             line=dict(color='lightblue'), opacity=0.8))
    fig.add_trace(go.Scatter(y=smoothed, mode='lines',
                             name='MovingAverage (21-tap Hamming)',
                             line=dict(color='red')))
    fig.update_layout(
        title="FIR moving average with 21-tap Hamming window",
        xaxis_title="Index", yaxis_title="Value",
        margin=dict(l=20, r=20, t=80, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.show()
```

<!-- HELP_END -->

## Accuracy

Matches `numpy.convolve(x, taps, mode='full')[:len(x)]` post-warmup to floating-point precision. With uniform `1/n` taps it agrees with `RollingMean(n)` to rounding error (about 1e-15).
