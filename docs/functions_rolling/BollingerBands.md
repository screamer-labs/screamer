---
name: BollingerBands
title: Bollinger Bands
implementation_family: rolling
topics:
- volatility
- bands
tags:
- bollinger
- bands
- envelope
- bollinger bands
- volatility bands
short: Mean +/- num_std rolling standard deviations.
inputs: 1
outputs: 3
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Number of most recent samples used for the mean and the standard deviation.
- name: num_std
  type: float
  default: 2.0
  min: 0.0
  description: 'Width of the bands: the distance from the middle band to each outer band,
    in rolling standard deviations.'
- name: start_policy
  type: str
  default: strict
  enum:
  - strict
  - expanding
  - zero
  description: 'Output before the window is full: "strict" gives NaN, "expanding" computes
    from the samples seen so far, and "zero" treats the window as pre-filled with zeros.'
output_names:
- lower
- middle
- upper
see_also:
- RollingMean
- RollingStd
- KeltnerChannels
- DonchianChannels
- RollingZscore
references:
- Bollinger, J. (2001). *Bollinger on Bollinger Bands*. McGraw-Hill.
nan_policy: ignore
---

# `BollingerBands`

## Description

`BollingerBands` computes the Bollinger bands of a single stream over a trailing
window: the middle band is the rolling mean, and the lower and upper bands sit
`num_std` standard deviations below and above it. All three are returned per step.

$$
\begin{aligned}
\mathrm{mid}_w[t]   &= \frac{1}{n} \sum_{i=t-w+1}^{t} x_i \\
\mathrm{std}_w[t]   &= \sqrt{\frac{1}{n - 1} \sum_{i=t-w+1}^{t} \bigl(x_i - \mathrm{mid}_w[t]\bigr)^2} \\
\mathrm{lower}_w[t] &= \mathrm{mid}_w[t] - k\,\mathrm{std}_w[t] \\
\mathrm{upper}_w[t] &= \mathrm{mid}_w[t] + k\,\mathrm{std}_w[t]
\end{aligned}
$$

where `n = window_size` and `k = num_std`. The standard deviation uses the unbiased
sample estimator (`ddof=1`), matching `pandas.Series.rolling(w).std()`.

A 1-D input of shape `(T,)` returns shape `(T, 3)` with columns lower, middle and
upper; a scalar call returns a tuple of three floats.

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
    from screamer import BollingerBands

    np.random.seed(0)
    N = 250
    prices = 100.0 + np.cumsum(np.random.normal(scale=0.6, size=N))

    bands = BollingerBands(window_size=20, num_std=2.0)(prices)
    lower, mid, upper = bands[:, 0], bands[:, 1], bands[:, 2]

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=upper, mode="lines", name="upper",
                             line=dict(color="green", dash="dot")))
    fig.add_trace(go.Scatter(y=mid, mode="lines", name="mid (mean)",
                             line=dict(color="black")))
    fig.add_trace(go.Scatter(y=lower, mode="lines", name="lower",
                             line=dict(color="red", dash="dot")))
    fig.add_trace(go.Scatter(y=prices, mode="lines", name="price",
                             line=dict(color="steelblue")))
    fig.update_layout(
        title="BollingerBands(window=20, num_std=2)",
        xaxis_title="time",
        yaxis_title="price",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.show()
```

<!-- HELP_END -->

## Implementation Details

Two rolling sums maintain `Σx` and `Σx²` over the window. Each step updates both sums
in `O(1)`, then computes the mean and the unbiased sample variance in closed form. The
standard deviation is `sqrt(max(var, 0))`, which guards against tiny negative values
from floating-point cancellation.

* **Time**: `O(1)` per new element.
* **Space**: `O(window_size)` (two circular buffers of length `window_size`).
* **Validation**: matches the pandas expression
    `mean = s.rolling(w).mean(); std = s.rolling(w).std(); (mean - k·std, mean, mean + k·std)`
  to floating-point tolerance, for windows `5, 10, 20, 30` and `num_std ∈ {0.5, 1.0, 2.0, 2.5}`
  in `tests/test_one_input_multi_output.py`.

### Edge cases

* Returns `(NaN, NaN, NaN)` until at least 2 samples have arrived (variance is undefined for a single sample under `ddof=1`).
* Under the `"strict"` policy, returns `(NaN, NaN, NaN)` until the window is full.
* `num_std = 0` is permitted; the upper, middle, and lower bands all equal the mean.
