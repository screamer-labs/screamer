---
name: BollingerBands
title: Bollinger Bands
implementation_family: rolling
topics:
- channels
- volatility
tags:
- bollinger
- bands
- envelope
short: Mean +/- num_std rolling standard deviations.
inputs: 1
outputs: 3
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Trailing-window length.
- name: num_std
  type: float
  default: 2.0
  min: 0.0
  description: Number of rolling-std offsets for the upper/lower bands.
- name: start_policy
  type: str
  default: strict
  enum:
  - strict
  - expanding
  - zero
  description: Warmup behaviour.
nan_policy: ignore
---

# `BollingerBands`

## Description

`BollingerBands` computes the classic Bollinger bands of a single stream over a sliding window: the middle band is the rolling mean, and the lower and upper bands sit `num_std` standard deviations away on either side. All three are returned per step.

*Equation*:

$$
\begin{aligned}
\mathrm{mid}_w[t]   &= \frac{1}{n} \sum_{i=t-w+1}^{t} x_i \\
\mathrm{std}_w[t]   &= \sqrt{\frac{1}{n - 1} \sum_{i=t-w+1}^{t} \bigl(x_i - \mathrm{mid}_w[t]\bigr)^2} \\
\mathrm{lower}_w[t] &= \mathrm{mid}_w[t] - k\,\mathrm{std}_w[t] \\
\mathrm{upper}_w[t] &= \mathrm{mid}_w[t] + k\,\mathrm{std}_w[t]
\end{aligned}
$$

where `n = window_size` and `k = num_std`. The standard deviation uses the unbiased sample estimator (`ddof=1`), matching `pandas.Series.rolling(w).std()`.

*Parameters*:

- **`window_size`** (`int`, ≥ 2): size of the rolling window.
- **`num_std`** (`float`, ≥ 0, default `2.0`): width of the bands in standard deviations.
- **`start_policy`** (`str`, default `"strict"`): controls warmup behavior. See `RollingMean` for the full definition.

*Input shape*: a single stream (scalar, 1-D array, 2-D array, iterator, …). Same matrix as `RollingMean`.

*Output shape*: an extra trailing axis of size **3** is appended to the input shape. A 1-D input of shape `(T,)` returns shape `(T, 3)`; index `0` is the lower band, index `1` is the middle band, index `2` is the upper band. A scalar call returns a Python `tuple` of three floats.


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

Two `detail::RollingSum` buffers maintain `Σx` and `Σx²` over the window. Each step updates both sums in `O(1)`, then the mean and (unbiased sample) variance are computed in closed form, the standard deviation is `sqrt(max(var, 0))` to guard against tiny negative values from floating-point cancellation, and the three band values are returned as a tuple.

* **Time**: `O(1)` per new element.
* **Space**: `O(window_size)` (two circular buffers of length `window_size`).
* **Reference**: parity with the pandas expression
    `mean = s.rolling(w).mean(); std = s.rolling(w).std(); (mean - k·std, mean, mean + k·std)`
  to floating-point tolerance, parametrised over windows `5, 10, 20, 30` and `num_std ∈ {0.5, 1.0, 2.0, 2.5}` in `tests/test_one_input_multi_output.py`.

### Edge cases

* Returns `(NaN, NaN, NaN)` until at least 2 samples have arrived (variance is undefined for a single sample under `ddof=1`).
* Under `"strict"` policy, returns `(NaN, NaN, NaN)` until the window is full.
* `num_std = 0` is permitted; the upper, middle, and lower bands all equal the mean.

### Related

- [`RollingMean`](RollingMean.md), [`RollingStd`](RollingStd.md): the building blocks. Use them directly if you only need one of the two.
