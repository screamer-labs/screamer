---
name: RollingArgmax
title: Rolling argmax
implementation_family: rolling
topics:
- statistics
tags:
- argmax
- rolling
short: Window-offset of the trailing-window maximum (TA-Lib MAXINDEX).
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Trailing-window length.
nan_policy: ignore
---

# `RollingArgmax`

## Description

`RollingArgmax` returns the *position* (within the current window) of the rolling maximum value, rather than the maximum itself. Convention: **0 = oldest sample in the window**, **window_size−1 = newest**. Matches `numpy.argmax` applied to the trailing window slice and `pandas.Series.rolling(w).apply(np.argmax)`.

*Parameters*: `window_size` (int, positive).

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
    from plotly.subplots import make_subplots
    from screamer import RollingArgmax

    rng = np.random.default_rng(0)
    N = 300
    data = np.cumsum(rng.standard_normal(N))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5],
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=data, mode='lines', name='Input', line=dict(color='steelblue', width=1)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(y=RollingArgmax(window_size=30)(data), mode='lines',
                             name='RollingArgmax(window_size=30)',
                             line=dict(color='crimson', width=2)), row=2, col=1)
    fig.update_layout(
        title="Rolling argmax (window offset of maximum) over a random walk",
        yaxis=dict(title='Input'), yaxis2=dict(title='Argmax offset'),
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    fig.show()
```

<!-- HELP_END -->

## Implementation Details

### Algorithm

Same monotonic-deque primitive (`detail::MaxDeque`) used by `RollingMax`, `RollingMinMax`, and `RollingRange`. Each deque entry stores `(value, absolute_sample_index)`; the front entry is always the current rolling maximum, and we expose its window offset.

### Complexity

* Time complexity: `O(1)` amortised per step.
* Space complexity: `O(window_size)`.

## Reference

Equivalent to `pandas.Series.rolling(w).apply(np.argmax, raw=True)` for samples after warmup. Also equivalent to TA-Lib's `MAXINDEX`.
