---
name: RollingArgmax
title: Rolling argmax
implementation_family: rolling
topics:
- statistics
tags:
- argmax
- rolling
- maxindex
- position of maximum
- index of maximum
short: Position of the maximum within the trailing window (0 is the oldest sample).
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Number of most recent samples searched for the maximum.
see_also:
- RollingMax
- RollingArgmin
- RollingMinMax
- RollingRange
nan_policy: ignore
---

# `RollingArgmax`

## Description

`RollingArgmax` returns the position of the rolling maximum within the current window,
rather than the maximum itself. Position 0 is the oldest sample in the window and
`window_size - 1` the newest. When the maximum occurs more than once in the window,
the newest position is returned. Otherwise the result matches `numpy.argmax` applied to
the trailing window slice; `numpy.argmax` returns the oldest position on ties.

Before the window is full, the output is the position within the samples seen so far,
so there is no NaN warmup.

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

A monotonic deque, the same structure that [`RollingMax`](RollingMax.md) uses. Each deque
entry stores the value and its absolute sample index; the front entry is the current
rolling maximum, and its index is converted to a window offset.

### Complexity

* Time complexity: `O(1)` amortized per step.
* Space complexity: `O(window_size)`.

## Validation

Equivalent to `pandas.Series.rolling(w).apply(np.argmax, raw=True)` after warmup when the
maximum is unique in the window. Pandas returns NaN during warmup, where `RollingArgmax` returns
the position within the partial window. TA-Lib's `MAXINDEX` locates the same maximum but
returns its absolute index in the series rather than its offset in the window.
