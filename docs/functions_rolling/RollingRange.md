---
name: RollingRange
title: Rolling range
implementation_family: rolling
topics:
- statistics
- volatility
tags:
- range
- min
- max
- rolling
- high-low range
- rolling range
short: Trailing-window max minus min.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 1
  description: Number of most recent samples in the window.
nan_policy: ignore
see_also:
- RollingMinMax
- DonchianChannels
- TrueRange
- RollingStd
---

# `RollingRange`

## Description

`RollingRange` returns the rolling range, `max - min` over the trailing window. It is used in volatility heuristics, breakout detection, and as the width of channel indicators such as Donchian channels.

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
    from screamer import RollingRange

    rng = np.random.default_rng(0)
    N = 300
    data = np.cumsum(rng.standard_normal(N))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5],
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=data, mode='lines', name='Input', line=dict(color='steelblue', width=1)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(y=RollingRange(window_size=30)(data), mode='lines', name='RollingRange(window_size=30)',
                             line=dict(color='crimson', width=2)), row=2, col=1)
    fig.update_layout(
        title="Rolling range (max - min) over a random walk",
        yaxis=dict(title='Input'), yaxis2=dict(title='Range'),
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    fig.show()
```

<!-- HELP_END -->

## Implementation Details

### Algorithm

`RollingRange` keeps one monotonic deque for the minimum and one for the maximum, as `RollingMin`, `RollingMax` and `RollingMinMax` do. On each step both deques are updated and the difference between their fronts is returned. This is the same algorithm as `RollingMinMax` followed by a subtraction, with one output column instead of two.

### Complexity

* Time complexity: `O(1)` amortized per step.
* Space complexity: `O(window_size)` (two deques, each bounded by the window).

## Validation

`RollingMax(w)(x) - RollingMin(w)(x)` and `pandas.Series.rolling(w).max() - pandas.Series.rolling(w).min()` are both algorithmically equivalent (post-warmup, to floating-point precision).
