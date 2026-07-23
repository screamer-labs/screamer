---
name: HullMA
title: Hull MA
implementation_family: rolling
topics:
- smoothing
- trend
tags:
- hull
- hullma
- moving-average
short: 'Hull''s responsive MA: WMA(2*WMA(n/2) - WMA(n), sqrt(n)).'
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

# `HullMA`

## Description

`HullMA` (Alan Hull, 2005) is a low-lag responsive moving average defined entirely in terms of `WMA`:

$$
\text{HullMA}(x, n)[t] = \text{WMA}\Big( 2 \cdot \text{WMA}(x, n/2) - \text{WMA}(x, n),\ \sqrt{n} \Big)[t]
$$

with integer floor on the inner window arguments: `n_half = n // 2` and `n_sqrt = floor(sqrt(n))`.

The construction subtracts a slow `WMA` from twice a fast `WMA` (anticipating the trend, similar to `DEMA`'s linear extrapolation), then smooths the result with a much shorter `WMA`. The output tracks the price closely with markedly less lag than a plain SMA / EMA / WMA of comparable window.

## Parameters

- `window_size` (int, **at least 4**). The construction degenerates below `n=4` because `n_half` must be `>= 2` and `floor(sqrt(n))` must be `>= 2`.

## Implementation Details

### Algorithm

Pure composition of three `WMA` instances:

1. `wma_half_` -- `WMA(n // 2, "expanding")` on the input `x`.
2. `wma_full_` -- `WMA(n, "expanding")` on the input `x`.
3. `wma_outer_` -- `WMA(floor(sqrt(n)), "expanding")` on `2*wma_half - wma_full`.

The inner WMAs run with `start_policy="expanding"` so they never emit `NaN` (which would poison the outer's state). `HullMA` enforces strict warmup itself by counting samples and emitting `NaN` until `n + floor(sqrt(n)) - 1` samples have been processed.

### Complexity

* Time complexity: `O(1)` per step (three `WMA` updates).
* Space complexity: `O(window_size)` (dominated by the longest internal `WMA`).


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
    from screamer import HullMA

    rng = np.random.default_rng(0)
    N = 300
    data = np.cumsum(rng.standard_normal(N))

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data, mode='lines', name='Input', line=dict(color='steelblue', width=1)))
    fig.add_trace(go.Scatter(y=HullMA(window_size=20)(data), mode='lines', name='HullMA(window_size=20)', line=dict(color='crimson', width=2)))
    fig.update_layout(
        title="HullMA smoother over a random walk",
        xaxis_title="Index", yaxis_title="Value",
        margin=dict(l=20, r=20, t=80, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.show()
```

<!-- HELP_END -->

## Reference

Standard Hull MA definition. Validated in `tests/test_moving_averages.py` against the explicit three-`WMA` composition for several window sizes.
