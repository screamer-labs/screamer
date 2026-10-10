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
- hull moving average
- hma
short: 'Hull''s responsive MA: WMA(2*WMA(n/2) - WMA(n), sqrt(n)).'
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 4
  description: Main window n in samples; the inner windows are n // 2 and floor(sqrt(n)).
    Must be at least 4 so that both inner windows are at least 2.
see_also:
- WMA
- DEMA
- TEMA
- KAMA
nan_policy: ignore
---

# `HullMA`

## Description

`HullMA` (Alan Hull, 2005) is a low-lag moving average built from three
[`WMA`](WMA.md)s:

$$
\text{HullMA}(x, n)[t] = \text{WMA}\Big( 2 \cdot \text{WMA}(x, n/2) - \text{WMA}(x, n),\ \sqrt{n} \Big)[t]
$$

with the inner window lengths rounded down: `n_half = n // 2` and
`n_sqrt = floor(sqrt(n))`. Subtracting a slow `WMA` from twice a fast `WMA`
extrapolates the trend, much like `DEMA`, and the short outer `WMA` smooths the
result. The output follows the input with less lag than an SMA, EMA or WMA of the same
window.

The first valid output is at sample index `n + floor(sqrt(n)) - 2`; earlier outputs
are NaN. Each step costs `O(1)` time and `O(window_size)` memory.

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

## Validation

Matches the explicit three-`WMA` composition for several window sizes in
`tests/test_moving_averages.py`.
