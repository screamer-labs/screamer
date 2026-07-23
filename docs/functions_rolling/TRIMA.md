---
name: TRIMA
title: Triangular MA (TRIMA)
implementation_family: rolling
topics:
- smoothing
tags:
- trima
- triangular
- moving-average
short: 'Triangular MA: SMA of an SMA. Heavier center-weighting than WMA.'
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

# `TRIMA`

## Description

`TRIMA` (Triangular Moving Average) is a double-smoothed simple mean: an SMA of an SMA. The effective per-sample weights form a symmetric triangle (rising then falling), giving more weight to the centre of the window than the ends.

$$
\text{TRIMA}(x, n)[t] = \text{SMA}(\text{SMA}(x, n_\text{inner}), n_\text{outer})[t]
$$

with TA-Lib's window split:

| Total window `n` | `n_inner` | `n_outer` |
|---|---|---|
| odd | `(n + 1) / 2` | `(n + 1) / 2` |
| even | `n/2 + 1` | `n/2` |

In both cases `n_inner + n_outer - 1 == n`, so the effective triangular weighting spans `n` samples.

## Parameters

- `window_size` (int, positive). Total triangle width.

## Implementation Details

### Algorithm

Pure composition of two chained `detail::RollingMean` instances. Both run with `start_policy="expanding"` so that the inner doesn't emit `NaN` (which would poison the outer's running sum permanently). `TRIMA` itself enforces strict warmup by counting samples and emitting `NaN` until `n` samples have been processed.

### Complexity

* Time complexity: `O(1)` per step (two `RollingMean` updates).
* Space complexity: `O(window_size)`.


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
    from screamer import TRIMA

    rng = np.random.default_rng(0)
    N = 300
    data = np.cumsum(rng.standard_normal(N))

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data, mode='lines', name='Input', line=dict(color='steelblue', width=1)))
    fig.add_trace(go.Scatter(y=TRIMA(window_size=20)(data), mode='lines', name='TRIMA(window_size=20)', line=dict(color='crimson', width=2)))
    fig.update_layout(
        title="TRIMA smoother over a random walk",
        xaxis_title="Index", yaxis_title="Value",
        margin=dict(l=20, r=20, t=80, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.show()
```

<!-- HELP_END -->

## Reference

Equivalent to TA-Lib's `TRIMA`. Validated in `tests/test_moving_averages.py` against the explicit two-`RollingMean` composition for several window sizes (both even and odd).
