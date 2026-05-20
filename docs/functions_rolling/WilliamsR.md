---
name: WilliamsR
title: Williams %R
implementation_family: rolling
topics:
- oscillator
tags:
- williams-r
- oscillator
- talib
- hlc
short: Inverse stochastic oscillator in [-100, 0].
inputs: 3
outputs: 1
parameters:
- name: window_size
  type: int
  default: 14
  min: 2
  description: Period (Wilder's default).
---

# `WilliamsR`

## Description

`WilliamsR` (Williams %R, Larry Williams 1973) returns the normalised position of the close within the recent (high, low) range, scaled to `[-100, 0]`:

$$
\%R[t] = -100 \cdot \frac{H_n[t] - C[t]}{H_n[t] - L_n[t]}
$$

where `H_n` and `L_n` are the rolling max of `high` and rolling min of `low` over the window. `%R = 0` means the close is at the period high (typically a strong reading); `%R = -100` means the close is at the period low.

This is a **3-input, 1-output** function (`FunctorBase<_, 3, 1>`). Input order is `(high, low, close)`, matching TA-Lib's `WILLR`.

## Parameters

- `window_size` (int, default `14`): the rolling-window length.

*Warmup*: NaN for the first `window_size − 1` samples; first valid output at sample index `window_size − 1` (TA-Lib's convention).

*Range-zero handling*: when `H_n == L_n` over the period (a perfectly flat segment), the formula is mathematically undefined. We return `0` in that case, matching TA-Lib.

*NaN handling*: NaN inputs should be preprocessed (the deque comparisons treat NaN as never beating an existing element).

## Implementation Details

Pure composition of two `detail::MonotonicDeque` instances -- the same primitive used by `RollingMin`/`RollingMax`/`RollingMinMax`/`RollingArgmin`/`RollingArgmax`/`RollingRange`. Amortised O(1) per step.


* Time complexity: `O(1)` amortised per step.
* Space complexity: `O(window_size)`.

## Output shape

| You pass... | You get back... |
|---|---|
| three scalars `high, low, close` | `float` |
| three 1D arrays of shape `(T,)` | array of shape `(T,)` |
| three 2D arrays of shape `(T, K)` | array of shape `(T, K)`, column-by-column |
| three parallel iterables | `list[float]` (eager) |

## Examples

### Implementation Details

```
high_n = max_deque.append(high)
low_n  = min_deque.append(low)
range  = high_n - low_n
return -100 * (high_n - close) / range   (or 0 if range == 0)
```

### Usage example

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import WilliamsR

    rng = np.random.default_rng(0)
    n = 300
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + np.abs(rng.normal(0, 0.5, n))
    low = close - np.abs(rng.normal(0, 0.5, n))

    wr = WilliamsR(14)(high, low, close)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=high, mode='lines', name='High',
                             line=dict(color='lightgray')), row=1, col=1)
    fig.add_trace(go.Scatter(y=low, mode='lines', name='Low',
                             line=dict(color='lightgray')), row=1, col=1)
    fig.add_trace(go.Scatter(y=close, mode='lines', name='Close',
                             line=dict(color='steelblue')), row=1, col=1)
    fig.add_trace(go.Scatter(y=wr, mode='lines', name='Williams %R(14)',
                             line=dict(color='red')), row=2, col=1)
    fig.add_hline(y=-20, line=dict(color='gray', dash='dot'), row=2, col=1)
    fig.add_hline(y=-80, line=dict(color='gray', dash='dot'), row=2, col=1)
    fig.update_layout(
        title="Williams %R(14): close position within the rolling H/L range",
        xaxis_title="Index",
        yaxis_title="Price",
        yaxis2_title="%R",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(range=[-100, 0], row=2, col=1)
    fig.show()
```

<!-- HELP_END -->

## Reference

Bit-exact match to TA-Lib's `WILLR(high, low, close, timeperiod)` post-warmup (verified in `tests/test_third_party_alignment.py` to ~1e-14).
