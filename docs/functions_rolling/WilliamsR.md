---
name: WilliamsR
title: Williams %R
implementation_family: rolling
topics:
- momentum
tags:
- williams-r
- oscillator
- talib
- hlc
- williams %r
- '%R'
- WILLR
short: Inverse stochastic oscillator in [-100, 0].
inputs: 3
outputs: 1
parameters:
- name: window_size
  type: int
  default: 14
  min: 1
  description: Number of most recent bars over which the highest high and lowest low are taken.
nan_policy: ignore
see_also:
- Stoch
- UltimateOscillator
- RollingRSI
- DonchianChannels
references:
- Williams, L. (1979). *How I Made One Million Dollars Last Year Trading Commodities*. Windsor
  Books.
---

# `WilliamsR`

## Description

`WilliamsR` (Williams %R, by Larry Williams) returns the normalized position of the close within the recent (high, low) range, scaled to `[-100, 0]`:

$$
\%R[t] = -100 \cdot \frac{H_n[t] - C[t]}{H_n[t] - L_n[t]}
$$

where `H_n` and `L_n` are the rolling max of `high` and rolling min of `low` over the window. `%R = 0` means the close is at the period high, and `%R = -100` means the close is at the period low.

The inputs are `(high, low, close)`, in the order of TA-Lib's `WILLR`.

## Warmup and edge cases

The first `window_size - 1` outputs are NaN, so the first valid output is at sample index `window_size - 1`, following TA-Lib.

When `H_n == L_n` over the period (a flat segment), the formula is undefined and `WilliamsR` returns 0, matching TA-Lib.

## Implementation Details

The rolling high and low are kept in two monotonic deques, as in `RollingMin` and `RollingMax`.

* Time complexity: `O(1)` amortized per step.
* Space complexity: `O(window_size)`.

## Output shape

| You pass... | You get back... |
|---|---|
| three scalars `high, low, close` | `float` |
| three 1D arrays of shape `(T,)` | array of shape `(T,)` |
| three 2D arrays of shape `(T, K)` | array of shape `(T, K)`, column-by-column |
| three parallel iterables | `list[float]` (eager) |


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

## Compatibility

Bit-exact match to TA-Lib's `WILLR(high, low, close, timeperiod)` post-warmup (verified in `tests/test_third_party_alignment.py` to ~1e-14).
