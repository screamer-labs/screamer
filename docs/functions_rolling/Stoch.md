---
name: Stoch
title: Stochastic Oscillator
implementation_family: rolling
topics:
- momentum
tags:
- stoch
- oscillator
- talib
- hlc
- stochastic oscillator
- stochastics
- '%K'
- '%D'
short: Stochastic oscillator %K and %D from high, low and close.
inputs: 3
outputs: 2
parameters:
- name: fastk_period
  type: int
  default: 14
  min: 1
  description: Number of most recent bars over which the highest high and lowest low are taken
    for raw %K.
- name: smooth_k
  type: int
  default: 3
  min: 1
  description: Period of the simple moving average applied to raw %K; 1 gives the fast stochastic,
    3 the slow stochastic of TA-Lib.
- name: d
  type: int
  default: 3
  min: 1
  description: Period of the simple moving average applied to %K to produce %D.
nan_policy: ignore
output_names:
- '%K'
- '%D'
see_also:
- StochRSI
- WilliamsR
- RollingRSI
- RollingMinMax
references:
- Lane, G. C. (1984). Lane's stochastics. *Technical Analysis of Stocks & Commodities*, 2(3),
  87-90.
---

# `Stoch`

## Description

`Stoch` (the stochastic oscillator of George Lane) returns the pair `(%K, %D)` per step. The raw stochastic measures where the close sits within the recent high-low range, scaled to `[0, 100]`. Two layers of simple moving average smoothing then give the "fast" and "slow" variants.

$$
\begin{aligned}
\text{raw\_K}[t] &= 100 \cdot \frac{C - L_n}{H_n - L_n} \\
\%K[t]           &= \text{SMA}(\text{raw\_K},\ \text{smooth\_k}) \\
\%D[t]           &= \text{SMA}(\%K,\ d)
\end{aligned}
$$

The inputs are `(high, low, close)`, and the outputs are `%K` and `%D`.

## Common configurations

The common variants differ only in `smooth_k`:

| Configuration | Constructor | TA-Lib equivalent |
|---|---|---|
| **Slow Stochastic** (charting default; what `talib.STOCH` returns) | `Stoch(14, 3, 3)` | `STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)` |
| **Fast Stochastic** (Lane's original, no smoothing of %K) | `Stoch(14, 1, 3)` | `STOCHF(high, low, close, fastk_period=14, fastd_period=3)` |
| TA-Lib's *function* defaults (rarely used by traders) | `Stoch(5, 3, 3)` | `STOCH(..., fastk_period=5, slowk_period=3, slowd_period=3)` |

With `smooth_k=1` the moving average of period 1 is the identity, so `%K = raw_K` and the class returns the fast variant.

## Warmup and edge cases

Both outputs are NaN until the `%D` line is valid, at sample index `fastk_period + smooth_k + d - 3`. This follows TA-Lib, which starts both lines together. For the default `(14, 3, 3)` that is index 17.

When `H_n == L_n` over the period (a flat segment) the raw stochastic is undefined, and `Stoch` returns 0, matching TA-Lib.

## Implementation Details

The rolling high and low are kept in two monotonic deques, and the two smoothing stages are rolling means.

* Time complexity: `O(1)` amortized per step.
* Space complexity: `O(fastk_period + smooth_k + d)`.

## Output shape

`%K` is `out[..., 0]` and `%D` is `out[..., 1]`:

| You pass... | You get back... |
|---|---|
| three scalars | tuple `(%K, %D)` |
| three 1D arrays of shape `(T,)` | array of shape `(T, 2)` |
| three 2D arrays of shape `(T, K)` | array of shape `(T, K, 2)` |


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
    from screamer import Stoch

    rng = np.random.default_rng(0)
    n = 300
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + np.abs(rng.normal(0, 0.5, n))
    low = close - np.abs(rng.normal(0, 0.5, n))

    out = Stoch(14, 3, 3)(high, low, close)   # slow stochastic
    pct_k = out[:, 0]
    pct_d = out[:, 1]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=close, mode='lines', name='Close',
                             line=dict(color='steelblue')), row=1, col=1)
    fig.add_trace(go.Scatter(y=pct_k, mode='lines', name='%K (slow)',
                             line=dict(color='steelblue')), row=2, col=1)
    fig.add_trace(go.Scatter(y=pct_d, mode='lines', name='%D',
                             line=dict(color='red')), row=2, col=1)
    fig.add_hline(y=20, line=dict(color='gray', dash='dot'), row=2, col=1)
    fig.add_hline(y=80, line=dict(color='gray', dash='dot'), row=2, col=1)
    fig.update_layout(
        title="Stoch(14, 3, 3): slow Stochastic with oversold/overbought lines",
        xaxis_title="Index",
        yaxis_title="Price",
        yaxis2_title="%K / %D",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    fig.show()
```

<!-- HELP_END -->

## Compatibility

`Stoch(14, 3, 3)` matches `talib.STOCH(high, low, close, 14, 3, 0, 3, 0)` bit-exactly post-warmup. `Stoch(14, 1, 3)` matches `talib.STOCHF(high, low, close, 14, 3, 0)`. Both verified to ~1e-13 in `tests/test_third_party_alignment.py`.
