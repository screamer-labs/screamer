---
name: Stoch
title: Stochastic Oscillator
implementation_family: rolling
topics:
- oscillator
tags:
- stoch
- oscillator
- talib
- hlc
short: Stochastic oscillator %K and %D (3 inputs -> 2 outputs).
inputs: 3
outputs: 2
parameters:
- name: fastk_period
  type: int
  default: 14
  min: 2
  description: Lookback for the %K stochastic.
- name: smooth_k
  type: int
  default: 3
  min: 1
  description: Smoothing period for %K (1 = fast stoch, 3 = TA-Lib slow stoch).
- name: d
  type: int
  default: 3
  min: 1
  description: SMA period for %D.
nan_policy: ignore
---

# `Stoch`

## Description

`Stoch` (Stochastic oscillator, George Lane, 1950s) returns the pair `(%K, %D)` per step. The raw stochastic measures where the close sits within the recent (high, low) range, scaled to `[0, 100]`. Two layers of optional SMA smoothing then give the canonical "fast" and "slow" variants.

$$
\begin{aligned}
\text{raw\_K}[t] &= 100 \cdot \frac{C - L_n}{H_n - L_n} \\
\%K[t]           &= \text{SMA}(\text{raw\_K},\ \text{smooth\_k}) \\
\%D[t]           &= \text{SMA}(\%K,\ d)
\end{aligned}
$$

This is a **3-input, 2-output** function (`FunctorBase<_, 3, 2>`). Input order is `(high, low, close)`.

## Setting it up for the popular cases

The one class covers every common parameterisation by choosing `smooth_k`:

| Configuration | Constructor | TA-Lib equivalent |
|---|---|---|
| **Slow Stochastic** (charting default; what `talib.STOCH` returns) | `Stoch(14, 3, 3)` | `STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)` |
| **Fast Stochastic** (Lane's original) -- skip the smooth-K SMA | `Stoch(14, 1, 3)` | `STOCHF(high, low, close, fastk_period=14, fastd_period=3)` |
| TA-Lib's *function* defaults (rarely used by traders) | `Stoch(5, 3, 3)` | `STOCH(..., fastk_period=5, slowk_period=3, slowd_period=3)` |

`smooth_k=1` is the trick: SMA of period 1 is the identity, so `%K = raw_K` and you get the fast variant out of the same class.

## Parameters

- `fastk_period` (int, default `14`): rolling-window length for the H / L deques.
- `smooth_k` (int, default `3`): SMA period applied to `raw_K`. Set to `1` for the fast Stochastic.
- `d` (int, default `3`): SMA period applied to `%K` to produce `%D`.

*Warmup*: both outputs are NaN until the `%D` line is valid, at sample index `fastk_period + smooth_k + d - 3` (TA-Lib's convention -- gate both K and D together). For the default `(14, 3, 3)` that is index 17.

*Range-zero handling*: when `H_n == L_n` over the period (a perfectly flat segment) the raw stochastic is undefined; we return 0, matching TA-Lib.

*NaN handling*: NaN inputs should be preprocessed.

## Implementation Details

Pure composition of two `detail::MonotonicDeque` (one each for high / low) plus two `detail::RollingMean` instances (`smooth_k` and `d`). Amortised O(1) per step.

* Time complexity: `O(1)` amortised per step.
* Space complexity: `O(fastk_period + smooth_k + d)`.

## Output shape

`%K` is `out[..., 0]`, `%D` is `out[..., 1]`. Otherwise standard `1 → 2` shape rules (after the 3-input pairing):

| You pass... | You get back... |
|---|---|
| three scalars | tuple `(%K, %D)` |
| three 1D arrays of shape `(T,)` | array of shape `(T, 2)` |
| three 2D arrays of shape `(T, K)` | array of shape `(T, K, 2)` |


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

## Reference

`Stoch(14, 3, 3)` matches `talib.STOCH(high, low, close, 14, 3, 0, 3, 0)` bit-exactly post-warmup. `Stoch(14, 1, 3)` matches `talib.STOCHF(high, low, close, 14, 3, 0)`. Both verified to ~1e-13 in `tests/test_third_party_alignment.py`.
