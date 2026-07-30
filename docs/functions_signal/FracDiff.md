---
name: FracDiff
title: Fractional differentiation (FracDiff)
implementation_family: signal
topics:
- filtering
- standardization
tags:
- fracdiff
- fractional-differentiation
- stationarity
- fir
short: Fractional differencing filter. Removes the trend while keeping more of the series' memory than an integer difference.
inputs: 1
outputs: 1
parameters:
- name: d
  type: float
  default: 0.4
  description: Differencing order. 0 is the identity, 1 the first difference, 2 the second.
- name: window_size
  type: int
  default: 100
  min: 1
  description: Maximum number of taps, which bounds memory and per-step cost.
- name: threshold
  type: float
  default: 1.0e-5
  min: 0
  description: Truncate the weight series at the first weight smaller than this in absolute value.
- name: start_policy
  type: str
  default: strict
  enum:
  - strict
  - expanding
  - zero
  description: Warmup behaviour.
nan_policy: ignore
---

# `FracDiff`

## Description

`FracDiff` applies fractional differencing of order `d`: the filter whose taps are the binomial expansion of $(1-B)^d$, where $B$ is the backshift operator $Bx[t] = x[t-1]$.

$$
w_0 = 1, \qquad w_k = -w_{k-1}\,\frac{d - k + 1}{k}, \qquad y[t] = \sum_{k=0}^{L-1} w_k \, x[t-k]
$$

which is $w_k = (-1)^k \binom{d}{k}$.

Integer orders give the ordinary differences: `d=0` produces taps `[1]` and returns the input, `d=1` produces `[1, -1]` and equals [`Diff(1)`](../functions_misc/Diff.md), `d=2` produces `[1, -2, 1]` and equals [`Diff2`](../functions_misc/Diff2.md). At a fractional order the series does not terminate. Its weights decay as a power law, so `y[t]` keeps a weighted trace of the whole lookback rather than only the last one or two samples. That trace is the memory a first difference discards, and the reason to prefer `d = 0.4` over `d = 1` when the differenced series feeds a model that has something to learn from the level.

The order controls the trade: raise `d` toward 1 to remove more of the trend, lower it toward 0 to keep more memory.

*Parameters*:

- `d` (float): the differencing order. Negative values are fractional integration, where the weights share one sign and decay more slowly.
- `window_size` (int, positive): the largest number of taps the filter will hold.
- `threshold` (float, non-negative): drop the tail of the weight series at the first $|w_k|$ below this. Pass `0.0` to keep all `window_size` taps.
- `start_policy` (str, optional): `"strict"` (default), `"expanding"`, or `"zero"`. See **Warmup behaviour** below.

## Implementation Details

### Truncation

The weight series is infinite for a fractional `d`, so the filter is truncated at

$$
L = \min\big(\text{window\_size},\ \text{first } k \text{ with } |w_k| < \text{threshold}\big)
$$

`window_size` bounds the cost, `threshold` removes taps too small to change the output. Both matter: at `d = 0.4` the weights fall below `1e-5` after a few hundred terms, while at `d = 0.05` they fall off far more slowly and `window_size` is what stops the filter.

`L` grows as `d` approaches 0 and shrinks as `d` approaches an integer, so the same `threshold` gives different filter lengths at different orders. Count the leading `NaN` values under `start_policy="strict"` to see the `L` a given configuration resolved to.

### Complexity

A circular buffer of the last `L` samples and one convolution sweep. **O(L) per step**, `O(L)` memory. The weights are computed once when the operator is constructed.

### Warmup behaviour

`"strict"` (the default) returns `NaN` until `L` finite samples have arrived, which is the first index at which every tap has a sample to multiply.

`"expanding"` and `"zero"` both return the convolution over the samples seen so far. For a linear filter these two are the same arithmetic: using only the available samples is what padding the missing past with zeros produces, since the missing terms contribute zero either way. The early values are biased toward zero, and they are the values Lopez de Prado's fixed-width method discards.

See [NaN and warmup](../nan_and_warmup.md) for the full definition.

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
    from screamer import FracDiff

    rng = np.random.default_rng(0)
    data = np.cumsum(rng.standard_normal(600))

    partial = FracDiff(d=0.4, window_size=100)(data)
    whole = FracDiff(d=1.0, window_size=100)(data)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("Random walk",
                                        "d = 0.4",
                                        "d = 1.0 (first difference)"))
    fig.add_trace(go.Scatter(y=data, mode='lines', name='Input',
                             line=dict(color='lightblue')), row=1, col=1)
    fig.add_trace(go.Scatter(y=partial, mode='lines', name='d = 0.4',
                             line=dict(color='red')), row=2, col=1)
    fig.add_trace(go.Scatter(y=whole, mode='lines', name='d = 1.0',
                             line=dict(color='gray')), row=3, col=1)
    fig.update_layout(
        height=700, showlegend=False,
        title="Fractional differencing at two orders",
        margin=dict(l=20, r=20, t=80, b=20),
    )
    fig.show()
```

At `d = 0.4` the output still tracks the slow swings of the walk. At `d = 1.0` it is white noise around zero: the level is gone, and so is everything a model could have learned from it.

<!-- HELP_END -->

## Reference

Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*, chapter 5. The fixed-width variant is the one implemented here: the weight series is truncated by `threshold` and `window_size`, and the filter is applied at full width from then on.

The integer orders reproduce the corresponding operators on finite input. `FracDiff(d=0)` returns its input unchanged and `FracDiff(d=1, window_size>=2)` is bit-exact to `Diff(1)`, both being a single subtraction. `FracDiff(d=2, window_size>=3)` matches `Diff2()` to about 5e-15 relative rather than bit-for-bit: this filter sums `1*x[t] - 2*x[t-1] + 1*x[t-2]` in tap order while `Diff2` chains two subtractions, so the same terms are associated differently.
