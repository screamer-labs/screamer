---
name: MACD
title: MACD
implementation_family: rolling
topics:
- trend
tags:
- macd
- ema
- appel
- talib
short: MACD line, signal line, and histogram (3 outputs).
inputs: 1
outputs: 3
parameters:
- name: fast
  type: int
  default: 12
  min: 2
  description: Fast EMA span.
- name: slow
  type: int
  default: 26
  min: 2
  description: Slow EMA span.
- name: signal
  type: int
  default: 9
  min: 2
  description: Signal-line EMA span.
nan_policy: ignore
---

# `MACD`

## Description

`MACD` (Moving Average Convergence Divergence, Gerald Appel) is a momentum indicator that subtracts a slow EMA from a fast EMA and then smooths the result. It returns three quantities at every step:

$$
\begin{aligned}
\text{macd}[t]      &= \text{EMA}_\text{fast}(x)[t] - \text{EMA}_\text{slow}(x)[t] \\
\text{signal}[t]    &= \text{EMA}_\text{signal}(\text{macd})[t] \\
\text{histogram}[t] &= \text{macd}[t] - \text{signal}[t]
\end{aligned}
$$

`MACD` is a **1 → 3** functor: pass one input stream, get back the triple `(macd, signal, histogram)` per step.

## Parameters

- `fast` (int, default `12`): span of the fast EMA.
- `slow` (int, default `26`): span of the slow EMA. Must be strictly greater than `fast`.
- `signal` (int, default `9`): span of the signal-line EMA.

Defaults match Appel's original choice and every charting platform's default.

*Warmup*: none. Each underlying EMA is `screamer.EwMean`, which is well-defined from sample t=0 (`EMA[0] = x[0]`). At t=0 both EMAs equal `x[0]`, so `macd[0] = 0`, `signal[0] = 0`, `histogram[0] = 0`. From t=1 onward all three outputs are non-trivial.

*NaN handling*: NaN inputs propagate through the EMA arithmetic and poison subsequent outputs.

## Output shape

| You pass... | You get back... |
|---|---|
| scalar `x` | `tuple(macd, signal, histogram)` |
| 1D array shape `(T,)` | array shape `(T, 3)` |
| 2D array shape `(T, K)` | array shape `(T, K, 3)`, column-by-column |

`out[..., 0]` is `macd`, `out[..., 1]` is `signal`, `out[..., 2]` is `histogram`.

## Implementation Details

Pure composition of three `EwMean` instances. Per step:


- Time complexity: `O(1)` per step.
- Space complexity: `O(1)` (each `EwMean` holds two scalars).


<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
<!-- NAN_FOOTNOTE_END -->

## Examples

### Implementation Details

```
f = ema_fast.process_scalar(x)        // span = fast
s = ema_slow.process_scalar(x)        // span = slow
macd = f - s
signal = ema_signal.process_scalar(macd)   // span = signal
histogram = macd - signal
```

### Usage example

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import MACD

    rng = np.random.default_rng(0)
    n = 300
    price = 100 + np.cumsum(rng.normal(0.0, 1.0, n))

    out = MACD()(price)
    macd, signal, hist = out[:, 0], out[:, 1], out[:, 2]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=price, mode='lines', name='Price',
                             line=dict(color='steelblue')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(y=macd, mode='lines', name='MACD',
                             line=dict(color='steelblue')),
                  row=2, col=1)
    fig.add_trace(go.Scatter(y=signal, mode='lines', name='Signal',
                             line=dict(color='red')),
                  row=2, col=1)
    fig.add_trace(go.Bar(y=hist, name='Histogram',
                         marker=dict(color='gray')),
                  row=2, col=1)
    fig.update_layout(
        title="MACD(12, 26, 9): fast/slow EMA difference plus signal smoothing",
        xaxis_title="Index",
        yaxis_title="Price",
        yaxis2_title="MACD components",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.show()
```

<!-- HELP_END -->

## Reference

Matches `pandas.Series.ewm(span=..., adjust=True).mean()` composition exactly (verified in `tests/test_macd.py`). The underlying EMA is the bias-corrected weighted mean -- the statistically clean form used throughout screamer. TA-Lib's `MACD` uses a different EMA convention (`adjust=False` with an SMA-seeded warmup) and therefore differs from our output by a few percent during early samples; see [conventions](../conventions.md) for the comparison matrix.
