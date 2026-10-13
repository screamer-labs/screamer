---
name: MACD
title: MACD
implementation_family: rolling
topics:
- trend
- momentum
tags:
- macd
- ema
- appel
- talib
- moving average convergence divergence
short: MACD line, signal line, and histogram (3 outputs).
inputs: 1
outputs: 3
parameters:
- name: fast
  type: int
  default: 12
  min: 2
  description: Span of the fast EMA, in samples. Must be less than slow.
- name: slow
  type: int
  default: 26
  min: 3
  description: Span of the slow EMA, in samples. Must be greater than fast.
- name: signal
  type: int
  default: 9
  min: 2
  description: Span of the EMA applied to the MACD line to form the signal line, in samples.
output_names:
- macd
- signal
- histogram
see_also:
- EwMean
- TRIX
- DEMA
- Momentum
references:
- 'Appel, G. (2005). *Technical Analysis: Power Tools for Active Investors*. FT Prentice Hall.'
nan_policy: ignore
---

# `MACD`

## Description

`MACD` (Moving Average Convergence Divergence, Gerald Appel) is a momentum indicator
that subtracts a slow EMA from a fast EMA and then smooths the difference. It returns
three columns per step, `macd`, `signal` and `histogram`:

$$
\begin{aligned}
\text{macd}[t]      &= \text{EMA}_\text{fast}(x)[t] - \text{EMA}_\text{slow}(x)[t] \\
\text{signal}[t]    &= \text{EMA}_\text{signal}(\text{macd})[t] \\
\text{histogram}[t] &= \text{macd}[t] - \text{signal}[t]
\end{aligned}
$$

The defaults `12, 26, 9` are Appel's choice and the usual charting default. All three
EMAs are [`EwMean`](../functions_ew/EwMean.md).

There is no warmup. `EwMean` is defined from the first sample (`EMA[0] = x[0]`), so at
t=0 both EMAs equal `x[0]` and all three outputs are 0. Each step costs `O(1)` time
and memory.

| You pass... | You get back... |
|---|---|
| scalar `x` | `tuple(macd, signal, histogram)` |
| 1D array shape `(T,)` | array shape `(T, 3)` |
| 2D array shape `(T, K)` | array shape `(T, K, 3)`, column-by-column |

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input gives a `NaN` output at that step. The step is skipped: it does not enter the window or update the internal state. See [NaN and warmup](../nan_and_warmup.md).
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

## Validation

Matches the `pandas.Series.ewm(span=..., adjust=True).mean()` composition exactly, in
`tests/test_macd.py`. The EMA is the bias-corrected weighted mean used throughout
screamer. TA-Lib's `MACD` uses a different EMA (`adjust=False` with a warmup seeded by
a simple average), so it differs by a few percent during the early samples; see
[conventions](../conventions.md) for the comparison.
