---
name: TRIX
title: TRIX (rate of change of triple EMA)
implementation_family: rolling
topics:
- trend
- momentum
tags:
- trix
- ema
- rate-of-change
- talib
- triple exponential
- momentum
- oscillator
short: ROC of a triple-smoothed EMA.
inputs: 1
outputs: 1
parameters:
- name: span
  type: int
  default: 14
  min: 2
  description: Span of each of the three exponential moving averages, giving alpha = 2 / (span
    + 1).
nan_policy: ignore
see_also:
- TEMA
- EwMean
- ROCP
- MACD
references:
- Hutson, J. K. (1983). Good TRIX. *Technical Analysis of Stocks & Commodities*, 1(5).
---

# `TRIX`

## Description

`TRIX(span)` (Jack Hutson, 1983) is the one-step percentage rate of change of a triple-smoothed EMA. It is a momentum oscillator: the two extra layers of exponential smoothing filter out short-term noise before the change is measured.

$$
\begin{aligned}
\text{ema}_1[t] &= \text{EwMean}(\text{span})(x)[t] \\
\text{ema}_2[t] &= \text{EwMean}(\text{span})(\text{ema}_1)[t] \\
\text{ema}_3[t] &= \text{EwMean}(\text{span})(\text{ema}_2)[t] \\
\text{TRIX}[t]  &= 100 \cdot \frac{\text{ema}_3[t] - \text{ema}_3[t-1]}{\text{ema}_3[t-1]}
\end{aligned}
$$

All three stages use the same `span`. Only the first output is NaN, because there is no previous `ema3` yet; each `EwMean` is defined from `t=0`, so TRIX is finite from `t=1` onward.

## Implementation Details

`TRIX` chains three `EwMean` stages and keeps the previous `ema3` value. Each step is O(1).

The underlying EMA follows pandas `adjust=True` (the bias-corrected weighted mean used throughout the library). TA-Lib's TRIX uses `adjust=False` with an SMA-seeded warmup, so ours differs from TA-Lib by a few percent during early samples; see [conventions](../conventions.md).


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
    from screamer import TRIX

    rng = np.random.default_rng(0)
    N = 300
    data = 100.0 + np.cumsum(rng.standard_normal(N))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5],
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=data, mode='lines', name='Input', line=dict(color='steelblue', width=1)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(y=TRIX(span=14)(data), mode='lines', name='TRIX(span=14)',
                             line=dict(color='crimson', width=2)), row=2, col=1)
    fig.update_layout(
        title="TRIX momentum oscillator over a price series",
        yaxis=dict(title='Input'), yaxis2=dict(title='TRIX'),
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    fig.show()
```

<!-- HELP_END -->

## Compatibility

Equivalent to `pandas.Series.ewm(span=..., adjust=True).mean()` triple composition followed by `100 * pct_change`. Cross-validated in `tests/test_third_party_alignment.py`: bit-exact vs pandas composition; documented divergence vs TA-Lib's `TRIX`.
