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
short: ROC of a triple-smoothed EMA.
inputs: 1
outputs: 1
parameters:
- name: span
  type: int
  default: 14
  min: 2
  description: EMA span for each of the three smoothing stages.
nan_policy: ignore
---

# `TRIX`

## Description

`TRIX(span)` is the one-step rate of change of a **triple-smoothed** EMA -- a momentum oscillator that filters short-term noise via two extra layers of exponential smoothing before measuring change.

$$
\begin{aligned}
\text{ema}_1[t] &= \text{EwMean}(\text{span})(x)[t] \\
\text{ema}_2[t] &= \text{EwMean}(\text{span})(\text{ema}_1)[t] \\
\text{ema}_3[t] &= \text{EwMean}(\text{span})(\text{ema}_2)[t] \\
\text{TRIX}[t]  &= 100 \cdot \frac{\text{ema}_3[t] - \text{ema}_3[t-1]}{\text{ema}_3[t-1]}
\end{aligned}
$$

*Parameters*:

- `span` (int, positive): the span for every EMA stage (all three use the same value).

*Warmup*: only the first sample is NaN (no `prev_ema3` yet). Each `EwMean` is well-defined from `t=0`, so from `t=1` onward TRIX is finite.

*NaN handling*: NaN inputs propagate through the EMA arithmetic.

## Implementation Details

Pure composition of three chained `EwMean` instances plus a single scalar holding `prev_ema3`. O(1) per step.

The underlying EMA is pandas's `adjust=True` (bias-corrected weighted mean -- the form we use throughout the library). TA-Lib's TRIX uses `adjust=False` with an SMA-seeded warmup, so ours differs from TA-Lib by a few percent during early samples; see [conventions](../conventions.md).


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

## Reference

Equivalent to `pandas.Series.ewm(span=..., adjust=True).mean()` triple composition followed by `100 * pct_change`. Cross-validated in `tests/test_third_party_alignment.py`: bit-exact vs pandas composition; documented divergence vs TA-Lib's `TRIX`.
