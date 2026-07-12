---
name: CCI
title: Commodity Channel Index (CCI)
implementation_family: rolling
topics:
- momentum
tags:
- cci
- lambert
- oscillator
- talib
- hlc
short: Commodity Channel Index over typical price.
inputs: 3
outputs: 1
parameters:
- name: window_size
  type: int
  default: 14
  min: 2
  description: Period (Wilder's default).
nan_policy: ignore
---

# `CCI`

## Description

`CCI` (Commodity Channel Index, Donald Lambert) measures how far the current bar's typical price has moved from its rolling mean, normalised by the mean absolute deviation of the same window:

$$
\begin{aligned}
\text{TP}[t]      &= (\text{high} + \text{low} + \text{close}) / 3 \\
\overline{\text{TP}} &= \text{SMA}(\text{TP},\ n) \\
\text{MAD}        &= \text{mean}\big(\ |\text{TP} - \overline{\text{TP}}|\ \big) \quad \text{over the same window} \\
\text{CCI}[t]     &= \frac{\text{TP}[t] - \overline{\text{TP}}[t]}{0.015 \cdot \text{MAD}[t]}
\end{aligned}
$$

The 0.015 constant is a Lambert convention: roughly 70-80% of CCI readings fall in `[-100, +100]` for a normal-distributed input.

**3-input, 1-output** (`FunctorBase<_, 3, 1>`) on `(high, low, close)`.

*Parameters*:

- `window_size` (int, default `14`).

*Warmup*: NaN for the first `window_size - 1` samples.

*Range-zero*: returns `0` when the MAD is 0 (a perfectly flat window).

*NaN handling*: NaN inputs poison the rolling sum.


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
    from screamer import CCI

    np.random.seed(0)
    close = 100*np.exp(np.cumsum(np.random.normal(0, 0.01, size=300)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    wick = np.abs(np.random.normal(0, 0.4, size=300))
    high = np.maximum(open_, close) + wick
    low  = np.minimum(open_, close) - wick
    out = CCI(20)(high, low, close)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=high, name="high", line=dict(color="#888")), row=1, col=1)
    fig.add_trace(go.Scatter(y=low, name="low", line=dict(color="#bbb")), row=1, col=1)
    fig.add_trace(go.Scatter(y=close, name="close", line=dict(color="royalblue")), row=1, col=1)
    fig.add_trace(go.Scatter(y=out, name="CCI(20)", line=dict(color="red")), row=2, col=1)
    fig.update_layout(title="Commodity channel index (CCI)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="CCI", row=2, col=1)
    fig.show()
```

<!-- HELP_END -->

## Implementation Details

Holds one circular buffer of TP values with a rolling sum (the SMA part is incremental O(1)); the per-step MAD sweep over the window is `O(window_size)`. Same trade-off as `RollingMad`, and the same reason: there is no closed-form O(1) MAD when the mean shifts each step.

## Reference

Bit-exact match to `talib.CCI(high, low, close, timeperiod)` post-warmup (verified to ~1e-11 in `tests/test_third_party_alignment.py`).
