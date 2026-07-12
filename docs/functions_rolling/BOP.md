---
name: BOP
title: Balance of Power (BOP)
implementation_family: rolling
topics:
- momentum
tags:
- bop
- oscillator
- talib
- ohlc
short: (close - open) / (high - low) per bar. No smoothing.
inputs: 4
outputs: 1
parameters: []
nan_policy: ignore
---

# `BOP`

## Description

`BOP` (Balance of Power, Igor Livshin) measures whether a bar closes near its high (buyers in control) or its low (sellers in control):

$$
\text{BOP}[t] = \frac{\text{close} - \text{open}}{\text{high} - \text{low}}
$$

Output range is `[-1, +1]` for any sensibly-formed bar (where `low ≤ open, close ≤ high`).

**4-input, 1-output** (`FunctorBase<_, 4, 1>`). Argument order matches TA-Lib's `BOP`: `(open, high, low, close)`.

*Warmup*: none -- stateless, value defined for every input.

*Range-zero*: returns `0` when `high == low` (flat bar; convention matches TA-Lib).


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
    from screamer import BOP

    np.random.seed(0)
    close = 100*np.exp(np.cumsum(np.random.normal(0, 0.01, size=300)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    wick = np.abs(np.random.normal(0, 0.4, size=300))
    high = np.maximum(open_, close) + wick
    low  = np.minimum(open_, close) - wick
    out = BOP()(open_, high, low, close)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=close, name="close", line=dict(color="royalblue")), row=1, col=1)
    fig.add_trace(go.Scatter(y=out, name="BOP", line=dict(color="red")), row=2, col=1)
    fig.update_layout(title="Balance of power (BOP)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="close", row=1, col=1)
    fig.update_yaxes(title_text="BOP (-1 to 1)", row=2, col=1)
    fig.show()
```

<!-- HELP_END -->

## Implementation Details

Single per-step arithmetic operation; O(1) per step, zero state.

## Reference

Bit-exact match to `talib.BOP(open, high, low, close)` (verified to 0.0 -- exact integer-rounded arithmetic).
