---
name: ROCR
title: Rate of change ratio (ROCR)
implementation_family: fin
topics:
- returns
tags:
- rocr
- rate-of-change
- talib
short: x[t] / x[t-k] - TA-Lib's ROCR.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 10
  min: 1
  description: Lookback k.
nan_policy: propagate
---

# `ROCR`

## Description

`ROCR(k)` is the **rate of change** over `k` steps, expressed as a *ratio*:

$$
\text{ROCR}[t] = \frac{x[t]}{x[t-k]}
$$

The ratio form is convenient when you want to chain returns multiplicatively (e.g. cumulative-product wealth paths) without subtracting 1 each step.

*Parameters*:

- `window_size` (int, positive): the lookback `k`.

*NaN handling*: NaN for the first `k` samples; NaN when `x[t-k] == 0`.

## Identity to ROCP

`ROCR(k) - 1 == ROCP(k)` exactly. Pick whichever form keeps the calling code cleaner.


<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `propagate`.** Input `NaN` values are stored in the lookback. Output is `NaN` at any index where the function's positional formula references a `NaN` input; recovery happens once the `NaN` slides out of the lookback.
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage example

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import ROCR

    np.random.seed(0)
    price = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, size=300)))
    rocr = ROCR(window_size=20)(price)      # price ratio versus 20 bars ago, 1.0 = unchanged

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=price, mode="lines", name="price"), row=1, col=1)
    fig.add_trace(go.Scatter(y=rocr, mode="lines", name="ROCR",
                             line=dict(color="red")), row=2, col=1)
    fig.update_layout(title="Price ratio over 20 bars (ROCR)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="ratio", row=2, col=1)
    fig.show()
```

<!-- HELP_END -->

## Reference

Equivalent to `talib.ROCR(x, timeperiod=k)`. Bit-exact match (cross-validated in `tests/test_third_party_alignment.py`).
