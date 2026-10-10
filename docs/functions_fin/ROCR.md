---
name: ROCR
title: Rate of change ratio (ROCR)
implementation_family: fin
topics:
- returns
tags:
- rocr
- rate-of-change
- rate of change
- price relative
- gross return
- talib
short: Rate of change ratio x[t] / x[t-k] (TA-Lib ROCR).
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 10
  min: 1
  description: 'Lookback k: the output compares each value with the value k steps earlier.
    TA-Lib uses 10.'
nan_policy: propagate
see_also:
- ROCP
- ROC
- Return
- LogReturn
---

# `ROCR`

## Description

`ROCR(k)` is the **rate of change** over `k` steps, expressed as a *ratio*:

$$
\text{ROCR}[t] = \frac{x[t]}{x[t-k]}
$$

The first `k` outputs are `NaN` because `x[t-k]` does not exist yet. The division follows IEEE arithmetic, so `x[t-k] == 0` gives `inf` (or `NaN` when `x[t]` is also zero).

`ROCR(k) - 1` equals [`ROCP(k)`](ROCP.md).


<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `propagate`.** A `NaN` input is stored in the lookback like any other value. The output is `NaN` wherever the formula references it and recovers once it leaves the lookback. See [NaN and warmup](../nan_and_warmup.md).
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

## TA-Lib compatibility

Equivalent to `talib.ROCR(x, timeperiod=k)`. The results agree exactly (checked in `tests/test_third_party_alignment.py`).
