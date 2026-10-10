---
name: ROC
title: Rate of change (ROC)
implementation_family: fin
topics:
- returns
- momentum
tags:
- roc
- rate-of-change
- rate of change
- pct_change
- percent change
- percentage change
- talib
short: Rate of change 100 * (x[t] / x[t-k] - 1), in percent (TA-Lib ROC).
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
- ROCR
- Return
- Momentum
---

# `ROC`

## Description

`ROC(k)` is the **rate of change** over `k` steps, expressed as a percentage:

$$
\text{ROC}[t] = 100 \cdot \frac{x[t] - x[t-k]}{x[t-k]}
$$

The three rate-of-change variants in this library, all matching their TA-Lib namesakes:

| Class | Formula | Returns |
|---|---|---|
| **`ROC`** | $100 \cdot (x[t] - x[t-k]) / x[t-k]$ | percentage (e.g. `5.0` = +5%) |
| `ROCP` | $(x[t] - x[t-k]) / x[t-k]$ | fraction (e.g. `0.05` = +5%); same as [`Return`](Return.md) |
| `ROCR` | $x[t] / x[t-k]$ | ratio (e.g. `1.05`) |

The first `k` outputs are `NaN` because `x[t-k]` does not exist yet. The division follows IEEE arithmetic, so `x[t-k] == 0` gives `inf` (or `NaN` when `x[t]` is also zero).


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
    from screamer import ROC

    np.random.seed(0)
    price = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, size=300)))
    roc = ROC(window_size=20)(price)        # percent change over the last 20 bars

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=price, mode="lines", name="price"), row=1, col=1)
    fig.add_trace(go.Scatter(y=roc, mode="lines", name="ROC",
                             line=dict(color="red")), row=2, col=1)
    fig.update_layout(title="Rate of change over 20 bars (ROC)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="percent", row=2, col=1)
    fig.show()
```

<!-- HELP_END -->

## TA-Lib compatibility

Equivalent to `talib.ROC(x, timeperiod=k)`. The results agree to about 1e-14 (checked in `tests/test_third_party_alignment.py`).
