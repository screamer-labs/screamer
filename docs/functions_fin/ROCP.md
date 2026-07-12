---
name: ROCP
title: Rate of change percent (ROCP)
implementation_family: fin
topics:
- returns
tags:
- rocp
- rate-of-change
- talib
short: x[t] / x[t-k] - 1 - TA-Lib's ROCP. Identical to Return.
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

# `ROCP`

## Description

`ROCP(k)` is the **rate of change** over `k` steps, expressed as a *fraction*:

$$
\text{ROCP}[t] = \frac{x[t] - x[t-k]}{x[t-k]}
$$

**This is exactly `Return(k)`.** It exists as a separately-named class because TA-Lib calls this indicator `ROCP`. Internally `ROCP` is a thin subclass of `Return` -- the implementation (delay buffer + subtract + divide) is shared, not duplicated.

| You want... | Use |
|---|---|
| TA-Lib parity (writing `ROCP` in ported code) | `ROCP(k)` |
| Any other context (returns workflows, log-returns family) | `Return(k)` |

*Parameters*:

- `window_size` (int, positive): the lookback `k`.

*NaN handling*: NaN for the first `k` samples; NaN when `x[t-k] == 0`.

## See also

- [`Return`](Return.md) -- same class, the documentation lives there.
- [`ROC`](ROC.md) -- `100 * ROCP` (percentage form).
- [`ROCR`](ROCR.md) -- `x[t] / x[t-k]` (ratio form).


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
    from screamer import ROCP

    np.random.seed(0)
    price = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, size=300)))
    rocp = ROCP(window_size=20)(price)      # fractional change over the last 20 bars

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=price, mode="lines", name="price"), row=1, col=1)
    fig.add_trace(go.Scatter(y=rocp, mode="lines", name="ROCP",
                             line=dict(color="red")), row=2, col=1)
    fig.update_layout(title="Fractional rate of change over 20 bars (ROCP)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="fraction", row=2, col=1)
    fig.show()
```

<!-- HELP_END -->

## Reference

Equivalent to `talib.ROCP(x, timeperiod=k)`. Bit-exact match (cross-validated in `tests/test_third_party_alignment.py`).
