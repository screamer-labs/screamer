---
name: RollingRSI
title: Rolling RSI
implementation_family: rolling
topics:
- momentum
tags:
- rsi
- wilder
- cutler
- momentum
- oscillator
- relative strength index
short: Relative Strength Index. Wilder's smoothing by default; Cutler's via method='cutler'.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 14
  min: 2
  description: Smoothing period of the average up and down moves. Wilder's original choice
    is 14.
- name: method
  type: str
  default: wilder
  enum:
  - wilder
  - cutler
  description: 'Smoothing of the up and down moves: "wilder" uses Wilder''s recursive average
    (matches TA-Lib and pandas-ta), "cutler" uses a simple moving average.'
- name: start_policy
  type: str
  default: strict
  enum:
  - strict
  - expanding
  - zero
  description: 'Output before the window is full: "strict" gives NaN, "expanding" computes
    from the samples seen so far, and "zero" treats the window as pre-filled with zeros.'
nan_policy: ignore
see_also:
- StochRSI
- MFI
- Stoch
- WilliamsR
references:
- Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.
---

# `RollingRSI`

## Description

Wilder's Relative Strength Index. Up- and down-moves are separately smoothed; the index
is `100 - 100 / (1 + RS)` where `RS` is the smoothed ratio of average up to average down
moves:

$$
\begin{aligned}
\Delta[t] &= x[t] - x[t-1] \\
U[t] &= \max(\Delta, 0),\quad D[t] = \max(-\Delta, 0) \\
\overline{U}[t] &= \text{smooth}(U,\ \text{window\_size}),\quad \overline{D}[t] = \text{smooth}(D,\ \text{window\_size}) \\
\text{RS}[t] &= \overline{U}[t] / \overline{D}[t] \\
\text{RSI}[t] &= 100 - 100 / (1 + \text{RS})
\end{aligned}
$$

The smoothing depends on `method`:

- `'wilder'` (default): Wilder's smoothing
  $\overline{X}[t] = \overline{X}[t-1] + (X[t] - \overline{X}[t-1]) / w$, seeded with
  the simple mean of the first `w` values. Matches `talib.RSI` and `pandas-ta-classic.rsi`
  bit-exactly after warmup.
- `'cutler'`: Cutler's variant, a plain rolling mean of the last `w` values. It does
  not match TA-Lib.

The first valid output is at sample index `window_size`, and the output lies in `[0, 100]`.


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
    from screamer import RollingRSI

    np.random.seed(0)
    price = 100*np.exp(np.cumsum(np.random.normal(0.0005, 0.02, size=300)))
    rsi = RollingRSI(window_size=14)(price)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=price, name="price"), row=1, col=1)
    fig.add_trace(go.Scatter(y=rsi, name="RollingRSI", line=dict(color="red")), row=2, col=1)
    fig.update_layout(title="Price with RSI (RollingRSI)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="RSI (0-100)", row=2, col=1)
    fig.show()
```

<!-- HELP_END -->
