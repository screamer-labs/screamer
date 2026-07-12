---
name: DonchianChannels
title: Donchian Channels
implementation_family: rolling
topics:
- trend
- bands
tags:
- donchian
- channels
- breakout
- envelope
short: 'Trend-following envelope: rolling max(high), rolling min(low), and midline.'
inputs: 2
outputs: 3
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Window for the rolling max/min.
nan_policy: ignore
---

# `DonchianChannels`

## Description

Trend-following envelope. The upper line is the rolling max of `high`, the lower line is
the rolling min of `low`, and the midline is their average:

$$
\begin{aligned}
\text{upper}[t] &= \max(\text{high},\ w\ \text{bars}) \\
\text{lower}[t] &= \min(\text{low},\ w\ \text{bars}) \\
\text{mid}[t]   &= (\text{upper} + \text{lower}) / 2
\end{aligned}
$$

**2-input, 3-output** (`FunctorBase<_, 2, 3>`). Inputs `(high, low)`; outputs
`(lower, mid, upper)`. First valid at sample index `window_size - 1`.

Composes two `detail::MonotonicDeque` instances. Amortised O(1) per step. Bit-exact to
`pandas-ta-classic.donchian`.


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
    from screamer import DonchianChannels

    np.random.seed(0)
    close = 100*np.exp(np.cumsum(np.random.normal(0, 0.01, size=300)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    wick = np.abs(np.random.normal(0, 0.4, size=300))
    high = np.maximum(open_, close) + wick
    low  = np.minimum(open_, close) - wick
    out = np.asarray(DonchianChannels(20)(high, low))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=high, name="high", line=dict(color="#888")), row=1, col=1)
    fig.add_trace(go.Scatter(y=low, name="low", line=dict(color="#bbb")), row=1, col=1)
    fig.add_trace(go.Scatter(y=out[:, 2], name="upper", line=dict(color="red")), row=2, col=1)
    fig.add_trace(go.Scatter(y=out[:, 1], name="mid", line=dict(color="gray")), row=2, col=1)
    fig.add_trace(go.Scatter(y=out[:, 0], name="lower", line=dict(color="green")), row=2, col=1)
    fig.update_layout(title="Donchian channels (DonchianChannels)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="high / low", row=1, col=1)
    fig.update_yaxes(title_text="channel bands", row=2, col=1)
    fig.show()
```

<!-- HELP_END -->
