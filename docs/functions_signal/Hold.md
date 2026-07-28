---
name: Hold
title: Hold (time-latch operator)
implementation_family: signal
topics:
- filtering
- logic
tags:
- hold
- latch
- regime
- signal
short: Time-latch operator. Latches a nonzero finite input and holds it for n bars; returns the release value once the hold expires.
inputs: 1
outputs: 1
parameters:
- name: n
  type: int
  default: 3
  min: 1
  description: Number of bars to hold the latched value, including the trigger bar.
    Must be >= 1.
- name: release
  type: float
  default: 0.0
  description: Value returned when no hold is active and the input is zero. May be
    finite or NaN.
nan_policy: ignore
---

# `Hold`

## Description

`Hold` latches a nonzero finite input and outputs it for `n` bars total (the trigger bar plus `n-1` continuation bars). When the hold expires and the next input is zero, the output returns to `release`. A new nonzero input during an active hold replaces the latched value and resets the counter to `n-1` remaining bars. Zero is the trigger-absent sentinel. A NaN input emits NaN at that index and leaves both the held value and the remaining counter unchanged.

$$
\text{output}[t] = \begin{cases}
x[t] & \text{if } x[t] \neq 0 \text{ (latch; set remaining} = n-1\text{)} \\\\
\text{held} & \text{if } x[t] = 0 \text{ and remaining} > 0 \text{ (decrement remaining)} \\\\
\text{release} & \text{if } x[t] = 0 \text{ and remaining} = 0 \\\\
\text{NaN} & \text{if } x[t] = \text{NaN (state unchanged)}
\end{cases}
$$

`reset()` sets `remaining` to 0 and `held` to `release`.

## Parameters

- `n`: number of bars to hold the latched value. The trigger bar counts as bar 1, so a new nonzero input at bar `t` produces the latched value at bars `t`, `t+1`, ..., `t+n-1`. Must be >= 1; the constructor raises `ValueError` otherwise.
- `release`: value emitted when no hold is active (zero input, `remaining == 0`). Defaults to `0.0`. May be finite or NaN; `inf` is rejected.

## Implementation Details

O(1) per step. Two scalars of state: `held` (the most recently latched nonzero value, initialised to `release`) and `remaining` (bars left in the current hold, initialised to 0).

## Examples

### Usage example

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import Hold

    # A sparse signal with nonzero events and stretches of silence.
    rng = np.random.default_rng(42)
    x = np.zeros(120)
    event_idx = rng.choice(120, size=8, replace=False)
    x[event_idx] = rng.standard_normal(8) * 2

    # Hold each event for 10 bars before returning to zero.
    y = Hold(n=10)(x)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.5], vertical_spacing=0.05)
    fig.add_trace(go.Bar(y=x, name='input (sparse events)'), row=1, col=1)
    fig.add_trace(go.Scatter(y=y, mode='lines', name='Hold(n=10)',
                             line=dict(color='red')), row=2, col=1)
    fig.update_layout(
        title="Hold: latch each nonzero event for 10 bars",
        xaxis2_title="Sample",
        yaxis=dict(title="Input"),
        yaxis2=dict(title="Output"),
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    fig.show()
```

<!-- HELP_END -->
