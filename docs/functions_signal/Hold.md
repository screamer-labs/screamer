---
name: Hold
title: Hold (time-latch operator)
implementation_family: signal
topics:
- filtering
- logic
- signals
tags:
- hold
- latch
- regime
- signal
- sample and hold
- time latch
short: Time-latch operator. Latches a nonzero finite input and holds it for n steps; returns
  the release value once the hold expires.
inputs: 1
outputs: 1
parameters:
- name: n
  type: int
  default: 3
  min: 1
  description: 'Number of steps the latched value is held, counting the trigger step: a nonzero
    input at t is output at t, t+1, ..., t+n-1.'
- name: release
  type: float
  default: 0.0
  description: Value returned when no hold is active and the input is zero. May be finite
    or NaN; inf is rejected.
nan_policy: ignore
see_also:
- SchmittTrigger
- BandSignal
- Last
- Lag
---

# `Hold`

## Description

`Hold` latches a nonzero finite input and outputs it for `n` steps in total: the trigger step plus `n-1` continuation steps. When the hold expires and the next input is zero, the output returns to `release`. A new nonzero input during an active hold replaces the latched value and resets the counter to `n-1` remaining steps. A zero input means no trigger.

$$
\text{output}[t] = \begin{cases}
x[t] & \text{if } x[t] \neq 0 \text{ (latch; set remaining} = n-1\text{)} \\\\
\text{held} & \text{if } x[t] = 0 \text{ and remaining} > 0 \text{ (decrement remaining)} \\\\
\text{release} & \text{if } x[t] = 0 \text{ and remaining} = 0 \\\\
\text{NaN} & \text{if } x[t] = \text{NaN (state unchanged)}
\end{cases}
$$

`reset()` sets `remaining` to 0 and `held` to `release`.

## Implementation Details

O(1) per step. Two scalars of state: `held` (the most recently latched nonzero value, initialized to `release`) and `remaining` (steps left in the current hold, initialized to 0).


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
