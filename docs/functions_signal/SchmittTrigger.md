---
name: SchmittTrigger
title: Schmitt trigger (hysteresis comparator)
implementation_family: signal
topics:
- signal-processing
- transforms
tags:
- schmitt
- trigger
- hysteresis
- comparator
- binary
short: Hysteresis comparator. Latches 1.0 above the upper threshold, 0.0 below the lower threshold, and retains its previous value in between.
inputs: 1
outputs: 1
parameters:
- name: lower
  type: float
  default: -1.0
  description: Lower threshold. The output latches to 0.0 when the input falls strictly
    below this value.
- name: upper
  type: float
  default: 1.0
  description: Upper threshold. The output latches to 1.0 when the input rises strictly
    above this value. Must be strictly greater than `lower`.
nan_policy: ignore
---

# `SchmittTrigger`

## Description

The Schmitt trigger (Otto Schmitt, 1934) is a hysteresis comparator: a circuit whose output changes only when the input crosses the *opposite* threshold from the one that last triggered it. This double-threshold behavior gives the trigger its noise immunity — a noisy input that briefly dips back across a single threshold cannot rapidly toggle the output.

$$
\text{output}[t] = \begin{cases}
1.0 & \text{if } x[t] > \text{upper} \\\\
0.0 & \text{if } x[t] < \text{lower} \\\\
\text{output}[t-1] & \text{otherwise}
\end{cases}
$$

The window `[lower, upper]` is the *dead band*. Inside it the output is latched: whichever value (1.0 or 0.0) the trigger last committed to is held until the input crosses out the other side.

Until the first input crosses either threshold the output is `NaN` — the trigger has no prior state to retain.

## Parameters

- `lower`: lower threshold. Must be strictly less than `upper`. The output latches to `0.0` when the input falls below this value.
- `upper`: upper threshold. The output latches to `1.0` when the input rises above this value.

## Implementation Details

O(1) per step. One scalar of state (the latched output). The constructor rejects `lower >= upper` and non-finite thresholds; `reset()` clears the latched state back to `NaN`.

## Examples

### Usage example

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import SchmittTrigger

    # A noisy sine wave that repeatedly crosses zero.
    t = np.linspace(0, 4 * np.pi, 400)
    x = np.sin(t) + 0.25 * np.random.default_rng(0).standard_normal(400)

    # A naive zero-cross comparator would chatter on the noise. The
    # Schmitt trigger requires the signal to swing past +/- 0.3 before
    # toggling, so it produces a clean square wave.
    trig = SchmittTrigger(lower=-0.3, upper=0.3)
    y = trig(x)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.05)
    fig.add_trace(go.Scatter(y=x, mode='lines', name='input'),     row=1, col=1)
    fig.add_hline(y=+0.3, line_dash='dash', line_color='gray', row=1, col=1)
    fig.add_hline(y=-0.3, line_dash='dash', line_color='gray', row=1, col=1)
    fig.add_trace(go.Scatter(y=y, mode='lines', name='trigger',
                             line=dict(color='red')),             row=2, col=1)
    fig.update_layout(
        title="Schmitt trigger on a noisy sine wave (lower=-0.3, upper=+0.3)",
        xaxis2_title="Sample", yaxis=dict(title="Input"),
        yaxis2=dict(title="Trigger", range=[-0.2, 1.2]),
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    fig.show()
```

<!-- HELP_END -->

## Reference

A simple debouncer pattern from analog electronics. The exact same logic appears in microcontroller GPIO input filtering, regime-classification rules in algorithmic trading (e.g. "go long above `+x`, flat between `-x` and `+x`, short below `-x`"), and any signal-flow stage that needs to convert a continuous signal into a sticky binary state.
