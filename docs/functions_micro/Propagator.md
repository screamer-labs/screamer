---
name: Propagator
title: Bouchaud Propagator Price Impact
implementation_family: micro
topics:
- microstructure
tags:
- propagator
- price impact
- order flow
- long memory
- bouchaud
- microstructure
short: "Bouchaud (2004) propagator model: price impact as a decaying-kernel convolution over past signed order flow."
inputs: 1
outputs: 1
parameters:
- name: window
  type: int
  default: 20
  min: 2
  description: Number of past flow samples included in the convolution (the kernel support).
- name: g0
  type: float
  default: 1.0
  description: Kernel amplitude; scales the overall impact magnitude.
- name: gamma
  type: float
  default: 0.5
  description: Power-law decay exponent. Larger values make the kernel decay faster.
nan_policy: propagate
see_also:
- RollingKyleLambda
- HawkesIntensity
---

# `Propagator`

## Description

`Propagator` computes the predicted price impact at each time step by convolving
past signed order flow with a power-law decaying kernel. The model was introduced
by Bouchaud, Gefen, Potters, and Wyart (2004) to capture the empirical observation
that the price impact of a trade does not disappear instantly but decays slowly
over many subsequent periods.

The predicted impact at time `t` is:

    impact_t = sum_{k=0}^{window-1} G(k) * flow_{t-k}

where the propagator kernel is:

    G(k) = g0 * (k + 1)^(-gamma)

Here `flow_{t-k}` is the signed order flow `k` periods in the past. The kernel
`G` assigns full weight `g0` to the current period (`k = 0`) and gradually
decreasing weight to older flow, following a power law with exponent `gamma`.
When `gamma = 0.5` (the default), the memory decays slowly enough to be called
long-memory: past flow continues to affect the predicted price even after many
periods.

The operator requires `window` samples before producing its first output. The
first `window - 1` outputs are `NaN` (warmup period).

Because this is a positional (FIR) filter, it follows the `propagate` NaN policy,
like `Lag` and `Diff`: a NaN flow value is kept in the window and flows through
the convolution, so the output is NaN while the NaN is inside the window and
recovers once it drops out. Dropping the NaN instead (as an `ignore`-policy
statistic would) would misalign the kernel with the wrong lags.

The operator processes one sample per step, so batch and streaming modes produce
identical results.

**References:**

- Bouchaud, J.-P., Gefen, Y., Potters, M., and Wyart, M. (2004). "Fluctuations
  and response in financial markets: the subtle nature of 'random' price changes."
  *Quantitative Finance*, 4(2), 176-190.

## Examples

### Basic usage

```python
import numpy as np
from screamer import Propagator

# Isolated unit of buy flow at t=0 and t=3; window=3 reveals the kernel shape
flow = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
impact = Propagator(window=3, g0=1.0, gamma=0.5)(flow)
# G = [1.0, 2^-0.5, 3^-0.5] = [1.0, 0.70711, 0.57735]
# t=0, t=1: NaN (warmup, fewer than window samples seen)
# t=2: G[0]*flow[2] + G[1]*flow[1] + G[2]*flow[0] = 0 + 0 + 0.57735 = 0.57735
# t=3: G[0]*flow[3] + G[1]*flow[2] + G[2]*flow[1] = 1.0 + 0 + 0 = 1.0
# t=4: G[0]*flow[4] + G[1]*flow[3] + G[2]*flow[2] = 0 + 0.70711 + 0 = 0.70711
# t=5: G[0]*flow[5] + G[1]*flow[4] + G[2]*flow[3] = 0 + 0 + 0.57735 = 0.57735
```

### Streaming one sample at a time

```python
op = Propagator(window=20, g0=1.0, gamma=0.5)
for signed_flow in flow_stream:
    predicted_impact = op(float(signed_flow))
```

### Reset clears accumulated history

```python
op = Propagator(window=10)
impact_first_pass = op(flow)
op.reset()
impact_second_pass = op(flow)   # identical to first pass
```

<!-- HELP_END -->

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import Propagator

    # A single unit of buy flow, so the output traces out the impact kernel: impact
    # builds and then relaxes through the decaying propagator.
    flow = np.zeros(80)
    flow[10] = 1.0
    impact = Propagator(window=40, g0=1.0, gamma=0.5)(flow)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.35, 0.65],
                        vertical_spacing=0.08)
    fig.add_trace(go.Bar(y=flow, name='signed flow', marker_color='seagreen'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(y=impact, name='price impact',
                             line=dict(color='teal')), row=2, col=1)
    fig.update_layout(title='Propagator: impact of one trade decays over time',
                      yaxis=dict(title='flow'), yaxis2=dict(title='impact'),
                      margin=dict(l=20, r=20, t=60, b=20), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
