---
name: HawkesIntensity
title: Hawkes Process Conditional Intensity
implementation_family: micro
topics:
- microstructure
tags:
- hawkes
- self-exciting
- intensity
- order flow
- clustering
- microstructure
short: "Conditional intensity of an exponential-kernel Hawkes process: lambda_t = mu + kappa_t."
inputs: 1
outputs: 1
parameters:
- name: decay
  type: float
  default: 0.9
  description: Exponential decay rate applied to the self-exciting kernel each step. Must be in (0, 1).
- name: alpha
  type: float
  default: 1.0
  description: Excitation amplitude; scales how much each event raises the intensity.
- name: mu
  type: float
  default: 0.0
  description: Baseline intensity (the unconditional floor).
nan_policy: ignore
see_also:
- EwMean
---

# `HawkesIntensity`

## Description

`HawkesIntensity` computes the conditional intensity of an exponential-kernel
Hawkes process driven by an event-mark series. A Hawkes process is
self-exciting: each event raises the near-term rate of further events by an
amount proportional to the mark, and that excitation then decays exponentially
over time. In order-flow terms, a burst of aggressive buying momentarily
increases the probability of more buying, capturing the clustering and momentum
structure of trades.

The recursion is:

    lambda_t = mu + kappa_t
    kappa_{t+1} = decay * (kappa_t + alpha * x_t)
    kappa_0 = 0

where `x_t` is the event mark at time `t` (trade count, signed flow magnitude,
or a 0/1 event indicator), `mu` is the baseline intensity, `alpha` scales the
excitation from each event, and `decay` controls how fast the excitation fades.

The operator is causal: `lambda_t` depends only on `x_{t-1}` and earlier. At
`t = 0`, when no history exists, the output is `mu`.

A NaN event mark is treated as missing (nan_policy: ignore): the output at
that step is `NaN`, but `kappa` is not updated, so intensity recovers on the
next finite sample. This prevents a single bad tick from permanently corrupting
the state.

The operator processes one sample per step, so batch and streaming modes produce
identical results.

**References:**

- Hawkes, A. G. (1971). "Spectra of some self-exciting and mutually exciting
  point processes." *Biometrika*, 58(1), 83-90.
- Bacry, E., Mastromatteo, I., and Muzy, J.-F. (2015). "Hawkes processes in
  finance." *Market Microstructure and Liquidity*, 1(1), 1550005.

## Examples

### Basic usage

```python
import numpy as np
from screamer import HawkesIntensity

# Event-mark series: one burst at t=0, then two events at t=3
x = np.array([1.0, 0.0, 0.0, 2.0, 0.0])
intensity = HawkesIntensity(decay=0.9, alpha=1.0, mu=0.0)(x)
# Output: [0.0, 0.9, 0.81, 0.729, 2.4561]
# At t=0: no prior history, intensity = mu = 0
# At t=1: the event at t=0 excites intensity to 0.9
# At t=3: the excitation has decayed to 0.729, then the 2-unit event fires
# At t=4: the combined excitation from t=3 is 0.9*(0.729 + 1.0*2.0) = 2.4561
```

### Streaming one sample at a time

```python
op = HawkesIntensity(decay=0.9, alpha=1.0, mu=0.0)
for v in x:
    lam = op(float(v))   # returns a Python float each call
```

### NaN tolerance

```python
x = np.array([1.0, np.nan, 1.0])
out = HawkesIntensity(decay=0.5, alpha=1.0, mu=0.0)(x)
# out[1] is NaN (missing input), but out[2] is finite (state not poisoned)
```

<!-- HELP_END -->

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import HawkesIntensity

    rng = np.random.default_rng(10)
    n = 300
    events = np.zeros(n)
    bursts = np.sort(rng.choice(n, size=70, replace=False))   # clustered arrivals
    events[bursts] = 1.0
    intensity = HawkesIntensity(decay=0.9, alpha=0.6, mu=0.2)(events)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.35, 0.65],
                        vertical_spacing=0.08)
    fig.add_trace(go.Bar(y=events, name='events', marker_color='lightslategray'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(y=intensity, name='intensity',
                             line=dict(color='purple')), row=2, col=1)
    fig.update_layout(title='HawkesIntensity: each event raises the arrival rate',
                      yaxis=dict(title='event'), yaxis2=dict(title='intensity'),
                      margin=dict(l=20, r=20, t=60, b=20), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
