---
name: Delay
title: Delay
kind: class
short: Re-stamp each event's index by a fixed time offset (a latency line).
topics:
- streams
covers:
- delay
---

# `Delay`

`Delay(duration)` shifts every event's index by `duration` (in index units) and
leaves its value unchanged: event `(t, v)` becomes `(t + duration, v)`. It is the
time-based counterpart of `Lag`, which shifts by a fixed number of events. On a
regular grid the two coincide (`duration = N * step` equals `Lag(N)`); on an
irregular feed `Delay` shifts by wall-time and `Lag` shifts by event count.

The shift is lossless, one output per input, order-preserving, and starts `duration`
late (no NaN warmup). `duration` is numeric in index units (a millisecond index uses
`Delay(600_000)` for a 10-minute delay); there is no calendar parsing.

`Delay` requires an explicit index (there is nothing to shift against without one);
calling it on a bare array is a `TypeError`.

## Parameters

- `duration`: the index offset to add to every event, numeric, in index units.

## Limitations

A `Delay` feeding a downstream merge (`CombineLatest`) inside a single live
`Pipeline` emits the delayed event eagerly, so the merge would align it against a
not-yet-advanced input. For an as-of alignment across a delay, apply `Delay` and
`CombineLatest` as separate calls (batch `CombineLatest` sees all events), which is
what `forecast_pairs` does. A fused-live-merge form needs a reorder buffer and is a
planned follow-on.

## Examples

### Delaying an irregular feed

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from screamer import Delay

    idx = np.array([0, 7, 14, 21, 28], dtype=np.int64)
    vals = np.array([1.0, 3.0, 2.0, 4.0, 3.5])
    dv, di = Delay(5)(vals, idx)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=vals, mode="lines+markers", name="input"))
    fig.add_trace(go.Scatter(x=di, y=dv, mode="lines+markers", name="Delay(5)"))
    fig.update_layout(title="Delay shifts the index by 5 units",
                      xaxis_title="index", yaxis_title="value",
                      margin=dict(l=20, r=20, t=50, b=20))
    fig.show()
```

<!-- HELP_END -->

## Reference

A delay line from signal processing. Composes with `CombineLatest` to build an
as-of-lagged join, and underpins `forecast_pairs(duration=)`.
