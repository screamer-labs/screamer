---
name: Segment
title: Segment-wise operator
implementation_family: misc
topics:
- cumulative
- streams
tags:
- segment
- groupby
- group by
- reset
- per day
- session
- month to date
- intraday
- restart
short: Runs an operator on each segment of rows with the same label, resetting it where the
  label changes.
inputs: 2
outputs: 1
parameters:
- name: op
  type: op
  default: CumSum()
  required: true
  description: The operator to run on each segment. Segment holds its own copy, so the
    instance passed in is never used or changed.
nan_policy: nan-aware
see_also:
- CumSum
- ExpandingMean
- Resample
---

# `Segment`

## Description

`Segment(op)` runs the operator `op` on each segment of a series from a fresh state. It takes
the segment label first and then the inputs of `op`, `Segment(op)(label, *inputs)`, and
returns the outputs of `op` at every row, in the shape `op` returns them. An operator of
`n` inputs and `m` outputs gives a `Segment` of `1 + n` inputs and `m` outputs.

A segment is a run of rows with the same label. At each row whose label differs from the
last non-`NaN` label, `Segment` resets its copy of `op`; then it evaluates `op` on the row.
So a label that returns to an earlier value starts a new segment, and the result is pandas
`groupby(...).transform(...)` over the runs of the label: with
`runs = (label != label.shift()).cumsum()`, `Segment(CumSum())(label, x)` is
`x.groupby(runs).cumsum()`.

A label is any number: a day number, a session id, a month index, a regime from a
classifier. `Segment` emits one row per input row, unlike `Resample`, which emits one row
per bucket.

`Segment` holds a copy of `op` made when it is built, and `Segment.reset()` resets that copy
and forgets the current label. A `Segment` works wherever an operator does: on arrays, one
event at a time, on iterators, and in a `Pipeline`.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `nan-aware`.** A `NaN` label keeps the current segment: the row is evaluated by
`op` and nothing is reset. Rows before the first non-`NaN` label belong to no segment and
give `NaN` without reaching `op`. A `NaN` in the inputs of `op` is handled by `op`'s own
policy. See [NaN and warmup](../nan_and_warmup.md).
<!-- NAN_FOOTNOTE_END -->

## Examples

### Per-day rolling volatility and month-to-date sum

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import CumSum, RollingStd, Segment

    rng = np.random.default_rng(0)
    time = np.datetime64("2024-01-29T00:00") + np.arange(6 * 24 * 12) * np.timedelta64(5, "m")
    returns = rng.normal(0.0, 1.0, time.size) * np.where((time.astype("datetime64[D]").astype(int) % 2) == 0, 0.5, 1.5)

    # The label of a row is its day (or its month) as a number. RollingStd(30)
    # starts over each day, with its warmup of 29 NaN rows, and CumSum()
    # restarts at each month.
    day = time.astype("datetime64[D]").astype(float)
    month = time.astype("datetime64[M]").astype(float)

    daily_vol = Segment(RollingStd(30))(day, returns)
    month_to_date = Segment(CumSum())(month, returns)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
    fig.add_trace(go.Scatter(x=time, y=daily_vol, mode='lines', name='RollingStd(30), per day'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=time, y=month_to_date, mode='lines', name='CumSum, month to date'),
                  row=2, col=1)
    fig.update_layout(
        title="Segment: per-day rolling volatility and month-to-date sum",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.show()
```

### Two inputs and several outputs

```python
import numpy as np
from screamer import BollingerBands, RollingCorr, Segment

# op keeps its arity: RollingCorr takes two inputs, so its Segment takes
# three, and BollingerBands returns three outputs per row.
session = np.repeat([1.0, 2.0, 3.0], 100)
x, y = np.random.default_rng(1).normal(size=(2, 300))

corr = Segment(RollingCorr(20))(session, x, y)          # shape (300,)
bands = Segment(BollingerBands(20))(session, x)         # shape (300, 3)
```

<!-- HELP_END -->

## Implementation Details

On an array, `Segment` finds the segment boundaries in one pass over the labels and runs
`op` once per segment as a batch, so its cost is that pass plus `op`'s own. One event at a
time it compares the label with the current one and calls `op` on the row. Both give the
same results and leave the same state.
