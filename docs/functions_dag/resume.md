---
name: resume
title: resume
kind: guide
short: The batch call that starts from a function's state and keeps the end state.
tags:
- resume
- warm up
- continue
- save
- state
- panel
see_also:
- load
- from_settings
- Pipeline
topics:
- save-restore
---

# `resume`

`f.resume(*inputs)` is a batch call that starts from the function's current
state and keeps the end state in the function. Every function has it. A
default call, `f(x)`, starts from a fresh state and leaves the function fresh;
`f.resume(x)` continues instead, so feeding `A` and then `B` with `resume`
gives the numbers of one call on `A` followed by `B`. It takes the inputs a
default call takes and returns the same shape.

On a fresh function, `resume` is the warm-up call: it runs the history, after
which the function can be saved with `save`, or fed one value at a time. After
`load`, it continues from the loaded state.

On a `(T, K)` array, `resume` keeps one state per column (a panel), saved as
one state per column. The function then takes only `(T, K)` arrays with the
same `K`: another `K`, or a call with a single event, raises `ValueError`
until `reset()` or a default batch call drops the panel. A row operator (the
cross-section operators and the portfolio engines) and `PortfolioReport` are
one object for `K` assets already: their `resume` continues that state, and
rows of another width raise `ValueError`. In JavaScript the method is
`f.resume(...inputs)`.

<!-- HELP_END -->

## Examples

```{eval-rst}
.. exec_code::

    import numpy as np
    import screamer

    x = np.arange(10.0)
    f = screamer.RollingMean(3)
    f.resume(x[:6])            # warm up on history
    print(f.resume(x[6:]))     # continues: the same as RollingMean(3)(x)[6:]
    print(f(10.0))             # then one value at a time
```
