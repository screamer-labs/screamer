---
name: select
title: select
kind: function
short: Pick column(s) from a wide (M, N) value stream.
topics:
- streams
covers:
- Select
---

# `select`

Pick one or more columns from a wide `(rows, columns)` value stream by position.
The index and the number of rows are unchanged; only the width changes. Usable
eagerly and inside a `Dag`.

Feeding a lazy iterator of `(value, index)` pairs returns a lazy iterator of column-selected events; feeding arrays or a `Stream` returns the batch result.

<!-- HELP_END -->

```{eval-rst}
.. autofunction:: screamer.streams.select
```

## Example

Keep column 1 of a two-column stream.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer.streams import select
   # --- hide: stop ---
   wide = np.array([[10.0, 11.0],
                    [20.0, 21.0],
                    [30.0, 31.0]])

   column, idx = select(wide, 1)
   print(column)
```

Pass a list of positions (for example `[0, 2]`) to keep several columns in a
chosen order.
