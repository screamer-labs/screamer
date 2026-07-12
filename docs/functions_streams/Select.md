---
name: Select
title: Select
kind: class
short: Pick column(s) from a wide (M, N) value stream.
topics:
- streams
covers:
- select
---

# `Select`

Pick one or more columns from a wide `(rows, columns)` value stream by position.
The index and the number of rows are unchanged; only the width changes. Usable
eagerly and inside a `Pipeline`.

Feeding a lazy iterator of `(value, index)` pairs returns a lazy iterator of column-selected events; feeding arrays or `(values, index)` tuples returns the batch result.

<!-- HELP_END -->

## Example

Keep column 1 of a two-column stream.

```{eval-rst}
.. exec_code::

   import numpy as np
   from screamer import Select

   wide = np.array([[10.0, 11.0],
                    [20.0, 21.0],
                    [30.0, 31.0]])

   column, idx = Select(1)(wide)
   print(column)
```

Pass a list of positions (for example `[0, 2]`) to keep several columns in a
chosen order.
