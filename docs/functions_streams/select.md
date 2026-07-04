# `select`

Pick one or more columns from a wide `(rows, columns)` value stream by position.
The index and the number of rows are unchanged; only the width changes. Usable
eagerly and inside a `Dag`.

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
