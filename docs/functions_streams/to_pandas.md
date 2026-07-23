---
name: to_pandas
title: to_pandas
kind: function
short: Convert a (values, index) tuple to a pandas Series or DataFrame.
topics:
- streams
---

# `to_pandas`

Free helper: converts the screamer `(values, index)` tuple convention to a
pandas Series (1-D) or DataFrame (2-D). Pass `columns` to label a 2-D result.
A positional stream (`index=None`) gets pandas' default RangeIndex.

<!-- HELP_END -->

## Signature

`to_pandas(values, index=None, columns=None)`

- **`values`** - a 1-D or 2-D NumPy array.
- **`index`** - a 1-D array or `None` (positional -> RangeIndex).
- **`columns`** - column names for a 2-D values array, or `None`.

Returns a `pandas.Series` (1-D) or `pandas.DataFrame` (2-D).

## Examples

### Usage example

```{eval-rst}
.. exec_code::

    import numpy as np
    import pandas as pd
    from screamer import from_pandas, to_pandas

    ser = pd.Series([1.0, 2.0, 3.0], index=[100, 200, 300])
    values, index = from_pandas(ser)
    print(to_pandas(values, index))
```
