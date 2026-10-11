---
name: from_pandas
title: from_pandas
kind: function
short: Convert a pandas Series or DataFrame to a (values, index) tuple.
tags:
- pandas
- dataframe
- series
- convert
see_also:
- to_pandas
- Resample
- CombineLatest
topics:
- streams
---

# `from_pandas`

Free helper: converts a pandas Series or DataFrame to the screamer
`(values, index)` tuple convention. The pandas data becomes the values array
and the pandas index becomes the index array.

<!-- HELP_END -->

## Signature

`from_pandas(obj, layout=None)`

- **`obj`** - a `pandas.Series` or `pandas.DataFrame`.
- **`layout`** - optional layout string naming the axes of the values as read.
  The default `None` returns `obj.to_numpy()` with the frame's own dtype and
  columns, so two-level columns stay flat. A layout reads the values as
  float64 and reshapes them like `reshape`: `"TA"` for one column per asset,
  or `"TAF"` for a DataFrame whose two-level columns hold every
  `(asset, field)` pair, which gives `(T, A, F)` values.

Returns `(values, index)` where both are NumPy arrays.

## Examples

### Usage example

```{eval-rst}
.. exec_code::

    import pandas as pd
    from screamer import from_pandas

    ser = pd.Series([1.0, 2.0, 3.0], index=[100, 200, 300])
    values, index = from_pandas(ser)
    print("values:", values)
    print("index: ", index)
```
