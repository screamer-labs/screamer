---
name: from_pandas
title: from_pandas
kind: function
short: Convert a pandas Series or DataFrame to a (values, index) tuple.
topics:
- streams
---

# `from_pandas`

Free helper: converts a pandas Series or DataFrame to the screamer
`(values, index)` tuple convention. The pandas data becomes the values array
and the pandas index becomes the index array.

<!-- HELP_END -->

## Signature

`from_pandas(obj)`

- **`obj`** - a `pandas.Series` or `pandas.DataFrame`.

Returns `(values, index)` where both are NumPy arrays.

## Example

```python
import pandas as pd
from screamer import from_pandas

ser = pd.Series([1.0, 2.0, 3.0], index=[100, 200, 300])
values, index = from_pandas(ser)
# values -> array([1., 2., 3.]), index -> array([100, 200, 300])
```
