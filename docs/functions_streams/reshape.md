---
name: reshape
title: reshape
kind: function
short: Name the axes of data with a layout string (T, A, F) and reshape it to match.
tags:
- reshape
- layout
- axes
- assets
- fields
- broadcast
see_also:
- from_pandas
- CombineLatest
topics:
- streams
---

# `reshape`

Free helper: names the axes of data with a layout string and reshapes it to
match. A **layout** spells out the axes of the input in order with the letters
`T` (time), `A` (asset), and `F` (field): `"TA"` reads a `(T, K)` array as `K`
assets, and `"TF"` reads it as one series with `K` fields. Without `->` the
result is the canonical `(T, *A, F)` shape that every screamer function reads;
an optional `->` names the wanted result instead.

<!-- HELP_END -->

## Signature

`reshape(x, layout)`

- **`x`** - the data: a NumPy array, a `(values, index)` tuple, a pandas
  Series or DataFrame, a list of events, or an iterator of events.
- **`layout`** - a string such as `"TA"` or `"TAF->TA"`. The left side names
  every axis of the input, so its length equals the input's rank. It holds
  exactly one `T`, at most one `F`, and any number of `A` (for example
  `"TAAF"` for assets by venue). A letter missing on the left is inserted with
  size 1; a letter missing on the right is dropped and must have size 1;
  shared letters are reordered.

Returns the same kind of container as `x`. A layout that does not fit the
input raises `ValueError` naming the layout and the input shape.

## Examples

### Five layouts

```{eval-rst}
.. exec_code::

    import numpy as np
    from screamer import reshape

    T, K = 100, 3
    print(reshape(np.zeros((T, K)), "TA").shape)         # canonical (T, *A, F)
    print(reshape(np.zeros((T, K, 1)), "TAF->TA").shape) # drop the size-1 field axis
    print(reshape(np.zeros((T, 4)), "TF->TAF").shape)    # insert a size-1 asset axis
    print(reshape(np.zeros((K, T)), "AT->TA").shape)     # time first
    print(reshape(np.zeros((T, 4, K)), "TFA").shape)     # fields after assets
```

### Containers

`reshape` accepts every container screamer accepts and returns the same kind:

| Input | Result |
|---|---|
| NumPy array | a view; a copy only when axes are reordered |
| `(values, index)` tuple | `(reshaped values, index)` |
| pandas Series or DataFrame | a DataFrame; flat columns with `"TA"` become two-level columns `(column, "value")` |
| list of events, iterator of events | the same kind, each event reshaped; `T` refers to the sequence |

`from_pandas` reads a DataFrame with two-level `(asset, field)` columns as
`(T, A, F)` values, so the DataFrame form round-trips:

```{eval-rst}
.. exec_code::

    import numpy as np
    import pandas as pd
    from screamer import reshape, from_pandas

    df = pd.DataFrame(np.arange(6.0).reshape(3, 2), columns=["BTC", "ETH"])
    wide = reshape(df, "TA")
    print(wide.columns.tolist())
    values, index = from_pandas(wide)
    print(values.shape)
```

### JavaScript

The JavaScript package exports the same function for an `NdArray` or a 1-D
array. It returns an `NdArray` that shares the input buffer, or a copy when
axes are reordered:

```js
import { ready, reshape } from "@screamer-labs/screamer";

await ready();
const x = { data: new Float64Array(6), shape: [3, 2] };
reshape(x, "TA").shape;   // [3, 2, 1]
```

The JavaScript `combineLatest` does not take a `layout` option yet.
