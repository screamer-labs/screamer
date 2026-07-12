---
name: stream-tuple-convention
title: Stream tuple convention
kind: guide
short: The (values, index) tuple at every operator boundary.
topics:
- streams
---

# Stream tuple convention

Every screamer stream operator accepts and returns plain `(values, index)`
2-tuples at the batch boundary.

- **`values`** - a 1-D `(T,)` or 2-D `(T, N)` NumPy array.
- **`index`** - a 1-D array of length `T`, or `None` for positional streams
  (row number as ordering coordinate).

<!-- HELP_END -->

## The tuple forms

A stream value at an operator boundary is one of:

| Form | Meaning |
|---|---|
| bare ndarray | positional (index is row number; not stored) |
| `(values, index)` 2-tuple | indexed (this is the standard return form) |
| `Node` | graph node (builds a Pipeline) |
| lazy iterator | lazy streaming (unchanged) |

All batch operators return `(values, index)`. For positional streams `index`
is `None`. Multi-column results (e.g. ohlc) return a 2-D values array; columns
are positional in documented order.

## OHLC column order

| agg | col 0 | col 1 | col 2 | col 3 | col 4 | col 5 |
|---|---|---|---|---|---|---|
| `ohlc` | open | high | low | close | - | - |
| `ohlcv` | open | high | low | close | volume | - |
| `ohlcv2` | open | high | low | close | buy_vol | sell_vol |

## pandas helpers

`to_pandas(values, index=None, columns=None)` and `from_pandas(obj)` convert
between the tuple convention and pandas objects. Both are importable from the top level:
`from screamer import to_pandas, from_pandas`.

## Example

```{eval-rst}
.. exec_code::

    import numpy as np
    from screamer import CombineLatest, to_pandas

    a = np.array([10.0, 20.0, 40.0])
    b = np.array([1.0, 3.0, 4.0])

    # a stream operator returns a (values, index) pair
    values, index = CombineLatest()(a, b, index=[np.array([1, 2, 4]), np.array([1, 3, 4])])
    print(values)     # (T, 2): one column per input
    print(index)      # the shared index

    # to_pandas turns the pair into a labelled DataFrame
    print(to_pandas(values, index, columns=["a", "b"]))
```

## See also

- [to_pandas](to_pandas.md) - convert tuple to Series/DataFrame.
- [from_pandas](from_pandas.md) - convert Series/DataFrame to tuple.
- [Streams, values, and alignment](../multistream.md) - the conceptual model.
