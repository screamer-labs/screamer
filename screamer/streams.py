"""Internal streaming graph hooks (not public API yet).

Builds and runs C++ node graphs. dtype detection here chooses the int64 or
float64 key instantiation; the per-event work is all C++.
"""
import numpy as np

from . import screamer_bindings as _b


def _run_chain(functors, values, keys=None, return_keys=False):
    """Run source -> functors[0] -> ... -> collector in batch.

    Returns values array by default. Pass return_keys=True to get
    (out_keys, out_values) instead.

    keys=None uses the row number (int64) as the ordering key.
    """
    values = np.ascontiguousarray(values, dtype=np.float64)
    n = values.shape[0]
    functors = list(functors)

    if keys is None:
        keys = np.arange(n, dtype=np.int64)
        out_k, out_v = _b._run_chain_i64(functors, keys, values)
    else:
        keys = np.asarray(keys)
        if np.issubdtype(keys.dtype, np.floating):
            keys = np.ascontiguousarray(keys, dtype=np.float64)
            out_k, out_v = _b._run_chain_f64(functors, keys, values)
        else:
            if keys.dtype.kind == "M":                  # datetime64 -> int64 view
                keys = keys.view("int64")
            keys = np.ascontiguousarray(keys, dtype=np.int64)
            out_k, out_v = _b._run_chain_i64(functors, keys, values)

    if return_keys:
        return out_k, out_v
    return out_v


def _key_dtype_kind(keys):
    """Return ('f64' | 'i64', normalized_keys) for a single key array."""
    keys = np.asarray(keys)
    if np.issubdtype(keys.dtype, np.floating):
        return "f64", np.ascontiguousarray(keys, dtype=np.float64)
    if keys.dtype.kind == "M":
        keys = keys.view("int64")
    return "i64", np.ascontiguousarray(keys, dtype=np.int64)


def merge(*series):
    """Merge N (keys, values) series into one key-sorted (keys, values, sources).

    Each series must be individually sorted by key. `sources[i]` is the index
    of the series that emitted event i. Ties break by series order.
    """
    if not series:
        raise ValueError("merge: needs at least one series")
    kinds = set()
    norm_keys, norm_vals = [], []
    for keys, values in series:
        kind, k = _key_dtype_kind(keys)
        kinds.add(kind)
        norm_keys.append(k)
        norm_vals.append(np.ascontiguousarray(values, dtype=np.float64))
    if len(kinds) != 1:
        raise TypeError("merge: all series must share one key type (all int/datetime or all float)")
    kind = kinds.pop()
    fn = _b._merge_f64 if kind == "f64" else _b._merge_i64
    return fn(norm_keys, norm_vals)


def _make_merge_puller(series):
    kinds = set()
    norm_keys, norm_vals = [], []
    for keys, values in series:
        kind, k = _key_dtype_kind(keys)
        kinds.add(kind)
        norm_keys.append(k)
        norm_vals.append(np.ascontiguousarray(values, dtype=np.float64))
    if len(kinds) != 1:
        raise TypeError("merge: all series must share one key type")
    kind = kinds.pop()
    cls = _b._MergePuller_f64 if kind == "f64" else _b._MergePuller_i64
    return cls(norm_keys, norm_vals)


def merge_iter(*series):
    """Yield (key, value, source) events in key order, pulled one at a time."""
    puller = _make_merge_puller(series)
    while True:
        event = puller.next()
        if event is None:
            return
        yield event


def _merge_events(*series):
    """Return a list of (key, value, source) tuples from merge_iter (test helper)."""
    return list(merge_iter(*series))
