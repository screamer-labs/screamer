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
