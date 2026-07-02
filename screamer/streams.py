"""Internal streaming graph hooks (not public API yet).

Builds and runs C++ node graphs. dtype detection here chooses the int64 or
float64 key instantiation; the per-event work is all C++.
"""
import asyncio

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
        kind = "i64"
    else:
        kind, keys = _key_dtype_kind(keys)

    fn = _b._run_chain_f64 if kind == "f64" else _b._run_chain_i64
    result = fn(functors, keys, values, return_keys)
    return result  # tuple (keys, values) if return_keys else values array


def _key_dtype_kind(keys):
    """Return ('f64' | 'i64', normalized_keys) for a single key array."""
    keys = np.asarray(keys)
    if np.issubdtype(keys.dtype, np.floating):
        return "f64", np.ascontiguousarray(keys, dtype=np.float64)
    if keys.dtype.kind == "M":
        keys = keys.view("int64")
    return "i64", np.ascontiguousarray(keys, dtype=np.int64)


def _normalize_series(series, who):
    """Normalize and validate a sequence of (keys, values) series.

    Returns (kind, norm_keys, norm_vals) where kind is 'i64' or 'f64'.
    Raises ValueError if series is empty, TypeError if key types differ.
    """
    if not series:
        raise ValueError(f"{who}: needs at least one series")
    kinds = set()
    norm_keys, norm_vals = [], []
    for keys, values in series:
        kind, k = _key_dtype_kind(keys)
        kinds.add(kind)
        norm_keys.append(k)
        norm_vals.append(np.ascontiguousarray(values, dtype=np.float64))
    if len(kinds) != 1:
        raise TypeError(f"{who}: all series must share one key type (all int/datetime or all float)")
    return kinds.pop(), norm_keys, norm_vals


def merge(*series):
    """Merge N (keys, values) series into one key-sorted (keys, values, sources).

    Each series must be individually sorted by key. `sources[i]` is the index
    of the series that emitted event i. Ties break by series order.
    """
    kind, norm_keys, norm_vals = _normalize_series(series, "merge")
    fn = _b._merge_f64 if kind == "f64" else _b._merge_i64
    return fn(norm_keys, norm_vals)


def _make_merge_puller(series):
    kind, norm_keys, norm_vals = _normalize_series(series, "merge_iter")
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


def combine_latest(*series, emit="when_all", func=None):
    """As-of latest-value join of N (keys, values) series.

    Emits an aligned row whenever any input advances, carrying each input's most
    recent value (forward-fill). Returns (keys, aligned) where aligned is (M, N);
    aligned[:, j] is series j's latest value at each emitted key. emit="when_all"
    (default) suppresses output until every input is warm; emit="on_any" emits
    from the first event with NaN for not-yet-seen inputs. If `func` is given it is
    applied per row (func(*row)) and (keys, reduced) is returned instead.
    """
    if emit not in ("when_all", "on_any"):
        raise ValueError('combine_latest: emit must be "when_all" or "on_any"')
    kind, norm_keys, norm_vals = _normalize_series(series, "combine_latest")
    fn = _b._combine_latest_f64 if kind == "f64" else _b._combine_latest_i64
    keys, aligned = fn(norm_keys, norm_vals, emit == "when_all")
    if func is None:
        return keys, aligned
    reduced = np.array([func(*row) for row in aligned], dtype=np.float64)
    return keys, reduced


async def pace(*series, speed=1.0, sleep=None):
    """Replay merged series as an async event stream paced by key-deltas.

    Yields (key, value, source) in key order. Between consecutive events it
    awaits `sleep(key_delta / speed)` so wall-clock spacing tracks the key
    spacing. speed=inf disables pacing (backtest at max speed). Pacing never
    changes values or order. `sleep` is injectable for testing; defaults to
    asyncio.sleep. Requires a metric (subtractable) key.
    """
    if speed <= 0:
        raise ValueError("pace: speed must be positive (or float('inf') for no pacing)")
    if sleep is None:
        sleep = asyncio.sleep
    infinite = speed == float("inf")
    prev_key = None
    for key, value, source in merge_iter(*series):
        if not infinite and prev_key is not None:
            delta = key - prev_key
            wait = delta / speed
            if wait > 0:
                await sleep(wait)
        prev_key = key
        yield key, value, source
