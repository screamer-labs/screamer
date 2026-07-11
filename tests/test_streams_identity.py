import numpy as np
import pytest

from screamer import streams


def _make_series(n_series, size, dtype, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_series):
        if dtype == np.int64:
            k = np.sort(rng.integers(0, size * 4, size=size)).astype(np.int64)
        else:
            k = np.sort(rng.uniform(0, size * 4.0, size=size)).astype(np.float64)
        v = rng.standard_normal(size)
        out.append((k, v))
    return out


CONFIGS = [
    (n_series, dtype)
    for n_series in (2, 3, 5)
    for dtype in (np.int64, np.float64)
]


@pytest.mark.parametrize("n_series,dtype", CONFIGS)
def test_merge_batch_equals_stream_indexed(n_series, dtype):
    series = _make_series(n_series, 80, dtype, seed=100 + n_series)
    vals = [v for _, v in series]
    idxs = [k for k, _ in series]

    bv, bs, bi = streams.Merge()(*vals, index=idxs)
    # Use merge(*generators) with indexed (value, index) tuples; preserve dtype
    gens = [((float(v), k) for v, k in zip(vals[i], idxs[i]))
            for i in range(n_series)]
    events = list(streams.Merge()(*gens))

    ev_v = np.array([e[0] for e in events])
    ev_i = np.array([e[1] for e in events], dtype=bi.dtype)
    ev_s = np.array([e[2] for e in events], dtype=np.uint32)
    np.testing.assert_array_equal(ev_v, bv)
    np.testing.assert_array_equal(ev_i, bi)
    np.testing.assert_array_equal(ev_s, bs)


@pytest.mark.parametrize("n_series", [2, 3, 5])
def test_merge_batch_equals_stream_positional(n_series):
    rng = np.random.default_rng(seed=150 + n_series)
    # unequal lengths are allowed for positional merge
    vals = [rng.standard_normal(rng.integers(30, 80)) for _ in range(n_series)]

    bv, bs, bi = streams.Merge()(*vals)
    assert bi is None
    # Use merge(*generators) with bare-value (positional) sources
    gens = [(float(v) for v in val_arr) for val_arr in vals]
    events = list(streams.Merge()(*gens))

    np.testing.assert_array_equal(np.array([e[0] for e in events]), bv)
    assert all(e[1] is None for e in events)
    np.testing.assert_array_equal(np.array([e[2] for e in events], dtype=np.uint32), bs)


@pytest.mark.parametrize("n_series,dtype", CONFIGS)
@pytest.mark.parametrize("emit", ["when_all", "on_any"])
def test_combine_latest_batch_equals_stream(n_series, dtype, emit):
    series = _make_series(n_series, 80, dtype, seed=200 + n_series)
    vals = [v for _, v in series]
    # Convert indices to int64: the lazy Dag path (via _LazyDag._pull) converts
    # index to int, so we use int64 for both batch oracle and generators.
    idxs_int = [k.astype(np.int64) for k, _ in series]
    # Batch oracle with int64 indices
    stream_list = [(v, k) for v, k in zip(vals, idxs_int)]
    result = streams.CombineLatest(emit=emit)(*stream_list)
    bk = result[1]
    ba = result[0]
    # Lazy path: generators of (float(v), int(k)) -> indexed lazy sources
    gens = [((float(v), int(k)) for v, k in zip(vals[i], idxs_int[i]))
            for i in range(n_series)]
    events = list(streams.CombineLatest(emit=emit)(*gens))
    got_k = np.array([e[1] for e in events], dtype=bk.dtype)
    got_a = np.array([list(e[0]) for e in events],
                     dtype=np.float64).reshape(len(events), n_series)
    np.testing.assert_array_equal(got_k, bk)
    np.testing.assert_array_equal(got_a, ba)


def test_dropna_batch_equals_stream_on_combine_output():
    # combine_latest on_any produces NaN warmup rows; dropna must remove the
    # same rows whether applied to the batch array or the event stream.
    series = _make_series(3, 50, np.int64, seed=42)
    stream_list = [(v, k) for k, v in series]
    result = streams.CombineLatest(emit="on_any")(*stream_list)
    bk = result[1]
    ba = result[0]
    # dropna is values-first: Dropna()(values, index=...) -> (filtered_values, filtered_index)
    dv, dk = streams.Dropna(how="any")(ba, index=bk)

    # Lazy path: generators of (float(v), int(k)) (int64 indices from series)
    gens = [((float(v), int(k)) for v, k in zip(s[0], s[1]))
            for s in stream_list]
    events = list(streams.CombineLatest(emit="on_any")(*gens))
    # filter NaN rows inline
    kept = [(row, idx) for row, idx in events
            if not np.any(np.isnan(np.asarray(row, dtype=np.float64)))]
    dk_stream = np.array([idx for _, idx in kept], dtype=bk.dtype)
    dv_stream = np.array([np.asarray(row) for row, _ in kept],
                         dtype=np.float64).reshape(len(kept), -1)
    assert dv_stream.shape[1] == 3
    np.testing.assert_array_equal(dk_stream, dk)
    np.testing.assert_array_equal(dv_stream, dv)
