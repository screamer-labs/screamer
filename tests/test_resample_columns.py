"""Resample aggregates every column on its own (shape contract rule 4)."""
import numpy as np
import pytest

import screamer as s

rng = np.random.default_rng(2)
T, K = 60, 3
X = rng.normal(size=(T, K)).cumsum(axis=0)
idx = np.arange(T) * 10


def _per_column(agg, data=X):
    return [s.Resample(freq=50, agg=agg)((data[:, k], idx))[0] for k in range(data.shape[1])]


@pytest.mark.parametrize("agg", ["first", "last", "min", "max", "sum", "count", "mean"])
def test_single_value_aggs(agg):
    v, i = s.Resample(freq=50, agg=agg)((X, idx))
    assert v.shape == (len(i), K)
    for k, col in enumerate(_per_column(agg)):
        np.testing.assert_array_equal(v[:, k], col)


def test_ohlc_appends_a_field_axis():
    v, i = s.Resample(freq=50, agg="ohlc")((X, idx))
    assert v.shape == (len(i), K, 4)
    for k, col in enumerate(_per_column("ohlc")):
        np.testing.assert_array_equal(v[:, k], col)


def test_unit_field_consumed():
    v, _ = s.Resample(freq=50, agg="ohlc")((X[:, :, None], idx))
    assert v.shape[1:] == (K, 4)


def test_t_by_1_keeps_its_column():
    v, _ = s.Resample(count=5, agg="last")(X[:, :1])
    assert v.shape[1:] == (1,)


def test_count_mode_and_positional():
    v, _ = s.Resample(count=7, agg="max")(X)
    assert v.shape == (int(np.ceil(T / 7)), K)


def test_lazy_matches_batch():
    batch, bidx = s.Resample(freq=50, agg="ohlc")((X, idx))
    lazy = list(s.Resample(freq=50, agg="ohlc")(iter(zip(X, idx))))
    np.testing.assert_array_equal(np.array([v for v, _ in lazy]), batch)
    assert [i for _, i in lazy] == list(bidx)


def test_pipeline_wide_input():
    x = s.Input("x")
    p = s.Pipeline([x], [s.Resample(count=5, agg="last")(x)])
    v, _ = p(X)
    np.testing.assert_array_equal(v, X[4::5])


def test_functor_agg_takes_one_column():
    with pytest.raises(ValueError, match="one column"):
        s.Resample(count=5, agg=s.ExpandingSkew())(X)


def test_ohlc_bars_unchanged():
    bars = np.column_stack([X[:, 0]] * 4)
    v, _ = s.Resample(count=5, agg="ohlc_bars")(bars)
    assert v.shape[1] == 4


# E2: a clock resample of (T, *A) gives (T', *A), with a last axis of 4 for
# ohlc, as freq= and count= do (2.8.0 gave each column's open for ohlc).
# Every result is checked against an independent numpy reference.
clk_idx = np.arange(25, T * 10, 70)
clk = np.zeros(len(clk_idx))

_REDUCE = {
    "first": lambda a: a[0], "last": lambda a: a[-1], "min": lambda a: a.min(0),
    "max": lambda a: a.max(0), "sum": lambda a: a.sum(0), "mean": lambda a: a.mean(0),
    "ohlc": lambda a: np.stack([a[0], a.max(0), a.min(0), a[-1]], axis=-1),
    "ohlcv": lambda a: np.r_[a[0, 0], a[:, 0].max(), a[:, 0].min(), a[-1, 0], a[:, 1].sum()],
    "ohlc_bars": lambda a: np.r_[a[0, 0], a[:, 1].max(), a[:, 2].min(), a[-1, 3]],
    "ohlcv_bars": lambda a: np.r_[a[0, 0], a[:, 1].max(), a[:, 2].min(), a[-1, 3],
                                  a[:, 4:].sum(0)],
}


def _buckets(mode, index):
    """(label, rows) of each non-empty bar: clock ticks clk_idx, count=7, freq=50."""
    if mode == "clock":
        prev, out = -np.inf, []
        for k in clk_idx:
            rows = np.flatnonzero((index > prev) & (index <= k))
            if len(rows):
                out.append((k, rows))
            prev = k
        return out
    if mode == "count":
        return [(index[r], np.arange(r, min(r + 7, len(index)))) for r in range(0, len(index), 7)]
    keys = index // 50
    return [(b * 50, np.flatnonzero(keys == b)) for b in np.unique(keys)]


def _reference(data, mode, agg):
    bars = _buckets(mode, idx)
    return (np.array([_REDUCE[agg](data[rows]) for _, rows in bars]),
            np.array([label for label, _ in bars]))


def _run(data, mode, agg, lazy):
    cfg = {"count": dict(count=7), "freq": dict(freq=50)}
    if mode == "clock":
        if lazy:
            out = list(s.Resample(iter(zip(data, idx)), iter(zip(clk, clk_idx)),
                                  clock=True, agg=agg))
        else:
            return s.Resample((data, idx), (clk, clk_idx), clock=True, agg=agg)
    elif lazy:
        out = list(s.Resample(agg=agg, **cfg[mode])(iter(zip(data, idx))))
    else:
        return s.Resample(agg=agg, **cfg[mode])((data, idx))
    for v, _ in out:
        assert isinstance(v, np.ndarray) or agg in ("ohlcv", "ohlc_bars", "ohlcv_bars")
    return np.array([v for v, _ in out]), np.array([i for _, i in out])


@pytest.mark.parametrize("mode", ["clock", "count", "freq"])
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("agg", ["ohlc", "last", "mean", "min"])
@pytest.mark.parametrize("shape", [(K,), (2, 3), (3, 1)])
def test_column_aggs_keep_the_asset_axes(mode, lazy, agg, shape):
    data = rng.normal(size=(T,) + shape)
    v, i = _run(data, mode, agg, lazy)
    ref, ref_idx = _reference(data.reshape((T,) + (shape[:1] if shape[-1:] == (1,) else shape)),
                              mode, agg)
    np.testing.assert_allclose(v, ref, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(i, ref_idx)


@pytest.mark.parametrize("mode", ["clock", "count", "freq"])
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("agg,width", [("ohlcv", 2), ("ohlc_bars", 4), ("ohlcv_bars", 6)])
def test_bar_aggs_read_the_columns_as_one_bar(mode, lazy, agg, width):
    data = np.abs(rng.normal(size=(T, width))) + 1
    v, i = _run(data, mode, agg, lazy)
    ref, ref_idx = _reference(data, mode, agg)
    np.testing.assert_allclose(v, ref, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(i, ref_idx)


def test_one_column_lazy_rows_are_numbers_or_tuples():
    # As any lazy operator on scalar events: one value a float, several a tuple.
    out = list(s.Resample(iter(zip(X[:, 0], idx)), iter(zip(clk, clk_idx)),
                          clock=True, agg="ohlc"))
    assert isinstance(out[0][0], tuple) and len(out[0][0]) == 4
    out = list(s.Resample(count=7, agg="last")(iter(zip(X[:, 0], idx))))
    assert isinstance(out[0][0], float)


# E2: inside a Pipeline a wide ohlc is flat, (T', 4 * K), in column-major field
# order [o0, h0, l0, c0, o1, ...]: out.reshape(T', K, 4) is the eager result.
def _ohlc_pipeline(mode):
    x, c = s.Input("x"), s.Input("c")
    if mode == "clock":
        return s.Pipeline([x, c], [s.Resample(x, c, clock=True, agg="ohlc")])
    cfg = {"freq": dict(freq=50), "count": dict(count=7)}[mode]
    return s.Pipeline([x], [s.Resample(agg="ohlc", **cfg)(x)])


@pytest.mark.parametrize("mode", ["freq", "count", "clock"])
@pytest.mark.parametrize("run", ["batch", "lazy", "live"])
def test_pipeline_wide_ohlc_is_flat_column_major(mode, run):
    p = _ohlc_pipeline(mode)
    feeds = [(X, idx)] + ([(clk, clk_idx)] if mode == "clock" else [])
    if run == "batch":
        v, i = p(*feeds)
    elif run == "lazy":
        out = list(p(*(iter(zip(a, b)) for a, b in feeds)))
        v, i = np.array([e for e, _ in out]), np.array([j for _, j in out])
    else:
        live = p.live()
        events = sorted([(int(idx[t]), 0, t) for t in range(T)] +
                        ([(int(k), 1, 0) for k in clk_idx] if mode == "clock" else []))
        for k, port, t in events:
            if port == 0:
                live.push("x", k, list(X[t]))
            else:
                live.push("c", k, 0.0)
        live.flush()
        v, i = live.result()
    ref, ref_idx = _reference(X, mode, "ohlc")
    assert np.asarray(v).shape == (len(ref_idx), 4 * K)
    np.testing.assert_allclose(np.asarray(v), ref.reshape(len(ref), -1), rtol=0, atol=1e-12)
    np.testing.assert_array_equal(np.asarray(i), ref_idx)
    eager, _ = _run(X, mode, "ohlc", lazy=False)
    np.testing.assert_array_equal(np.asarray(v).reshape(-1, K, 4), eager)


def test_pipeline_lazy_events_keep_their_rules():
    # The lazy resample reads (T, *A) events; a Pipeline stream still takes
    # one value or one row per event. Strided and float32 rows read as before.
    x = s.Input("x")
    p = s.Pipeline([x], [s.Dropna()(x)])
    with pytest.raises(ValueError, match="expected a number"):
        list(p(iter([(np.zeros((2, 3)), 0)])))
    strided = np.asfortranarray(np.arange(12.0).reshape(4, 3))[0]
    rows = list(p(iter([(strided, 0), (np.arange(3, dtype=np.float32), 1)])))
    assert rows == [((0.0, 1.0, 2.0), 0), ((0.0, 1.0, 2.0), 1)]


# The lazy resample reads an event of any numeric kind or layout as a batch
# call reads the same rows (fix round 2, N1).
_D = rng.normal(size=(T, 2, 3))
_EVENT_KINDS = {
    "float32": lambda: _D.astype(np.float32),
    "int64": lambda: (_D * 10).astype(np.int64),
    "fortran": lambda: np.asfortranarray(_D),
    "strided": lambda: rng.normal(size=(T, 2, 6))[:, :, ::2],
    "big-endian": lambda: _D.astype(">f8"),
    "bool": lambda: _D > 0,
    "float16": lambda: _D.astype(np.float16),
}


def _lazy_rows(rows, mode, agg):
    if mode == "clock":
        out = s.Resample(iter(zip(rows, idx)), iter(zip(clk, clk_idx)), clock=True, agg=agg)
    else:
        cfg = {"count": dict(count=7), "freq": dict(freq=50)}[mode]
        out = s.Resample(agg=agg, **cfg)(iter(zip(rows, idx)))
    out = list(out)
    return np.array([v for v, _ in out]), np.array([i for _, i in out])


@pytest.mark.parametrize("mode", ["clock", "count", "freq"])
@pytest.mark.parametrize("kind", sorted(_EVENT_KINDS) + ["lists", "tuples", "list of rows"])
def test_lazy_reads_every_event_kind_as_batch(mode, kind):
    if kind in _EVENT_KINDS:
        data = _EVENT_KINDS[kind]()
        rows = list(data)
    else:
        data = _D
        rows = {"lists": [r.tolist() for r in _D],
                "tuples": [tuple(map(tuple, r)) for r in _D],
                "list of rows": [list(r) for r in _D]}[kind]
    v, i = _lazy_rows(rows, mode, "ohlc")
    batch, batch_idx = _run(data, mode, "ohlc", lazy=False)
    np.testing.assert_array_equal(v, batch)
    np.testing.assert_array_equal(i, batch_idx)


def test_lazy_event_shape_is_fixed_by_the_first_event():
    with pytest.raises(ValueError, match=r"shape \(3, 2\), but the first event had shape \(2, 3\)"):
        list(s.Resample(count=2, agg="last")(iter([(np.zeros((2, 3)), 0), (np.zeros((3, 2)), 1)])))
    with pytest.raises(ValueError, match=r"a number, but the first event had shape \(1,\)"):
        list(s.Resample(count=2, agg="last")(iter([(np.zeros(1), 0), (0.0, 1)])))
    with pytest.raises(ValueError, match="irregular shape"):
        list(s.Resample(count=1, agg="last")(iter([([[1.0, 2.0], [3.0]], 0)])))
    with pytest.raises(TypeError, match="is a str"):
        list(s.Resample(count=1, agg="last")(iter([([["a", 1.0], [1.0, 2.0]], 0)])))
    with pytest.raises(TypeError, match="complex128"):
        list(s.Resample(count=1, agg="last")(iter([(np.zeros((2, 3), complex), 0)])))
    (v, _), = s.Resample(count=1, agg="last")(iter([([[None, 1.0], [2.0, 3.0]], 0)]))
    np.testing.assert_array_equal(v, [[np.nan, 1.0], [2.0, 3.0]])
