import numpy as np
from screamer import RollingCorr
from screamer.streams import combine_latest


def _ref_combine_latest(streams, when_all):
    """Independent as-of join reference, coalesced to one row per distinct index.

    streams: list of (index_array, value_array). Returns (values, index) matching
    the new combine_latest contract (one row per distinct index).
    """
    events = []
    for i, (k, v) in enumerate(streams):
        for kk, vv in zip(np.asarray(k), np.asarray(v)):
            events.append((kk, i, float(vv)))
    events.sort(key=lambda e: (e[0], e[1]))          # index then source
    n = len(streams)
    latest = [np.nan] * n
    seen = [False] * n
    out_k, out_rows = [], []
    for kk, src, vv in events:
        latest[src] = vv
        seen[src] = True
        if when_all and not all(seen):
            continue
        out_k.append(kk)
        out_rows.append(list(latest))
    keys = np.array(out_k)
    rows = np.array(out_rows, dtype=np.float64).reshape(len(out_k), n)
    if len(keys):                                    # coalesce: last row per index
        keep = np.empty(len(keys), dtype=bool)
        keep[:-1] = keys[:-1] != keys[1:]
        keep[-1] = True
        keys, rows = keys[keep], rows[keep]
    return rows, keys


def test_combine_latest_when_all_default():
    ta = np.array([1, 3, 5], dtype=np.int64); a = np.array([10.0, 30.0, 50.0])
    tb = np.array([2, 4], dtype=np.int64);     b = np.array([20.0, 40.0])
    got_v, got_i = combine_latest(a, b, index=[ta, tb])
    exp_v, exp_i = _ref_combine_latest([(ta, a), (tb, b)], when_all=True)
    np.testing.assert_array_equal(got_i, exp_i)
    np.testing.assert_array_equal(got_v, exp_v)


def test_combine_latest_on_any_warmup_is_nan():
    ta = np.array([1, 3, 5], dtype=np.int64); a = np.array([10.0, 30.0, 50.0])
    tb = np.array([2, 4], dtype=np.int64);     b = np.array([20.0, 40.0])
    got_v, got_i = combine_latest(a, b, index=[ta, tb], emit="on_any")
    exp_v, exp_i = _ref_combine_latest([(ta, a), (tb, b)], when_all=False)
    np.testing.assert_array_equal(got_i, exp_i)
    np.testing.assert_array_equal(got_v, exp_v)       # first row has NaN for b


def test_combine_latest_float_index_three_streams():
    rng = np.random.default_rng(3)
    idx, vals = [], []
    for _ in range(3):
        idx.append(np.sort(rng.uniform(0, 100, size=40)))
        vals.append(rng.standard_normal(40))
    got_v, got_i = combine_latest(*vals, index=idx)
    exp_v, exp_i = _ref_combine_latest(list(zip(idx, vals)), when_all=True)
    np.testing.assert_array_equal(got_i, exp_i)
    np.testing.assert_array_equal(got_v, exp_v)


def test_combine_latest_lazy_matches_batch_identity():
    """Lazy indexed generators must give the same result as batch combine_latest."""
    rng = np.random.default_rng(9)
    idx, vals = [], []
    for _ in range(3):
        idx.append(np.sort(rng.integers(0, 500, size=120)).astype(np.int64))
        vals.append(rng.standard_normal(120))
    bv, bi = combine_latest(*vals, index=idx)                      # batch (coalesced)
    gens = [((float(v), int(k)) for v, k in zip(vals[i], idx[i])) for i in range(3)]
    events = list(combine_latest(*gens))                           # lazy indexed
    got_i = np.array([e[1] for e in events], dtype=np.int64)
    got_v = np.array([list(e[0]) for e in events], dtype=np.float64).reshape(len(events), 3)
    np.testing.assert_array_equal(got_i, bi)
    np.testing.assert_array_equal(got_v, bv)


def test_combine_latest_lazy_positional():
    """Positional lazy generators yield (row, None) per position, matching batch."""
    a = np.array([10.0, 20.0, 30.0]); b = np.array([1.0, 2.0, 3.0])
    events = list(combine_latest((x for x in a), (x for x in b)))
    assert all(idx is None for _, idx in events)
    got = np.array([list(row) for row, _ in events])
    bv, bi = combine_latest(a, b)
    assert bi is None
    np.testing.assert_array_equal(got, bv)


def test_combine_latest_func_reducer_spread():
    ta = np.array([1, 3, 5], dtype=np.int64); a = np.array([10.0, 30.0, 50.0])
    tb = np.array([2, 4], dtype=np.int64);     b = np.array([20.0, 40.0])
    spread, _ = combine_latest(a, b, index=[ta, tb], func=lambda x, y: x - y)
    aligned, _ = combine_latest(a, b, index=[ta, tb])
    np.testing.assert_array_equal(spread, aligned[:, 0] - aligned[:, 1])


def test_rollingcorr_over_combine_latest():
    # align two async streams, feed the aligned columns to a 2-input functor.
    rng = np.random.default_rng(21)
    ta = np.sort(rng.integers(0, 2000, size=300)).astype(np.int64); a = rng.standard_normal(300)
    tb = np.sort(rng.integers(0, 2000, size=250)).astype(np.int64); b = rng.standard_normal(250)
    aligned, idx = combine_latest(a, b, index=[ta, tb])
    ref_v, ref_i = _ref_combine_latest([(ta, a), (tb, b)], when_all=True)
    np.testing.assert_array_equal(idx, ref_i)
    np.testing.assert_array_equal(aligned, ref_v)
    assert aligned.shape[1] == 2
    got = RollingCorr(20)(np.ascontiguousarray(aligned[:, 0]),
                          np.ascontiguousarray(aligned[:, 1]))
    assert got.shape[0] == idx.shape[0]
