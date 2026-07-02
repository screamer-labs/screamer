import numpy as np
from screamer import streams, RollingCorr


def _ref_combine_latest(series, when_all):
    """Reference as-of join: merge by (key, source), carry last value per source."""
    events = []
    for i, (k, v) in enumerate(series):
        for kk, vv in zip(np.asarray(k), np.asarray(v)):
            events.append((kk, i, float(vv)))
    events.sort(key=lambda e: (e[0], e[1]))          # stable: key then source
    n = len(series)
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
    return keys, rows


def test_combine_latest_when_all_default():
    a_k = np.array([1, 3, 5], dtype=np.int64)
    a_v = np.array([10.0, 30.0, 50.0])
    b_k = np.array([2, 4], dtype=np.int64)
    b_v = np.array([20.0, 40.0])
    got_k, got_a = streams.combine_latest((a_k, a_v), (b_k, b_v))   # default when_all
    exp_k, exp_a = _ref_combine_latest([(a_k, a_v), (b_k, b_v)], when_all=True)
    np.testing.assert_array_equal(got_k, exp_k)
    np.testing.assert_array_equal(got_a, exp_a)


def test_combine_latest_on_any_warmup_is_nan():
    a_k = np.array([1, 3, 5], dtype=np.int64)
    a_v = np.array([10.0, 30.0, 50.0])
    b_k = np.array([2, 4], dtype=np.int64)
    b_v = np.array([20.0, 40.0])
    got_k, got_a = streams.combine_latest((a_k, a_v), (b_k, b_v), emit="on_any")
    exp_k, exp_a = _ref_combine_latest([(a_k, a_v), (b_k, b_v)], when_all=False)
    np.testing.assert_array_equal(got_k, exp_k)
    np.testing.assert_array_equal(got_a, exp_a)         # first row has NaN for b


def test_combine_latest_float_keys_three_series():
    rng = np.random.default_rng(3)
    series = []
    for _ in range(3):
        k = np.sort(rng.uniform(0, 100, size=40))
        v = rng.standard_normal(40)
        series.append((k, v))
    got_k, got_a = streams.combine_latest(*series)
    exp_k, exp_a = _ref_combine_latest(series, when_all=True)
    np.testing.assert_array_equal(got_k, exp_k)
    np.testing.assert_array_equal(got_a, exp_a)


def test_combine_latest_iter_matches_batch_identity():
    rng = np.random.default_rng(9)
    series = []
    for _ in range(3):
        k = np.sort(rng.integers(0, 500, size=120)).astype(np.int64)
        v = rng.standard_normal(120)
        series.append((k, v))

    bk, ba = streams.combine_latest(*series)                       # batch
    events = list(streams.combine_latest_iter(*series))            # streaming pull

    got_k = np.array([e[0] for e in events], dtype=np.int64)
    got_a = np.array([list(e[1]) for e in events], dtype=np.float64).reshape(len(events), 3)
    np.testing.assert_array_equal(got_k, bk)
    np.testing.assert_array_equal(got_a, ba)


def test_combine_latest_iter_on_any_identity():
    a_k = np.array([1, 4], dtype=np.int64); a_v = np.array([1.0, 4.0])
    b_k = np.array([2, 3], dtype=np.int64); b_v = np.array([2.0, 3.0])
    bk, ba = streams.combine_latest((a_k, a_v), (b_k, b_v), emit="on_any")
    events = list(streams.combine_latest_iter((a_k, a_v), (b_k, b_v), emit="on_any"))
    got_k = np.array([e[0] for e in events], dtype=np.int64)
    got_a = np.array([list(e[1]) for e in events], dtype=np.float64).reshape(len(events), 2)
    np.testing.assert_array_equal(got_k, bk)
    np.testing.assert_array_equal(got_a, ba)      # NaN warmup identical across modes


def test_rollingcorr_over_combine_latest():
    # Two async series -> align -> feed existing 2-input functor unchanged.
    rng = np.random.default_rng(21)
    a_k = np.sort(rng.integers(0, 2000, size=300)).astype(np.int64)
    a_v = rng.standard_normal(300)
    b_k = np.sort(rng.integers(0, 2000, size=250)).astype(np.int64)
    b_v = rng.standard_normal(250)

    keys, aligned = streams.combine_latest((a_k, a_v), (b_k, b_v))   # when_all
    assert aligned.shape[1] == 2
    # The idiom: existing functor consumes the aligned columns, untouched.
    corr = RollingCorr(20)(aligned[:, 0], aligned[:, 1])
    # Equivalent to calling the functor on the two aligned columns directly.
    exp = RollingCorr(20)(np.ascontiguousarray(aligned[:, 0]),
                          np.ascontiguousarray(aligned[:, 1]))
    np.testing.assert_array_equal(corr, exp)
    assert corr.shape[0] == keys.shape[0]


def test_combine_latest_func_reducer_spread():
    a_k = np.array([1, 3, 5], dtype=np.int64); a_v = np.array([10.0, 30.0, 50.0])
    b_k = np.array([2, 4], dtype=np.int64);     b_v = np.array([20.0, 40.0])
    keys, spread = streams.combine_latest((a_k, a_v), (b_k, b_v),
                                          func=lambda a, b: a - b)
    _, aligned = streams.combine_latest((a_k, a_v), (b_k, b_v))
    np.testing.assert_array_equal(spread, aligned[:, 0] - aligned[:, 1])
