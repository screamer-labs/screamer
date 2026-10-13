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
# ohlc, as freq= and count= do (2.8.0 gave a flat (T', 4 * A) for ohlc).
clk_idx = np.arange(25, T * 10, 70)
clk = np.zeros(len(clk_idx))


def test_eager_clock_ohlc_appends_a_field_axis():
    v, i = s.Resample((X, idx), (clk, clk_idx), clock=True, agg="ohlc")
    assert v.shape == (len(i), K, 4)
    for k in range(K):
        col, ci = s.Resample((X[:, k], idx), (clk, clk_idx), clock=True, agg="ohlc")
        np.testing.assert_array_equal(v[:, k], col)
        np.testing.assert_array_equal(i, ci)
    lazy = list(s.Resample(iter(zip(X, idx)), iter(zip(clk, clk_idx)),
                           clock=True, agg="ohlc"))
    np.testing.assert_array_equal(np.array([e for e, _ in lazy]), v)
    assert [j for _, j in lazy] == list(i)


@pytest.mark.parametrize("agg", ["last", "ohlc"])
def test_eager_clock_keeps_asset_axes(agg):
    data = rng.normal(size=(T, 2, 3))
    v, i = s.Resample((data, idx), (clk, clk_idx), clock=True, agg=agg)
    flat, _ = s.Resample((data.reshape(T, 6), idx), (clk, clk_idx), clock=True, agg=agg)
    assert v.shape == (len(i), 2, 3) + ((4,) if agg == "ohlc" else ())
    np.testing.assert_array_equal(v.reshape(len(i), -1), flat.reshape(len(i), -1))


@pytest.mark.parametrize("agg", ["ohlcv", "ohlc_bars", "ohlcv_bars"])
def test_eager_clock_bar_aggs_match_freq_shape(agg):
    width = {"ohlcv": 2, "ohlc_bars": 4, "ohlcv_bars": 6}[agg]
    data = np.abs(X[:, :1]) + np.arange(width)
    v, _ = s.Resample((data, idx), (clk, clk_idx), clock=True, agg=agg)
    w, _ = s.Resample(freq=50, agg=agg)((data, idx))
    assert v.shape[1:] == w.shape[1:]


# E2: inside a Pipeline a wide ohlc is flat, (T', 4 * K), in column-major field
# order [o0, h0, l0, c0, o1, ...]: out.reshape(T', K, 4) is the eager result.
def _ohlc_pipeline(mode):
    x, c = s.Input("x"), s.Input("c")
    if mode == "clock":
        return s.Pipeline([x, c], [s.Resample(x, c, clock=True, agg="ohlc")])
    cfg = {"every": dict(freq=50), "count": dict(count=7)}[mode]
    return s.Pipeline([x], [s.Resample(agg="ohlc", **cfg)(x)])


def _eager_ohlc(mode):
    if mode == "clock":
        return s.Resample((X, idx), (clk, clk_idx), clock=True, agg="ohlc")
    cfg = {"every": dict(freq=50), "count": dict(count=7)}[mode]
    return s.Resample(agg="ohlc", **cfg)((X, idx))


@pytest.mark.parametrize("mode", ["every", "count", "clock"])
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
    eager, eager_idx = _eager_ohlc(mode)
    assert eager.shape == (len(eager_idx), K, 4)
    assert np.asarray(v).shape == (len(eager_idx), 4 * K)
    np.testing.assert_array_equal(np.asarray(v).reshape(-1, K, 4), eager)
    np.testing.assert_array_equal(np.asarray(i), eager_idx)
