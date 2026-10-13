import inspect

import numpy as np
import pytest
from screamer.supervised import forecast_pairs


# ---------------------------------------------------------------------------
# Batch regression: output must be byte-identical to the original implementation
# ---------------------------------------------------------------------------

def test_forecast_pairs_count_pairs_features_with_future_target():
    # feature at row t pairs with the target `count` rows later
    X = np.arange(6.0)                       # 0,1,2,3,4,5
    y = np.arange(6.0) * 10                  # 0,10,20,30,40,50
    Xs, ys = forecast_pairs(X, y, count=2)
    # row t holds feature X[t-2] and target y[t]; first 2 rows warm up to NaN
    assert np.isnan(Xs[:2]).all()
    np.testing.assert_array_equal(Xs[2:], [0.0, 1.0, 2.0, 3.0])   # X[t-2]
    np.testing.assert_array_equal(ys, y)                          # y untouched


def test_forecast_pairs_count_dropna_returns_clean_pairs():
    X = np.arange(6.0)
    y = np.arange(6.0) * 10
    Xs, ys = forecast_pairs(X, y, count=2, dropna=True)
    assert not np.isnan(Xs).any()
    np.testing.assert_array_equal(Xs, [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(ys, [20.0, 30.0, 40.0, 50.0])


def test_forecast_pairs_count_matches_forward_return_reference():
    # forecast_pairs(X, RollingSum(h)(ret), count=h) reproduces the forward-return pairing
    from screamer import RollingSum
    rng = np.random.default_rng(0)
    n, h = 50, 5
    ret = rng.standard_normal(n) * 1e-3
    X = rng.standard_normal(n)
    y = np.asarray(RollingSum(h)(ret))
    Xs, ys = forecast_pairs(X, y, count=h, dropna=True)
    # reference: X[s] paired with sum(ret[s+1..s+h]) for valid s
    fwd = np.array([ret[s + 1:s + 1 + h].sum() for s in range(n - h)])
    np.testing.assert_allclose(Xs, X[:n - h])
    np.testing.assert_allclose(ys, fwd)


def test_forecast_pairs_requires_exactly_one_of_count_duration():
    X = np.arange(5.0); y = np.arange(5.0)
    with pytest.raises(ValueError):
        forecast_pairs(X, y)                       # neither
    with pytest.raises(ValueError):
        forecast_pairs(X, y, count=1, duration=1)  # both


def test_forecast_pairs_count_2d_features_per_column():
    X = np.column_stack([np.arange(6.0), np.arange(6.0) * 2])
    y = np.arange(6.0)
    Xs, ys = forecast_pairs(X, y, count=2, dropna=True)
    np.testing.assert_array_equal(Xs, np.column_stack([[0, 1, 2, 3], [0, 2, 4, 6]]))


def test_forecast_pairs_duration_matches_count_on_regular_grid():
    # on a regular grid, duration = count * step gives the same pairs as count
    rng = np.random.default_rng(1)
    n, step, h = 40, 100, 3
    idx = np.arange(n, dtype=np.int64) * step
    Xv = rng.standard_normal(n)
    yv = rng.standard_normal(n)
    Xc, yc = forecast_pairs(Xv, yv, count=h, dropna=True)
    Xd, yd = forecast_pairs((Xv, idx), (yv, idx), duration=h * step, dropna=True)
    np.testing.assert_allclose(Xd, Xc)
    np.testing.assert_allclose(yd, yc)


def test_forecast_pairs_duration_requires_index():
    with pytest.raises(TypeError):
        forecast_pairs(np.arange(5.0), np.arange(5.0), duration=2)   # bare arrays


def test_forecast_pairs_duration_async_pairs_by_walltime():
    # X ticks every 10, y ticks every 15; duration 10 pairs y[t] with X as-of (t-10)
    Xv = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    Xi = np.array([0, 10, 20, 30, 40, 50], dtype=np.int64)
    yv = np.array([100.0, 200.0, 300.0])
    yi = np.array([15, 30, 45], dtype=np.int64)
    Xs, ys = forecast_pairs((Xv, Xi), (yv, yi), duration=10, dropna=True)
    # y at 15 -> X as-of 5 -> X at 0 = 1 ; y at 30 -> X as-of 20 = 3 ; y at 45 -> X as-of 35 -> X at 30 = 4
    np.testing.assert_array_equal(ys, [100.0, 200.0, 300.0])
    np.testing.assert_allclose(Xs, [1.0, 3.0, 4.0])


def test_forecast_pairs_count_dropna_also_drops_target_nan():
    # dropna must return a clean training set: rows with a NaN target are dropped too
    # (matching duration mode), not just feature-warmup rows.
    X = np.arange(8.0)
    y = np.array([0., 1., 2., np.nan, 4., 5., np.nan, 7.])
    Xs, ys = forecast_pairs(X, y, count=1, dropna=True)
    assert not np.isnan(Xs).any() and not np.isnan(ys).any()
    # count=1 lags X by 1 (row 0 warmup dropped); rows 3 and 6 dropped for NaN target
    np.testing.assert_array_equal(Xs, [0., 1., 3., 4., 6.])   # surviving X[t-1]
    np.testing.assert_array_equal(ys, [1., 2., 4., 5., 7.])   # surviving targets


# ---------------------------------------------------------------------------
# All-regime: count mode runs identically in eager, graph, and lazy
# ---------------------------------------------------------------------------

def test_forecast_pairs_count_graph_matches_batch():
    """Graph regime: call with Input nodes, wrap in a Pipeline, run, compare."""
    from screamer import Input, Pipeline
    from screamer.dag import is_node

    X = np.arange(10.0)
    y = X * 2
    Xb, yb = forecast_pairs(X, y, count=2)

    # Build graph by calling with Node inputs
    X_in = Input("X")
    y_in = Input("y")
    combined_node = forecast_pairs(X_in, y_in, count=2)
    assert is_node(combined_node), "graph mode must return a Node"

    # Run the graph pipeline
    dag = Pipeline([X_in, y_in], [combined_node])
    result, _ = dag(X, y)
    # result is (N, 2): col0 = Xs, col1 = y
    Xg, yg = result[:, 0], result[:, 1]
    np.testing.assert_allclose(Xg, Xb, equal_nan=True)
    np.testing.assert_allclose(yg, yb)


def test_forecast_pairs_count_graph_dropna_matches_batch():
    """Graph regime with dropna=True."""
    from screamer import Input, Pipeline
    from screamer.dag import is_node

    X = np.arange(10.0)
    y = X * 2
    Xb, yb = forecast_pairs(X, y, count=2, dropna=True)

    X_in = Input("X")
    y_in = Input("y")
    combined_node = forecast_pairs(X_in, y_in, count=2, dropna=True)
    assert is_node(combined_node)

    dag = Pipeline([X_in, y_in], [combined_node])
    result, _ = dag(X, y)
    Xg, yg = result[:, 0], result[:, 1]
    np.testing.assert_allclose(Xg, Xb, equal_nan=True)
    np.testing.assert_allclose(yg, yb)


def test_forecast_pairs_count_lazy_matches_batch():
    """Lazy regime: pass (value, index) generators, collect rows, compare to batch."""
    X = np.arange(10.0)
    y = X * 2
    Xb, yb = forecast_pairs(X, y, count=2)

    gen_X = ((float(v), int(i)) for i, v in enumerate(X))
    gen_y = ((float(v), int(i)) for i, v in enumerate(y))
    it = forecast_pairs(gen_X, gen_y, count=2)
    # each event: ((xs_val, y_val), index) or bare scalar pair
    rows = list(it)
    Xl = np.array([r[0][0] for r in rows], dtype=float)
    yl = np.array([r[0][1] for r in rows], dtype=float)
    np.testing.assert_allclose(Xl, Xb, equal_nan=True)
    np.testing.assert_allclose(yl, yb)


def test_forecast_pairs_count_lazy_dropna_matches_batch():
    """Lazy regime with dropna=True."""
    X = np.arange(10.0)
    y = X * 2
    Xb, yb = forecast_pairs(X, y, count=2, dropna=True)

    gen_X = ((float(v), int(i)) for i, v in enumerate(X))
    gen_y = ((float(v), int(i)) for i, v in enumerate(y))
    it = forecast_pairs(gen_X, gen_y, count=2, dropna=True)
    rows = list(it)
    Xl = np.array([r[0][0] for r in rows], dtype=float)
    yl = np.array([r[0][1] for r in rows], dtype=float)
    np.testing.assert_allclose(Xl, Xb, equal_nan=True)
    np.testing.assert_allclose(yl, yb)


# ---------------------------------------------------------------------------
# Compliance: no numpy data-path masking in supervised.py
# ---------------------------------------------------------------------------

def test_no_numpy_datapath_in_supervised():
    """Verify banned numpy masking/glue patterns are absent from supervised.py source."""
    import screamer.supervised as S
    src = inspect.getsource(S)
    banned_patterns = (
        "np.isfinite", "np.isin", "[keep]", "[m]", "_leading_nan_mask",
        "np.column_stack", "np.concatenate", "np.delete",
        "np.stack", "np.hstack", "np.vstack",
    )
    for banned in banned_patterns:
        assert banned not in src, f"data-path numpy still present: {banned!r}"


# ---------------------------------------------------------------------------
# IMPORTANT 2: 2-D count-mode all-regime test (multi-feature X with dropna)
# ---------------------------------------------------------------------------

def test_forecast_pairs_count_2d_all_regimes():
    """2-D X, count-mode with dropna: eager == graph == lazy, all all-C++ path."""
    from screamer import Input, Pipeline

    rng = np.random.default_rng(42)
    n = 12
    # Two-feature X; first count rows will be NaN in shifted X
    X = np.column_stack([np.arange(n, dtype=float), rng.standard_normal(n)])
    y = np.arange(n, dtype=float) * 3.0
    count = 3

    # Eager baseline (the result we lock the other regimes against)
    Xb, yb = forecast_pairs(X, y, count=count, dropna=True)
    assert Xb.shape[1] == 2, "eager result must have 2 feature columns"
    assert not np.isnan(Xb).any() and not np.isnan(yb).any()

    # Graph regime
    X_in = Input("X0")
    y_in = Input("y")
    # For 2-D eager we pass X column-by-column via a 2-D array;
    # for graph we build two separate Input nodes for the two feature columns.
    # forecast_pairs with 2-D Node input is tested via the single-node API.
    # The graph path in _forecast_pairs_count returns a combined Node when
    # given Node inputs; we need to split columns from it.
    from screamer import Lag
    from screamer.streams import CombineLatest, Dropna, Select
    # Replicate the 2-D all-C++ internal wiring by hand to verify the output.
    X0_in = Input("X0")
    X1_in = Input("X1")
    y_in2 = Input("y")
    Xs0 = Lag(count)(X0_in)
    Xs1 = Lag(count)(X1_in)
    combined = CombineLatest()(Xs0, Xs1, y_in2)
    filtered = Dropna(how="any")(combined)
    dag = Pipeline([X0_in, X1_in, y_in2], [filtered])
    graph_out, _ = dag(X[:, 0], X[:, 1], y)
    Xg = graph_out[:, :2]
    yg = graph_out[:, 2]
    np.testing.assert_allclose(Xg, Xb, equal_nan=True,
                               err_msg="graph != eager for 2-D count+dropna")
    np.testing.assert_allclose(yg, yb, equal_nan=True,
                               err_msg="graph y != eager y for 2-D count+dropna")

    # Lazy regime (using the dag above)
    from tests._dag_oracle import lazy_batch as _lazy_batch
    idx = np.arange(n, dtype=np.int64)
    lazy_out, _ = _lazy_batch(dag, (X[:, 0], idx), (X[:, 1], idx), (y, idx))
    Xl = lazy_out[:, :2]
    yl = lazy_out[:, 2]
    np.testing.assert_allclose(Xl, Xb, equal_nan=True,
                               err_msg="lazy != eager for 2-D count+dropna")
    np.testing.assert_allclose(yl, yb, equal_nan=True,
                               err_msg="lazy y != eager y for 2-D count+dropna")


def test_forecast_pairs_count_2d_dropna_equals_batch_reference():
    """The current 2-D eager batch result is the concrete oracle for the refactored path.

    This test locks in exact values so any regression in _forecast_pairs_count
    (the 2-D ndim>1 branch) is caught against an independent computation.
    """
    X = np.array([[1., 10.],
                  [2., 20.],
                  [3., 30.],
                  [4., 40.],
                  [5., 50.]])
    y = np.array([100., 200., 300., 400., 500.])
    count = 2

    Xs, ys = forecast_pairs(X, y, count=count, dropna=True)

    # After lagging by 2 and dropping NaN warmup rows 0 and 1:
    # row 2: Xs=[1,10], y=300; row 3: Xs=[2,20], y=400; row 4: Xs=[3,30], y=500
    expected_Xs = np.array([[1., 10.],
                             [2., 20.],
                             [3., 30.]])
    expected_ys = np.array([300., 400., 500.])
    np.testing.assert_allclose(Xs, expected_Xs, err_msg="2-D Xs oracle mismatch")
    np.testing.assert_allclose(ys, expected_ys, err_msg="2-D ys oracle mismatch")


# ---------------------------------------------------------------------------
# All-regime: duration mode runs identically in eager, graph, and lazy
# ---------------------------------------------------------------------------

def test_forecast_pairs_duration_graph_matches_batch():
    """Graph regime (duration=): Node inputs return a Node; Pipeline matches batch."""
    from screamer.dag import Input, Pipeline, is_node

    n = 20
    rng = np.random.default_rng(7)
    idx = np.arange(n, dtype=np.int64) * 10      # spacing = 10
    X_vals = rng.standard_normal(n)
    y_vals = rng.standard_normal(n)
    duration = 30                                   # 3 ticks at spacing 10

    # Batch eager baseline.
    Xb, yb = forecast_pairs((X_vals, idx), (y_vals, idx), duration=duration, dropna=True)

    # Graph regime.
    X_in = Input("X")
    y_in = Input("y")
    node = forecast_pairs(X_in, y_in, duration=duration, dropna=True)
    assert is_node(node), "graph mode must return a Node"

    dag = Pipeline([X_in, y_in], [node])
    result, _ = dag((X_vals, idx), (y_vals, idx))
    Xg, yg = result[:, 0], result[:, 1]
    np.testing.assert_allclose(Xg, Xb, equal_nan=True,
                               err_msg="graph Xs != batch Xs (duration)")
    np.testing.assert_allclose(yg, yb, equal_nan=True,
                               err_msg="graph ys != batch ys (duration)")


def test_forecast_pairs_duration_lazy_matches_batch():
    """Lazy regime (duration=): generators produce the same output as batch."""
    n = 20
    rng = np.random.default_rng(13)
    idx = np.arange(n, dtype=np.int64) * 10
    X_vals = rng.standard_normal(n)
    y_vals = rng.standard_normal(n)
    duration = 20

    Xb, yb = forecast_pairs((X_vals, idx), (y_vals, idx), duration=duration, dropna=True)

    gen_X = ((float(v), int(i)) for v, i in zip(X_vals, idx))
    gen_y = ((float(v), int(i)) for v, i in zip(y_vals, idx))
    it = forecast_pairs(gen_X, gen_y, duration=duration, dropna=True)
    rows = list(it)
    assert len(rows) == len(Xb), f"lazy row count {len(rows)} != batch {len(Xb)}"
    Xl = np.array([r[0][0] for r in rows], dtype=float)
    yl = np.array([r[0][1] for r in rows], dtype=float)
    np.testing.assert_allclose(Xl, Xb, equal_nan=True,
                               err_msg="lazy Xs != batch Xs (duration)")
    np.testing.assert_allclose(yl, yb, equal_nan=True,
                               err_msg="lazy ys != batch ys (duration)")


# ---------------------------------------------------------------------------
# One C++ graph: eager, lazy, live and a Pipeline node give identical results
# ---------------------------------------------------------------------------

def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and np.array_equal(a, b, equal_nan=True)


def _regime_data(seed):
    rng = np.random.default_rng(seed)
    n = 60
    xi = np.cumsum(rng.integers(1, 4, size=n)).astype(np.int64)
    yi = np.cumsum(rng.integers(1, 5, size=n // 2)).astype(np.int64) + 2
    xv = rng.standard_normal(n)
    yv = rng.standard_normal(n // 2)
    xv[rng.integers(0, n, size=4)] = np.nan
    yv[rng.integers(0, n // 2, size=3)] = np.nan
    return xv, xi, yv, yi


def _live(pipe, xv, xi, yv, yi, drain_every):
    """Push the events of X and y in index order (X first on a tie), draining
    the result every `drain_every` events; the drains concatenated."""
    events = sorted([(int(i), 0, v) for v, i in zip(xv, xi)] +
                    [(int(i), 1, v) for v, i in zip(yv, yi)], key=lambda e: (e[0], e[1]))
    sess = pipe.live()
    parts = []
    for k, (i, src, v) in enumerate(events, 1):
        sess.push("X" if src == 0 else "y", i, v)
        if k % drain_every == 0:
            parts.append(sess.result())
    sess.flush()
    parts.append(sess.result())
    vals = [p[0].reshape(len(p[1]), -1) for p in parts if len(p[1])]
    idx = [p[1] for p in parts if len(p[1])]
    return (np.concatenate(vals) if vals else np.empty((0, 2))), (
        np.concatenate(idx) if idx else np.empty(0, np.int64))


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("mode", [{"count": 1}, {"count": 3}, {"duration": 0},
                                  {"duration": 4}, {"duration": 9}])
@pytest.mark.parametrize("dropna", [False, True])
def test_forecast_pairs_every_regime_is_identical(seed, mode, dropna):
    from screamer import Input, Pipeline
    xv, xi, yv, yi = _regime_data(seed)
    if "count" in mode:          # count mode pairs row t with row t: one clock
        yv, yi = xv[::-1].copy(), xi
        eager = forecast_pairs(xv, yv, dropna=dropna, **mode)
    else:
        eager = forecast_pairs((xv, xi), (yv, yi), dropna=dropna, **mode)

    gen = lambda v, i: ((float(a), int(b)) for a, b in zip(v, i))   # noqa: E731
    rows = list(forecast_pairs(gen(xv, xi), gen(yv, yi), dropna=dropna, **mode))
    lazy_vals = np.array([r[0] for r in rows]).reshape(len(rows), 2)
    lazy_idx = np.array([r[1] for r in rows], dtype=np.int64)
    assert _same(lazy_vals[:, 0], eager[0]) and _same(lazy_vals[:, 1], eager[1])

    X_in, y_in = Input("X"), Input("y")
    pipe = Pipeline([X_in, y_in], [forecast_pairs(X_in, y_in, dropna=dropna, **mode)])
    batch_vals, batch_idx = pipe((xv, xi), (yv, yi))
    assert _same(batch_vals, lazy_vals) and _same(batch_idx, lazy_idx)
    for every in (1, 7, 10_000):
        live_vals, live_idx = _live(pipe, xv, xi, yv, yi, every)
        assert _same(live_vals, lazy_vals) and _same(live_idx, lazy_idx)


def test_forecast_pairs_wide_features_equal_one_column_at_a_time():
    rng = np.random.default_rng(5)
    n = 40
    X = rng.standard_normal((n, 3))
    X[[3, 17], [0, 2]] = np.nan
    y = rng.standard_normal(n)
    idx = np.arange(n, dtype=np.int64) * 10
    for dropna in (False, True):
        Xs, ys = forecast_pairs(X, y, count=2, dropna=dropna)
        Xd, yd = forecast_pairs((X, idx), (y, idx), duration=20, dropna=dropna)
        assert Xs.shape[1] == 3 and Xd.shape[1] == 3
        if not dropna:            # each column alone gives the same pairs
            for c in range(3):
                assert _same(Xs[:, c], forecast_pairs(X[:, c], y, count=2)[0])
                # Duration samples with a carry resample, which carries a row
                # only when all of it is NaN; compare on rows without NaN.
                Xc = np.where(np.isnan(X), 0.0, X)
                Xw = forecast_pairs((Xc, idx), (y, idx), duration=20)[0]
                assert _same(Xw[:, c], forecast_pairs((Xc[:, c], idx), (y, idx), duration=20)[0])
            assert _same(ys, y) and _same(yd, y[2:])   # y starts at the first delayed X
    # a (T, 1) X keeps its rank in both modes
    assert forecast_pairs(X[:, :1], y, count=1)[0].shape == (n, 1)
    assert forecast_pairs((X[:, :1], idx), (y, idx), duration=10)[0].shape == (n - 1, 1)


def test_forecast_pairs_node_round_trips_through_to_dict():
    from screamer import Input, Pipeline
    X_in, y_in = Input("X"), Input("y")
    pipe = Pipeline([X_in, y_in], [forecast_pairs(X_in, y_in, duration=20, dropna=True)])
    d = pipe.to_dict()
    ops = [n for n in d["nodes"] if n["kind"] == "operator"]
    assert [n["op"] for n in ops] == ["forecast_pairs"]
    assert ops[0]["params"] == {"duration": 20, "dropna": True}
    rng = np.random.default_rng(3)
    idx = np.arange(30, dtype=np.int64) * 10
    feeds = ((rng.standard_normal(30), idx), (rng.standard_normal(30), idx))
    a, b = pipe(*feeds), Pipeline.from_dict(d)(*feeds)
    assert _same(a[0], b[0]) and _same(a[1], b[1])


def test_forecast_pairs_settings_and_errors():
    from screamer import Input
    v = np.arange(6.0)
    with pytest.raises(TypeError, match="count must be a number, got str"):
        forecast_pairs(v, v, count="1")
    with pytest.raises(TypeError, match="duration must be a number, got str"):
        forecast_pairs((v, None), (v, None), duration="1")
    with pytest.raises(ValueError, match="count must be at least 1"):
        forecast_pairs(v, v, count=0)
    with pytest.raises(ValueError, match="count must be at least 1"):
        forecast_pairs(v, v, count=-3)
    with pytest.raises(ValueError, match="same length"):
        forecast_pairs(v, v[:4], count=1)
    with pytest.raises(TypeError, match="both as Nodes"):
        forecast_pairs(Input("X"), v, count=1)
    # a float setting is truncated and a bool is a number, as for Delay
    assert _same(forecast_pairs(v, v, count=1.5)[0], forecast_pairs(v, v, count=1)[0])
    assert _same(forecast_pairs(v, v, count=True)[0], forecast_pairs(v, v, count=1)[0])
