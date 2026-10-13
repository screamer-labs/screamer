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
    with pytest.raises(TypeError, match="count must be an integer, got str"):
        forecast_pairs(v, v, count="1")
    with pytest.raises(TypeError, match="duration must be an integer, got str"):
        forecast_pairs((v, None), (v, None), duration="1")
    with pytest.raises(ValueError, match="count must be at least 1"):
        forecast_pairs(v, v, count=0)
    with pytest.raises(ValueError, match="count must be at least 1"):
        forecast_pairs(v, v, count=-3)
    with pytest.raises(ValueError, match="same length"):
        forecast_pairs(v, v[:4], count=1)
    with pytest.raises(TypeError, match="both as Nodes"):
        forecast_pairs(Input("X"), v, count=1)
    # a float setting is truncated; a bool is not a number setting (G25)
    assert _same(forecast_pairs(v, v, count=1.5)[0], forecast_pairs(v, v, count=1)[0])
    for flag in (True, np.True_):
        with pytest.raises(TypeError, match="count must be an integer, got bool"):
            forecast_pairs(v, v, count=flag)


# ---------------------------------------------------------------------------
# Seeded random cases: every regime equals eager, and eager equals a plain
# reference loop. Wide X in both modes, NaN values, int and datetime64 indexes,
# dropna on and off.
# ---------------------------------------------------------------------------

def _reference_count(X2, y, count):
    """Row t: X[t - count] (a NaN row before), and y[t]."""
    lagged = np.full_like(X2, np.nan)
    if count < len(X2):
        lagged[count:] = X2[:len(X2) - count]
    return np.column_stack([lagged, y])


def _reference_duration(X2, xi, y, yi, duration):
    """At each y tick t: each column holds the last non-NaN value of the X
    rows whose index + duration is at or before t (per column carry, G23). A
    row is emitted once some column has a value."""
    dx = xi + duration
    rows, last, prev = [], np.full(X2.shape[1], np.nan), None
    started = False
    for j, t in enumerate(yi):
        sel = (dx <= t) if prev is None else ((dx > prev) & (dx <= t))
        for row in X2[sel]:
            ok = ~np.isnan(row)
            last[ok] = row[ok]
            started = started or bool(ok.any())
        prev = t
        if started:
            rows.append(np.append(last, y[j]))
    return np.array(rows).reshape(len(rows), X2.shape[1] + 1)


def _random_case(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(0, 50))
    k = int(rng.choice([1, 1, 2, 4]))
    wide = k > 1 or rng.random() < 0.3                     # (T, 1) as well as (T,)
    X = rng.standard_normal((n, k))
    X[rng.random((n, k)) < 0.15] = np.nan
    xi = np.cumsum(rng.integers(1, 4, size=n)).astype(np.int64)
    m = int(rng.integers(0, 40))
    yi = np.cumsum(rng.integers(1, 5, size=m)).astype(np.int64)
    yv = rng.standard_normal(m)
    yv[rng.random(m) < 0.15] = np.nan
    yc = rng.standard_normal(n)
    yc[rng.random(n) < 0.15] = np.nan
    mode = {"count": int(rng.integers(1, 6))} if rng.random() < 0.5 else \
        {"duration": int(rng.integers(0, 12))}
    dropna = bool(rng.random() < 0.5)
    dates = bool(rng.random() < 0.4)
    return X if wide else X[:, 0], xi, yv, yi, yc, mode, dropna, dates


def _rows(xs, ys, k):
    return np.column_stack([np.asarray(xs).reshape(len(ys), k), ys])


@pytest.mark.parametrize("chunk", range(12))
def test_forecast_pairs_random_cases_all_regimes(chunk):
    from screamer import Input, Pipeline
    for seed in range(chunk * 30, chunk * 30 + 30):          # 360 cases
        X, xi, yv, yi, yc, mode, dropna, dates = _random_case(seed)
        X2 = X if X.ndim == 2 else X[:, None]
        if "count" in mode:                                 # one clock: row t with row t
            yv, yi = yc, xi
        ix, iy = ((xi.astype("datetime64[s]"), yi.astype("datetime64[s]")) if dates
                  else (xi, yi))
        if "count" in mode:
            xs, ys = forecast_pairs(X, yv, dropna=dropna, **mode)
            ref = _reference_count(X2, yv, mode["count"])
        else:
            xs, ys = forecast_pairs((X, ix), (yv, iy), dropna=dropna, **mode)
            ref = _reference_duration(X2, xi, yv, yi, mode["duration"])
        if dropna:
            ref = ref[~np.isnan(ref).any(axis=1)]
        k = X2.shape[1]
        got = _rows(xs, ys, k)
        assert xs.ndim == X.ndim, seed
        assert _same(got, ref), seed

        # lazy, a Pipeline node (batch) and a live session give the same rows
        def events(vals, idx):
            return ((v.tolist() if np.ndim(v) else float(v), i) for v, i in zip(vals, idx))
        lazy = list(forecast_pairs(events(X, ix), events(yv, iy), dropna=dropna, **mode))
        lazy_rows = np.array([r[0] for r in lazy], dtype=float).reshape(len(lazy), k + 1)
        assert _same(lazy_rows, got), seed
        X_in, y_in = Input("X"), Input("y")
        pipe = Pipeline([X_in, y_in], [forecast_pairs(X_in, y_in, dropna=dropna, **mode)])
        vals, idx = pipe((X, ix), (yv, iy))
        assert _same(np.asarray(vals).reshape(len(idx), k + 1), got), seed
        sess = pipe.live()
        evs = sorted([(int(t), 0, r) for t, r in zip(xi, X)] +
                     [(int(t), 1, v) for t, v in zip(yi, yv)], key=lambda e: e[:2])
        for t, src, v in evs:
            stamp = np.datetime64(t, "s") if dates else t
            sess.push("X" if src == 0 else "y", stamp, v)
        sess.flush()
        lv, li = sess.result()
        assert _same(np.asarray(lv).reshape(len(li), k + 1), got), seed
        assert _same(li, idx), seed


def test_forecast_pairs_wide_x_regimes_with_dates_and_nan():
    """Wide X with a datetime index and NaN: eager == lazy == node == live, in
    both modes (a fixed seed)."""
    from screamer import Input, Pipeline
    rng = np.random.default_rng(2026)
    n = 80
    X = rng.standard_normal((n, 3))
    X[rng.random((n, 3)) < 0.2] = np.nan
    xi = (np.cumsum(rng.integers(1, 3, size=n)) * 60).astype("datetime64[s]")
    yi = (np.cumsum(rng.integers(1, 4, size=50)) * 60 + 30).astype("datetime64[s]")
    yv = rng.standard_normal(50)
    yc = rng.standard_normal(n)
    for mode, ys_, iy in (({"count": 3}, yc, xi), ({"duration": 120}, yv, yi)):
        for dropna in (False, True):
            if "count" in mode:
                xs, ys = forecast_pairs(X, ys_, dropna=dropna, **mode)
            else:
                xs, ys = forecast_pairs((X, xi), (ys_, iy), dropna=dropna, **mode)
            got = _rows(xs, ys, 3)
            assert xs.shape[1] == 3 and len(got) > 0
            lazy = list(forecast_pairs(((r.tolist(), t) for r, t in zip(X, xi)),
                                       ((float(v), t) for v, t in zip(ys_, iy)),
                                       dropna=dropna, **mode))
            assert _same(np.array([r[0] for r in lazy]), got)
            X_in, y_in = Input("X"), Input("y")
            pipe = Pipeline([X_in, y_in], [forecast_pairs(X_in, y_in, dropna=dropna, **mode)])
            vals, idx = pipe((X, xi), (ys_, iy))
            assert _same(vals, got)
            # a live session drained every 9 events: the drains concatenated
            # equal the batch rows
            evs = sorted([(t, 0, r) for t, r in zip(xi, X)] +
                         [(t, 1, v) for t, v in zip(iy, ys_)], key=lambda e: e[:2])
            sess2 = pipe.live()
            parts = []
            for k, (t, src, v) in enumerate(evs, 1):
                sess2.push("X" if src == 0 else "y", t, v)
                if k % 9 == 0:
                    parts.append(sess2.result())
            sess2.flush()
            parts.append(sess2.result())
            live = np.concatenate([np.asarray(p[0]).reshape(len(p[1]), -1)
                                   for p in parts if len(p[1])])
            assert _same(live, got)


def test_forecast_pairs_input_rules():
    from screamer import Input, Pipeline
    v = np.arange(6.0)
    i = np.arange(6) * 10
    y2 = np.column_stack([v, v])
    with pytest.raises(ValueError, match=r"y holds 2 columns, expected 1"):
        forecast_pairs(v, y2, count=1)
    with pytest.raises(ValueError, match=r"y holds 2 columns, expected 1"):
        forecast_pairs((v, i), (y2, i), duration=1)
    X_in, y_in = Input("X"), Input("y")
    pipe = Pipeline([X_in, y_in], [forecast_pairs(X_in, y_in, count=1)])
    with pytest.raises(ValueError, match=r"y holds 2 columns, expected 1"):
        pipe(v, y2)
    with pytest.raises(ValueError, match=r"y holds 2 columns, expected 1"):
        pipe.live().push("y", 0, [1.0, 2.0])
    with pytest.raises(ValueError, match=r"X has 3 dimensions"):
        forecast_pairs(v.reshape(6, 1, 1), v, count=1)
    with pytest.raises(ValueError, match=r"X has 3 dimensions"):
        forecast_pairs((v.reshape(6, 1, 1), i), (v, i), duration=1)
    with pytest.raises(ValueError, match=r"X has 0 columns, expected at least 1"):
        forecast_pairs(np.empty((6, 0)), v, count=1)
    with pytest.raises(TypeError, match="pairs are for duration="):
        forecast_pairs((v, i), (v, i), count=1)
    with pytest.raises(ValueError, match="duration must be at least 0"):
        forecast_pairs((v, i), (v, i), duration=-1)
    with pytest.raises(ValueError, match=r"count must be at least 1 and at most 2147483647, "
                                         r"got 1180591620717411303424"):
        forecast_pairs(v, v, count=2**70)
    with pytest.raises(ValueError, match=r"got 1099511627776"):
        forecast_pairs(v, v, count=2**40)
    with pytest.raises(ValueError, match=r"duration must be at least 0 .* got -1180591620717"):
        forecast_pairs((v, i), (v, i), duration=-2**70)
    with pytest.raises(TypeError, match="dropna must be a bool, got str"):
        forecast_pairs(v, v, count=1, dropna="no")
    with pytest.raises(TypeError, match="dropna must be a bool, got int"):
        forecast_pairs(v, v, count=1, dropna=1)
    assert _same(forecast_pairs(v, v, count=1, dropna=np.bool_(True))[0],
                 forecast_pairs(v, v, count=1, dropna=True)[0])
    # (T, 1) y is one column; an empty wide X keeps its width
    assert _same(forecast_pairs(v, v[:, None], count=1)[1], v)
    assert forecast_pairs(np.empty((0, 3)), np.empty(0), count=1)[0].shape == (0, 3)


def test_graph_builder_compiles_once():
    from screamer import screamer_bindings as _b
    gb = _b._GraphBuilder()
    x = gb.add_input()
    gb.set_outputs([gb.add_dropna([x], False)])
    gb.compile()
    with pytest.raises(RuntimeError, match="compiled already"):
        gb.compile()


# ---------------------------------------------------------------------------
# Pipeline feeds: a rank-3 feed raises in every graph
# ---------------------------------------------------------------------------

def _graphs():
    from screamer import Input, Pipeline, RollingMean
    from screamer.streams import Dropna
    x = Input("x")
    yield "functor", Pipeline([x], [RollingMean(2)(x)]), 1
    x = Input("x")
    yield "dropna", Pipeline([x], [Dropna()(x)]), 1
    X_in, y_in = Input("X"), Input("y")
    yield "forecast count", Pipeline([X_in, y_in], [forecast_pairs(X_in, y_in, count=1)]), 2
    X_in, y_in = Input("X"), Input("y")
    yield "forecast duration", Pipeline([X_in, y_in], [forecast_pairs(X_in, y_in, duration=1)]), 2


@pytest.mark.parametrize("which", range(4))
def test_pipeline_refuses_rank3_feeds(which):
    name, pipe, n_in = list(_graphs())[which]
    v = np.arange(8.0)
    rank3 = np.arange(16.0).reshape(8, 2, 1)
    rest = (v,) * (n_in - 1)
    with pytest.raises(ValueError, match=r"Pipeline: the values of input 0 have shape "
                                         r"\[8,2,1\], expected \(T,\) or \(T, w\)"):
        pipe(rank3, *rest)
    with pytest.raises(ValueError, match=r"Pipeline: the values of input 0 have shape"):
        pipe((rank3, np.arange(8)), *[(r, np.arange(8)) for r in rest])
    if name.startswith("forecast"):      # the graph requires X to have a column
        with pytest.raises(ValueError, match=r"forecast_pairs: X has 0 columns, "
                                             r"expected at least 1"):
            pipe(np.empty((8, 0)), *rest)
    # a lazy feed and a live push refuse a 2-D event already
    with pytest.raises(ValueError):
        list(pipe(iter([(np.ones((2, 2)), 0)]), *[iter([(1.0, 0)])] * (n_in - 1)))
    with pytest.raises(ValueError, match="1-D sequence"):
        pipe.live().push(0, 0, np.ones((2, 2)))


def test_forecast_pairs_large_count_costs_memory_only_as_events_arrive():
    import resource
    import sys
    from screamer import Input, Pipeline

    def peak_mb():                     # ru_maxrss is in bytes on macOS, KB on Linux
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 2**20 if sys.platform == "darwin" else r / 2**10

    v = np.arange(10.0)
    before = peak_mb()
    xs, ys = forecast_pairs(v, v, count=2**31 - 1)
    assert np.isnan(xs).all() and _same(ys, v)
    X_in, y_in = Input("X"), Input("y")
    pipe = Pipeline([X_in, y_in], [forecast_pairs(X_in, y_in, count=2**31 - 1)])
    vals, _ = pipe(np.ones((10, 3)), v)
    assert vals.shape == (10, 4) and np.isnan(vals[:, :3]).all()
    # A count-sized buffer (16 GB) would raise the process peak far above any
    # earlier test's, so the peak's growth is the measure here.
    assert peak_mb() - before < 64


@pytest.mark.parametrize("count", [1, 63, 64, 65, 200, 500])
def test_forecast_pairs_lag_across_ring_growth(count):
    rng = np.random.default_rng(count)
    for k in (1, 3):
        X = rng.standard_normal((450, k))
        X[rng.random(X.shape) < 0.1] = np.nan
        y = rng.standard_normal(450)
        Xk = X if k > 1 else X[:, 0]
        xs, ys = forecast_pairs(Xk, y, count=count)
        assert _same(_rows(xs, ys, k), _reference_count(X, y, count))
        # the lazy path drives the same node event by event
        lazy = list(forecast_pairs(((r.tolist() if k > 1 else float(r[0]), t)
                                    for t, r in enumerate(X)),
                                   ((float(v), t) for t, v in enumerate(y)), count=count))
        assert _same(np.array([r[0] for r in lazy]).reshape(450, k + 1), _rows(xs, ys, k))


@pytest.mark.parametrize("mode", [{"count": 1}, {"duration": 1}])
def test_forecast_pairs_zero_width_x_raises_in_every_regime(mode):
    """A (T, 0) X raises the same error eagerly and in a node, before any event:
    the graph's width rule, not a node, refuses it. Lazy and live calls refuse
    an event with no values."""
    from screamer import Input, Pipeline
    v = np.arange(6.0)
    i = np.arange(6)
    msg = r"forecast_pairs: X has 0 columns, expected at least 1"
    eager = (lambda: forecast_pairs(np.empty((6, 0)), v, **mode)) if "count" in mode else \
        (lambda: forecast_pairs((np.empty((6, 0)), i), (v, i), **mode))
    with pytest.raises(ValueError, match=msg):
        eager()
    X_in, y_in = Input("X"), Input("y")
    pipe = Pipeline([X_in, y_in], [forecast_pairs(X_in, y_in, **mode)])
    with pytest.raises(ValueError, match=msg):
        pipe((np.empty((6, 0)), i), (v, i))
    # the pipeline still runs afterwards: nothing was pushed
    vals, _ = pipe((v, i), (v, i))
    assert vals.shape[1] == 2
    with pytest.raises(ValueError, match="needs at least one value|is empty"):
        pipe.live().push("X", 0, [])
    with pytest.raises(ValueError):
        list(forecast_pairs(iter([([], 0)]), iter([(1.0, 0)]), **mode))
