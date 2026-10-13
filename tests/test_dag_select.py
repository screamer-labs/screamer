import numpy as np
import pytest

from screamer import Input, Pipeline, RollingMean
from screamer.streams import Select, CombineLatest
from tests._dag_oracle import lazy_batch as _lazy_batch


def test_select_column_from_combine_latest():
    ak = np.array([1, 2, 3], dtype=np.int64); av = np.array([10.0, 20.0, 30.0])
    bk = np.array([1, 2, 3], dtype=np.int64); bv = np.array([1.0, 2.0, 3.0])
    a, b = Input("a"), Input("b")
    # CombineLatest()(a, b) is width-2; select column 0 -> a's latest.
    dag = Pipeline(inputs=[a, b], outputs=[Select(0)(CombineLatest()(a, b))])
    bv_, bk_ = dag((av, ak), (bv, bk))     # (values, index) feeds; values-first result
    sv_, sk_ = _lazy_batch(dag, (av, ak), (bv, bk))
    np.testing.assert_array_equal(bk_, sk_)
    np.testing.assert_array_equal(bv_, sv_)
    # eager oracle (values-first): align, then project – regression-proof against
    # CombineLatest emit changes (no hand-counted row expectations).
    cv, ck = CombineLatest()(av, bv, index=[ak, bk])
    ev, ek = Select(0)(cv, index=ck)
    np.testing.assert_array_equal(bk_, ek)
    np.testing.assert_array_equal(bv_.reshape(-1), ev)


def test_select_two_columns_reorder():
    ak = np.array([1, 2], dtype=np.int64); av = np.array([10.0, 20.0])
    bk = np.array([1, 2], dtype=np.int64); bv = np.array([1.0, 2.0])
    a, b = Input("a"), Input("b")
    dag = Pipeline(inputs=[a, b], outputs=[Select([1, 0])(CombineLatest()(a, b))])
    bv_, bk_ = dag((av, ak), (bv, bk))
    sv_, sk_ = _lazy_batch(dag, (av, ak), (bv, bk))
    np.testing.assert_array_equal(bk_, sk_)
    np.testing.assert_array_equal(bv_, sv_)
    # [1,0] swaps columns; compare against the eager oracle.
    cv, ck = CombineLatest()(av, bv, index=[ak, bk])
    ev, ek = Select([1, 0])(cv, index=ck)
    np.testing.assert_array_equal(bk_, ek)
    np.testing.assert_array_equal(bv_, ev)


def test_select_feeds_functor():
    ak = np.array([1, 2, 3], dtype=np.int64); av = np.array([2.0, 4.0, 6.0])
    bk = np.array([1, 2, 3], dtype=np.int64); bv = np.array([0.0, 0.0, 0.0])
    a, b = Input("a"), Input("b")
    # select a's column then smooth it
    dag = Pipeline(inputs=[a, b], outputs=[RollingMean(2)(Select(0)(CombineLatest()(a, b)))])
    bv_, bk_ = dag((av, ak), (bv, bk))
    sv_, sk_ = _lazy_batch(dag, (av, ak), (bv, bk))
    np.testing.assert_array_equal(bk_, sk_)
    np.testing.assert_array_equal(bv_, sv_)
    # value oracle: same graph computed eagerly (align -> select col 0 -> RollingMean(2))
    cv, ck = CombineLatest()(av, bv, index=[ak, bk])
    sel, _ = Select(0)(cv, index=ck)
    ev = RollingMean(2)(sel)
    np.testing.assert_array_equal(bv_.reshape(-1), ev)


def test_select_out_of_range_errors_batch_and_stream():
    ak = np.array([1], dtype=np.int64); av = np.array([10.0])
    bk = np.array([1], dtype=np.int64); bv = np.array([1.0])
    a, b = Input("a"), Input("b")
    # CombineLatest()(a, b) is width-2; column 5 is out of range.
    dag = Pipeline(inputs=[a, b], outputs=[Select(5)(CombineLatest()(a, b))])
    # The graph is checked against the input widths before any event is fed.
    with pytest.raises(ValueError, match="select needs column 5"):
        dag((av, ak), (bv, bk))
    with pytest.raises(ValueError, match="select needs column 5"):
        _lazy_batch(dag, (av, ak), (bv, bk))


def test_select_missing_columns_raises_dag():
    """columns is a required positional argument; omitting it raises TypeError."""
    with pytest.raises(TypeError):
        Select()(np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_positional_select_equals_the_pipeline_and_never_aliases():
    # A positional select copies columns directly; the result must equal the
    # Pipeline's bit for bit, in its shape, and never share the input's memory.
    from screamer import streams as S
    rng = np.random.default_rng(7)
    for _ in range(300):
        T, w, one = int(rng.integers(0, 6)), int(rng.integers(1, 4)), rng.random() < 0.4
        v = rng.normal(size=(T,) if one else (T, w))
        v[rng.random(v.shape) < 0.3] = np.nan
        width = 1 if one else w
        cols = (int(rng.integers(0, width)) if rng.random() < 0.5 else
                [int(rng.integers(0, width)) for _ in range(int(rng.integers(1, 4)))])
        got, idx = S.select(v, cols)
        c, scalar = S._normalize_columns(cols)
        arg = c[0] if scalar else c
        dag = S._one_node_pipeline(1, lambda x: S.select(x, columns=arg), "select")
        want, _ = S._run_eager_pipeline(dag, [(v, None)], "select")
        assert idx is None and got.shape == want.shape and got.tobytes() == want.tobytes()
        assert not np.shares_memory(got, v)


def test_a_scalar_index_names_the_index():
    import screamer as s
    with pytest.raises(TypeError, match=r"the index of a \(values, index\) stream must be a sequence"):
        s.Select(0)(np.arange(3.0), index=5)
