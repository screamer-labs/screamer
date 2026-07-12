"""Equivalence tests: CamelCase class form == lowercase function form.

For each of the five operators (Resample, Dropna, Select, CombineLatest, Merge)
assert byte-identical output across:
- batch regime (raw arrays / Streams)
- lazy regime (generator iterators)
- graph regime (Node inputs / Pipeline compilation)

These tests PIN the class-function equivalence before any call-site migration,
so that later batch tasks can replace the function calls safely.
"""
import itertools

import numpy as np
import pytest

from screamer import Input, Pipeline
from screamer.streams import (
    CombineLatest,
    Dropna,
    Merge,
    Resample,
    Select,
    combine_latest,
    dropna,
    merge,
    resample,
    select,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eq(a, b):
    """Compare two results that may be arrays, tuples, or None (NaN-aware)."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, tuple) and isinstance(b, tuple):
        return all(_eq(x, y) for x, y in zip(a, b))
    a, b = np.asarray(a), np.asarray(b)
    return np.array_equal(a, b, equal_nan=True)


def _lazy(vals, idx=None):
    """Produce a lazy iterator of (value, index) events from arrays."""
    if idx is None:
        for v in vals:
            yield float(v), None
    else:
        for v, k in zip(vals, idx):
            yield float(v), int(k)


def _collect_lazy(it):
    """Collect a lazy iterator of (value, index) events to (vals, idx)."""
    rows = list(it)
    if not rows:
        return np.array([]), None
    vals = np.array([r[0] for r in rows])
    idx0 = rows[0][1]
    if idx0 is None:
        return vals, None
    return vals, np.array([r[1] for r in rows])


def _collect_lazy_multi(it):
    """Collect a lazy iterator of (tuple_of_vals, index) to (vals_2d, idx)."""
    rows = list(it)
    if not rows:
        return np.empty((0, 0)), None
    vals = np.array([list(r[0]) for r in rows])
    idx0 = rows[0][1]
    if idx0 is None:
        return vals, None
    return vals, np.array([r[1] for r in rows])


# ---------------------------------------------------------------------------
# Resample
# ---------------------------------------------------------------------------

class TestResample:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    idx = np.array([0, 1, 2, 3, 4, 5])

    def test_batch_freq_equals_function(self):
        cls_out = Resample(freq=3, agg="sum")(self.x, self.idx)
        fn_out = resample(self.x, self.idx, freq=3, agg="sum")
        assert _eq(cls_out, fn_out)

    def test_batch_count_equals_function(self):
        cls_out = Resample(count=2, agg="last")(self.x, self.idx)
        fn_out = resample(self.x, self.idx, count=2, agg="last")
        assert _eq(cls_out, fn_out)

    def test_batch_defaults_equal_function(self):
        # both use count=None, freq=None => would fail validation unless one is given
        cls_out = Resample(freq=2)(self.x, self.idx)
        fn_out = resample(self.x, self.idx, freq=2)
        assert _eq(cls_out, fn_out)

    def test_graph_class_equals_function(self):
        src = Input("x")
        dag_cls = Pipeline([src], [Resample(freq=3, agg="sum")(src)])
        dag_fn = Pipeline([src], [resample(src, freq=3, agg="sum")])
        feed = (self.x, self.idx)
        v_cls, k_cls = dag_cls(feed)
        v_fn, k_fn = dag_fn(feed)
        np.testing.assert_array_equal(v_cls, v_fn)
        np.testing.assert_array_equal(k_cls, k_fn)

    def test_graph_count_equals_function(self):
        src = Input("x")
        dag_cls = Pipeline([src], [Resample(count=2, agg="mean")(src)])
        dag_fn = Pipeline([src], [resample(src, count=2, agg="mean")])
        feed = (self.x, self.idx)
        v_cls, k_cls = dag_cls(feed)
        v_fn, k_fn = dag_fn(feed)
        np.testing.assert_array_equal(v_cls, v_fn)
        np.testing.assert_array_equal(k_cls, k_fn)

    def test_lazy_equals_batch(self):
        cls_it = Resample(freq=2, agg="sum")(
            _lazy(self.x, self.idx))
        fn_it = resample(_lazy(self.x, self.idx), freq=2, agg="sum")
        v_cls, k_cls = _collect_lazy(cls_it)
        v_fn, k_fn = _collect_lazy(fn_it)
        np.testing.assert_array_equal(v_cls, v_fn)
        np.testing.assert_array_equal(k_cls, k_fn)


# ---------------------------------------------------------------------------
# Dropna
# ---------------------------------------------------------------------------

class TestDropna:
    x = np.array([1.0, float("nan"), 3.0, float("nan"), 5.0])
    idx = np.array([0, 1, 2, 3, 4])

    def test_batch_any_equals_function(self):
        assert _eq(Dropna()(self.x, self.idx), dropna(self.x, self.idx))

    def test_batch_how_all_equals_function(self):
        x2d = np.column_stack([self.x, self.x])
        assert _eq(Dropna(how="all")(x2d, self.idx),
                   dropna(x2d, self.idx, how="all"))

    def test_batch_stream_input(self):
        s = (self.x, self.idx)
        cls_out = Dropna()(s)
        fn_out = dropna(s)
        assert _eq(cls_out, fn_out)

    def test_graph_class_equals_function(self):
        src = Input("x")
        dag_cls = Pipeline([src], [Dropna()(src)])
        dag_fn = Pipeline([src], [dropna(src)])
        feed = (self.x, self.idx)
        v_cls, k_cls = dag_cls(feed)
        v_fn, k_fn = dag_fn(feed)
        np.testing.assert_array_equal(v_cls, v_fn)
        np.testing.assert_array_equal(k_cls, k_fn)

    def test_graph_how_all(self):
        a, b = Input("a"), Input("b")
        dag_cls = Pipeline([a, b], [Dropna(how="all")(CombineLatest()(a, b))])
        dag_fn = Pipeline([a, b], [dropna(combine_latest(a, b), how="all")])
        fa = (np.array([1.0, float("nan"), 3.0]), np.array([0, 1, 2]))
        fb = (np.array([10.0, 20.0, float("nan")]), np.array([0, 1, 2]))
        v_cls, k_cls = dag_cls(fa, fb)
        v_fn, k_fn = dag_fn(fa, fb)
        np.testing.assert_array_equal(v_cls, v_fn)
        np.testing.assert_array_equal(k_cls, k_fn)

    def test_lazy_equals_function(self):
        cls_out_v, cls_out_k = _collect_lazy(Dropna()(_lazy(self.x, self.idx)))
        fn_out_v, fn_out_k = _collect_lazy(dropna(_lazy(self.x, self.idx)))
        np.testing.assert_array_equal(cls_out_v, fn_out_v)
        np.testing.assert_array_equal(cls_out_k, fn_out_k)


# ---------------------------------------------------------------------------
# Select
# ---------------------------------------------------------------------------

class TestSelect:
    vals = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    idx = np.array([0, 1, 2])

    def test_batch_scalar_col_equals_function(self):
        cls_out = Select(0)(self.vals, self.idx)
        fn_out = select(self.vals, 0, self.idx)
        assert _eq(cls_out, fn_out)

    def test_batch_list_col_equals_function(self):
        cls_out = Select([0, 1])(self.vals, self.idx)
        fn_out = select(self.vals, [0, 1], self.idx)
        assert _eq(cls_out, fn_out)

    def test_batch_stream_input(self):
        s = (self.vals, self.idx)
        cls_out = Select(1)(s)
        fn_out = select(s, 1)
        assert _eq(cls_out, fn_out)

    def test_graph_class_equals_function(self):
        a, b = Input("a"), Input("b")
        dag_cls = Pipeline([a, b], [Select([0])(CombineLatest()(a, b))])
        dag_fn = Pipeline([a, b], [select(combine_latest(a, b), columns=[0])])
        fa = (np.array([1.0, 2.0, 3.0]), np.array([0, 1, 2]))
        fb = (np.array([10.0, 20.0, 30.0]), np.array([0, 1, 2]))
        v_cls, k_cls = dag_cls(fa, fb)
        v_fn, k_fn = dag_fn(fa, fb)
        np.testing.assert_array_equal(v_cls, v_fn)
        np.testing.assert_array_equal(k_cls, k_fn)

    def test_lazy_scalar_equals_function(self):
        events = [((1.0, 10.0), 0), ((2.0, 20.0), 1), ((3.0, 30.0), 2)]
        cls_v, cls_k = _collect_lazy(Select(0)(iter(events)))
        fn_v, fn_k = _collect_lazy(select(iter(events), 0))
        np.testing.assert_array_equal(cls_v, fn_v)
        np.testing.assert_array_equal(cls_k, fn_k)


# ---------------------------------------------------------------------------
# CombineLatest
# ---------------------------------------------------------------------------

class TestCombineLatest:
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([10.0, 20.0, 30.0])
    idx_a = np.array([0, 2, 4])
    idx_b = np.array([1, 3, 5])

    def test_batch_positional_equals_function(self):
        cls_out = CombineLatest()(self.a, self.b)
        fn_out = combine_latest(self.a, self.b)
        assert _eq(cls_out, fn_out)

    def test_batch_indexed_when_all_equals_function(self):
        cls_out = CombineLatest(emit="when_all")(
            self.a, self.b, index=[self.idx_a, self.idx_b])
        fn_out = combine_latest(
            self.a, self.b, index=[self.idx_a, self.idx_b], emit="when_all")
        assert _eq(cls_out, fn_out)

    def test_batch_on_any_equals_function(self):
        cls_out = CombineLatest(emit="on_any")(
            self.a, self.b, index=[self.idx_a, self.idx_b])
        fn_out = combine_latest(
            self.a, self.b, index=[self.idx_a, self.idx_b], emit="on_any")
        assert _eq(cls_out, fn_out)

    def test_graph_class_equals_function(self):
        a, b = Input("a"), Input("b")
        dag_cls = Pipeline([a, b], [CombineLatest()(a, b)])
        dag_fn = Pipeline([a, b], [combine_latest(a, b)])
        fa = (self.a, self.idx_a)
        fb = (self.b, self.idx_b)
        v_cls, k_cls = dag_cls(fa, fb)
        v_fn, k_fn = dag_fn(fa, fb)
        np.testing.assert_array_equal(v_cls, v_fn)
        np.testing.assert_array_equal(k_cls, k_fn)

    def test_graph_on_any_equals_function(self):
        a, b = Input("a"), Input("b")
        dag_cls = Pipeline([a, b], [CombineLatest(emit="on_any")(a, b)])
        dag_fn = Pipeline([a, b], [combine_latest(a, b, emit="on_any")])
        fa = (self.a, self.idx_a)
        fb = (self.b, self.idx_b)
        v_cls, k_cls = dag_cls(fa, fb)
        v_fn, k_fn = dag_fn(fa, fb)
        assert np.array_equal(v_cls, v_fn, equal_nan=True)
        np.testing.assert_array_equal(k_cls, k_fn)

    def test_graph_dropna_compose(self):
        """Dropna()(CombineLatest()(a, b)) compiles and matches the function form."""
        a, b = Input("a"), Input("b")
        dag_cls = Pipeline([a, b], [Dropna()(CombineLatest()(a, b))])
        dag_fn = Pipeline([a, b], [dropna(combine_latest(a, b))])
        fa = (np.array([1.0, float("nan"), 3.0]), np.array([0, 1, 2]))
        fb = (np.array([10.0, 20.0, 30.0]), np.array([0, 1, 2]))
        v_cls, k_cls = dag_cls(fa, fb)
        v_fn, k_fn = dag_fn(fa, fb)
        np.testing.assert_array_equal(v_cls, v_fn)
        np.testing.assert_array_equal(k_cls, k_fn)

    def test_lazy_indexed_equals_function(self):
        def _ev_a():
            for v, k in zip(self.a, self.idx_a):
                yield float(v), int(k)

        def _ev_b():
            for v, k in zip(self.b, self.idx_b):
                yield float(v), int(k)

        cls_rows = list(CombineLatest()(_ev_a(), _ev_b()))
        fn_rows = list(combine_latest(_ev_a(), _ev_b()))
        assert len(cls_rows) == len(fn_rows)
        for (cv, ck), (fv, fk) in zip(cls_rows, fn_rows):
            assert cv == fv
            assert ck == fk


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

class TestMerge:
    a = np.array([1.0, 3.0, 5.0])
    b = np.array([2.0, 4.0, 6.0])
    idx_a = np.array([0, 2, 4])
    idx_b = np.array([1, 3, 5])

    def test_batch_positional_equals_function(self):
        cls_v, cls_s, cls_k = Merge()(self.a, self.b)
        fn_v, fn_s, fn_k = merge(self.a, self.b)
        assert _eq(cls_v, fn_v)
        assert _eq(cls_s, fn_s)
        assert cls_k is None and fn_k is None

    def test_batch_indexed_equals_function(self):
        cls_v, cls_s, cls_k = Merge()(
            self.a, self.b, index=[self.idx_a, self.idx_b])
        fn_v, fn_s, fn_k = merge(
            self.a, self.b, index=[self.idx_a, self.idx_b])
        assert _eq(cls_v, fn_v)
        assert _eq(cls_s, fn_s)
        assert _eq(cls_k, fn_k)

    def test_lazy_equals_function(self):
        def _lazy_a():
            for v, k in zip(self.a, self.idx_a):
                yield float(v), int(k)

        def _lazy_b():
            for v, k in zip(self.b, self.idx_b):
                yield float(v), int(k)

        cls_rows = list(Merge()(_lazy_a(), _lazy_b()))
        fn_rows = list(merge(_lazy_a(), _lazy_b()))
        assert len(cls_rows) == len(fn_rows)
        for cr, fr in zip(cls_rows, fn_rows):
            assert cr == fr

    def test_merge_raises_on_node_input(self):
        """Merge raises for Node inputs (same as merge function)."""
        a = Input("a")
        with pytest.raises(ValueError, match="merge is not supported as a Pipeline"):
            Merge()(a)
