"""End-of-input flush coalescing for the graph combine_latest node.

Regression: at end-of-input, each input port's flush used to trigger a SEPARATE
emit, so a final index shared by all inputs was emitted once per input port
(the final row was duplicated). The mid-stream path already coalesces by index;
the flush path must too.
"""
import numpy as np
import pytest

from screamer.dag import Input, Dag
from screamer.streams import resample, Resample, combine_latest
from screamer import ExpandingMax, ExpandingMin


def test_repro_no_duplicate_final_row():
    t = np.arange(200, dtype=np.int64)
    price = 100 + np.cumsum(np.random.default_rng(7).normal(size=200))
    p = Input("price")
    # node-mode span: use every= (Resample(freq=W)(node) resolves to count mode)
    dag = Dag([p], [combine_latest(resample(p, every=40, agg=ExpandingMax()),
                                   resample(p, every=40, agg=ExpandingMin()))])
    values, index = dag((price, t))

    # Exactly one row per distinct index, no duplicated final index.
    np.testing.assert_array_equal(index, [0, 40, 80, 120, 160])
    assert values.shape == (5, 2)


def test_columns_equal_single_column_resample():
    t = np.arange(200, dtype=np.int64)
    price = 100 + np.cumsum(np.random.default_rng(7).normal(size=200))
    p = Input("price")

    dag = Dag([p], [combine_latest(resample(p, every=40, agg=ExpandingMax()),
                                   resample(p, every=40, agg=ExpandingMin()))])
    values, index = dag((price, t))

    # Each combined column must equal the corresponding single-stream resample.
    dag_max = Dag([p], [resample(p, every=40, agg=ExpandingMax())])
    vmax, imax = dag_max((price, t))
    dag_min = Dag([p], [resample(p, every=40, agg=ExpandingMin())])
    vmin, imin = dag_min((price, t))

    np.testing.assert_array_equal(index, imax)
    np.testing.assert_array_equal(index, imin)
    np.testing.assert_allclose(values[:, 0], np.asarray(vmax).ravel())
    np.testing.assert_allclose(values[:, 1], np.asarray(vmin).ravel())


def test_shared_final_index_matches_eager_combine_latest():
    """Two graph streams that share their final index emit exactly the distinct
    indices, with values equal to the eager combine_latest on the same data."""
    from screamer.streams import combine_latest as eager_combine_latest

    t = np.arange(200, dtype=np.int64)
    price = 100 + np.cumsum(np.random.default_rng(11).normal(size=200))
    p = Input("price")

    smax = resample(p, every=40, agg=ExpandingMax())
    smin = resample(p, every=40, agg=ExpandingMin())
    dag = Dag([p], [combine_latest(smax, smin)])
    values, index = dag((price, t))

    # Materialize each stream separately to feed the eager combine_latest.
    vmax, imax = Dag([p], [smax])((price, t))
    vmin, imin = Dag([p], [smin])((price, t))

    ev, ei = eager_combine_latest(
        np.asarray(vmax).ravel(), np.asarray(vmin).ravel(),
        index=[np.asarray(imax), np.asarray(imin)])

    # No duplicate final index: exactly len(distinct indices) rows.
    assert len(index) == len(np.unique(np.concatenate([imax, imin])))
    assert index[-1] != index[-2]

    np.testing.assert_array_equal(index, ei)
    np.testing.assert_allclose(values, ev)
