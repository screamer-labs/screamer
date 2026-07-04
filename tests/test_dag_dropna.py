import numpy as np
import pytest

from screamer import Input, Dag
from screamer.streams import dropna


def _run_modes(dag, feed):
    """Return (batch, stream) results as (keys, values) for a single-output dag."""
    bk, bv = dag(feed)
    sk, sv = dag.stream(feed)
    return (bk, bv), (sk, sv)


def test_dropna_graph_matches_eager_any():
    keys = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    vals = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    x = Input("x")
    dag = Dag(inputs=[x], outputs=[dropna(x)])
    (bk, bv), (sk, sv) = _run_modes(dag, (keys, vals))
    ek, ev = dropna(keys, vals)          # eager oracle
    np.testing.assert_array_equal(bk, ek)
    np.testing.assert_array_equal(bv.reshape(-1), ev)
    np.testing.assert_array_equal(sk, ek)
    np.testing.assert_array_equal(sv.reshape(-1), ev)


def test_dropna_graph_all_dropped():
    keys = np.array([1, 2], dtype=np.int64)
    vals = np.array([np.nan, np.nan])
    x = Input("x")
    dag = Dag(inputs=[x], outputs=[dropna(x)])
    bk, bv = dag((keys, vals))
    assert len(bk) == 0


def test_dropna_graph_none_dropped():
    keys = np.array([1, 2, 3], dtype=np.int64)
    vals = np.array([1.0, 2.0, 3.0])
    x = Input("x")
    dag = Dag(inputs=[x], outputs=[dropna(x)])
    bk, bv = dag((keys, vals))
    np.testing.assert_array_equal(bk, keys)
    np.testing.assert_array_equal(bv.reshape(-1), vals)


def test_dropna_before_functor():
    from screamer import RollingMean
    keys = np.array([1, 2, 3, 4], dtype=np.int64)
    vals = np.array([2.0, np.nan, 4.0, 6.0])
    x = Input("x")
    dag = Dag(inputs=[x], outputs=[RollingMean(2)(dropna(x))])
    bk, bv = dag((keys, vals))
    sk, sv = dag.stream((keys, vals))
    np.testing.assert_array_equal(bk, sk)
    np.testing.assert_array_equal(bv, sv)
    # dropna removes the NaN row, leaving keys [1,3,4]; RollingMean(2) over [2,4,6]
    np.testing.assert_array_equal(bk, [1, 3, 4])


def test_filter_rejected_in_graph():
    x = Input("x")
    from screamer.streams import filter as sfilter
    with pytest.raises(ValueError, match="not supported"):
        sfilter(x, lambda r: r > 0)
