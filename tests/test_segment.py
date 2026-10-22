"""Segment(op): op run on each run of equal labels, reset where the label changes.

The reference is pandas `groupby(runs).transform(...)`, with `runs` the runs
of the forward-filled label (a NaN label keeps the current segment) and no
group for the rows before the first label.
"""
import numpy as np
import pandas as pd
import pytest

import screamer
from screamer import (BollingerBands, CumSum, EwMean, ExpandingMean, Pipeline, RollingCorr,
                      RollingStd, Segment)
from screamer.dag_io import from_json, to_json
from tests.regime_helpers import assert_bitwise


def _labels(T=600, seed=3, nan=True):
    """Runs of 1 to 30 rows over 4 values (a value returns), leading NaNs and
    NaN labels inside segments."""
    rng = np.random.default_rng(seed)
    lab = np.repeat(rng.integers(0, 4, T).astype(float), rng.integers(1, 31, T))[:T]
    if nan:
        lab[:3] = np.nan
        lab[[40, 41, 200, 333]] = np.nan
    return lab


def _runs(label):
    """One id per run of equal forward-filled labels; NaN before the first label."""
    s = pd.Series(label).ffill()
    runs = (s != s.shift()).cumsum().astype(float)
    runs[s.isna()] = np.nan
    return runs


def _x(T=600, seed=5):
    return np.random.default_rng(seed).normal(size=T)


PANDAS_CASES = [
    ("CumSum", lambda: CumSum(), lambda g: g.cumsum()),
    ("ExpandingMean", lambda: ExpandingMean(), lambda g: g.expanding().mean()),
    ("RollingStd", lambda: RollingStd(10), lambda g: g.rolling(10).std()),
    ("EwMean", lambda: EwMean(span=8.0), lambda g: g.ewm(span=8.0, adjust=True).mean()),
]


@pytest.mark.parametrize("name,make,ref", PANDAS_CASES, ids=[c[0] for c in PANDAS_CASES])
def test_matches_pandas_groupby_transform(name, make, ref):
    label, x = _labels(), _x()
    expected = pd.Series(x).groupby(_runs(label)).transform(ref).to_numpy()
    got = Segment(make())(label, x)
    assert got.shape == x.shape
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_two_input_op_matches_pandas():
    label, x, y = _labels(), _x(seed=5), _x(seed=6)
    df = pd.DataFrame({"x": x, "y": y})
    expected = np.full(len(x), np.nan)
    for _, idx in df.groupby(_runs(label)).groups.items():
        g = df.loc[idx]
        expected[idx] = g["x"].rolling(12).corr(g["y"]).to_numpy()
    got = Segment(RollingCorr(12))(label, x, y)
    np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-12, equal_nan=True)


def test_multi_output_op_matches_a_fresh_op_per_segment():
    label, x = _labels(), _x()
    expected = np.full((len(x), 3), np.nan)
    for _, idx in pd.Series(x).groupby(_runs(label)).groups.items():
        expected[idx] = BollingerBands(5)(x[idx])
    got = Segment(BollingerBands(5))(label, x)
    assert got.shape == (len(x), 3)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_nan_labels():
    lab = np.array([np.nan, np.nan, 1, 1, np.nan, 1, 2, np.nan, 2, 1])
    x = np.arange(1.0, 11.0)
    got = Segment(CumSum())(lab, x)
    # Leading NaN labels: no segment. A NaN label keeps the segment; a label
    # that returns (1 after 2) starts a new one.
    expected = [np.nan, np.nan, 3, 7, 12, 18, 7, 15, 24, 10]
    np.testing.assert_array_equal(got, expected)


def test_returning_label_starts_a_new_segment():
    lab = np.array([1.0, 1, 2, 1, 1])
    np.testing.assert_array_equal(Segment(CumSum())(lab, np.ones(5)), [1, 2, 1, 1, 2])


def test_data_nan_is_the_ops_business():
    lab = np.ones(5)
    x = np.array([1.0, np.nan, 2, 3, np.nan])
    assert_bitwise(Segment(CumSum())(lab, x), CumSum()(x))


@pytest.mark.parametrize("make,n_in", [(lambda: CumSum(), 1), (lambda: RollingStd(10), 1),
                                       (lambda: RollingCorr(12), 2), (lambda: BollingerBands(5), 1)])
def test_batch_equals_per_event(make, n_in):
    label = _labels()
    inputs = [_x(seed=5 + i) for i in range(n_in)]
    batch = Segment(make())(label, *inputs)
    op = Segment(make())
    events = np.array([op(float(label[t]), *[float(a[t]) for a in inputs])
                       for t in range(len(label))]).reshape(batch.shape)
    assert_bitwise(events, batch)


def test_columns_are_independent():
    label = np.column_stack([_labels(seed=3), _labels(seed=4)])
    x = np.column_stack([_x(seed=5), _x(seed=6)])
    got = Segment(RollingStd(7))(label, x)
    for c in range(2):
        assert_bitwise(got[:, c], Segment(RollingStd(7))(label[:, c], x[:, c]))


def test_regimes_agree():
    label, x = _labels(), _x()
    eager = Segment(RollingStd(10))(label, x)
    row = screamer.Input("row")
    pipeline = Pipeline(inputs=[row], outputs=[Segment(RollingStd(10))(row)])
    index = np.arange(len(x))
    graph, _ = pipeline((np.column_stack([label, x]), index))
    lazy = np.array([r[0] for r in pipeline(((
        (float(label[t]), float(x[t])), t) for t in range(len(x))))])
    stream = np.array(list(Segment(RollingStd(10))(iter(label), iter(x))))
    assert_bitwise(graph, eager)
    assert_bitwise(lazy, eager)
    assert_bitwise(stream, eager)


def test_reset_restores_initial_state():
    label, x = _labels(), _x()
    seg = Segment(EwMean(span=8.0))
    first = seg(label, x)
    for t in range(50):
        seg(float(label[t]), float(x[t]))
    seg.reset()
    assert_bitwise(np.array([seg(float(label[t]), float(x[t])) for t in range(len(x))]), first)


def test_holds_its_own_copy_of_op():
    inner = CumSum()
    seg = Segment(inner)
    for _ in range(5):
        seg(1.0, 1.0)
    assert inner(1.0) == 1.0              # the caller's op was never run
    other = Segment(inner)
    assert other(1.0, 1.0) == 1.0         # nor shared between wrappers
    assert seg(1.0, 1.0) == 6.0


def test_nested_segments():
    outer = np.repeat([1.0, 2.0], 6)
    inner = np.tile(np.repeat([1.0, 2.0], 3), 2)
    got = Segment(Segment(CumSum()))(outer, inner, np.ones(12))
    np.testing.assert_array_equal(got, [1, 2, 3] * 4)


def test_repr_and_pipeline_json_round_trip():
    seg = Segment(RollingStd(window_size=10))
    assert repr(seg) == "Segment(op=RollingStd(window_size=10))"
    row = screamer.Input("row")
    pipeline = Pipeline(inputs=[row], outputs=[seg(row)])
    label, x = _labels(), _x()
    data = (np.column_stack([label, x]), np.arange(len(x)))
    again = from_json(to_json(pipeline))
    assert_bitwise(again(data)[0], pipeline(data)[0])


def test_rejects_what_is_not_an_operator():
    with pytest.raises(TypeError, match="Segment: op must be an operator"):
        Segment(3)
    with pytest.raises(ValueError, match="cross-sectional"):
        Segment(screamer.CrossRank())
