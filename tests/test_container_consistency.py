"""3.0.0 container consistency, tracker rows A7 and A8.

A7: a list of (K, 4) engine rows into PortfolioReport returns a list of result
rows, like every other operator (rule 3). A tuple keeps the 2.8.0 array (D9).

A8: a multi-input operator returns an array for any mix of lists and arrays, in
either argument order. All lists still return a list.
"""
import numpy as np
import pytest

import screamer as s

T = 8
rng = np.random.default_rng(11)
x = rng.normal(size=T).cumsum()
y = rng.normal(size=T).cumsum()
z = rng.normal(size=T).cumsum()


# ------------------------------------------------------------------ A7

def _engine(T=5, K=3):
    r = np.random.default_rng(3)
    return r.normal(size=(T, K, 4)).cumsum(axis=0)


def test_portfolio_report_list_of_rows_gives_list_of_rows():
    engine = _engine()
    want = s.PortfolioReport()(engine)
    got = s.PortfolioReport()([row for row in engine])
    assert isinstance(got, list) and len(got) == len(engine)
    for g, w in zip(got, want):
        assert isinstance(g, np.ndarray) and g.shape == (6,)
        np.testing.assert_array_equal(g, w)


def test_portfolio_report_nested_python_lists_give_list_of_rows():
    engine = _engine()
    got = s.PortfolioReport()(engine.tolist())
    assert isinstance(got, list)
    np.testing.assert_array_equal(np.array(got), s.PortfolioReport()(engine))


def test_portfolio_report_tuple_and_array_keep_the_array():
    engine = _engine()
    want = s.PortfolioReport()(engine)
    got = s.PortfolioReport()(tuple(engine))
    assert isinstance(got, np.ndarray)
    np.testing.assert_array_equal(got, want)


def test_portfolio_report_stream_unchanged():
    engine = _engine()
    want = s.PortfolioReport()(engine)
    got = list(s.PortfolioReport()(iter(engine)))
    np.testing.assert_array_equal(np.array(got), want)


def test_portfolio_report_one_event_list_stays_one_row():
    row = _engine()[0]
    got = s.PortfolioReport()(row.tolist())
    _eq(got, s.PortfolioReport()(row))
    assert isinstance(got, tuple) and len(got) == 6


# ------------------------------------------------------------------ A8

TWO = [s.RollingCorr(3), s.Add(), s.RollingBeta(4)]
THREE = [s.Where(), s.ATR(3), s.Stoch(3)]
MULTI_OUT = [s.RollingLinearRegression(4)]


def _eq(a, b):
    np.testing.assert_array_equal(np.asarray(a, dtype=float), np.asarray(b, dtype=float))


@pytest.mark.parametrize("op", TWO + MULTI_OUT, ids=lambda o: type(o).__name__)
@pytest.mark.parametrize("seq", [list, tuple], ids=["list", "tuple"])
def test_two_inputs_mixed_either_order_give_the_array_result(op, seq):
    want = op(x, y)
    for args in [(seq(x), y), (x, seq(y))]:
        got = op(*args)
        assert isinstance(got, np.ndarray)
        assert got.shape == want.shape
        _eq(got, want)


@pytest.mark.parametrize("op", THREE, ids=lambda o: type(o).__name__)
def test_three_inputs_any_mix_gives_the_array_result(op):
    a, b, c = np.abs(x) + 2, -np.abs(y) - 2, z
    want = op(a, b, c)
    for mask in range(1, 7):
        args = [list(v) if mask >> i & 1 else v for i, v in enumerate((a, b, c))]
        got = op(*args)
        assert isinstance(got, np.ndarray), mask
        _eq(got, want)


def test_one_input_multi_output_list_still_a_list():
    out = s.RollingMinMax(3)(list(x))
    assert isinstance(out, list)


@pytest.mark.parametrize("op", TWO + MULTI_OUT, ids=lambda o: type(o).__name__)
def test_all_lists_still_give_a_list(op):
    got = op(list(x), list(y))
    assert isinstance(got, list)
    _eq(got, op(x, y))


def test_list_broadcasts_against_asset_columns():
    X = np.column_stack([x, y, z])
    want = s.RollingCorr(3)(X, z)
    for args in [(X, list(z)), (list(z), X)]:
        got = s.RollingCorr(3)(*args)
        assert isinstance(got, np.ndarray) and got.shape == (T, 3)
    _eq(s.RollingCorr(3)(X, list(z)), want)
    _eq(s.RollingCorr(3)(list(z), X), s.RollingCorr(3)(z, X))


@pytest.mark.parametrize("args", [(list(x[:5]), y), (x, list(y[:5]))], ids=["list-short", "array-short"])
def test_mixed_lengths_still_raise_value_error(args):
    with pytest.raises(ValueError):
        s.RollingCorr(3)(*args)


def test_mixed_list_with_a_non_number_raises_type_error():
    bad = list(y)
    bad[2] = "a"
    with pytest.raises(TypeError, match="RollingCorr"):
        s.RollingCorr(3)(x, bad)
    with pytest.raises(TypeError, match="RollingCorr"):
        s.RollingCorr(3)(bad, x)


def test_mixed_call_leaves_the_operator_reset():
    op = s.RollingCorr(3)
    first = op(list(x), y)
    _eq(op(x, list(y)), first)


# ------------------------------------------- A9 and A10 (Python side of JS parity)
# The JS tests in js/test/container_consistency.test.ts pin the same values.

def test_pipeline_none_value_is_nan():
    i = s.Input("a")
    values, _ = s.Pipeline([i], [s.RollingMean(1)(i)])([1.0, None, 3.0])
    _eq(values, [1.0, np.nan, 3.0])


def test_list_of_number_events_gives_a_list():
    assert s.Add()([[1, 2], [3, 4]]) == [3.0, 7.0]
    out = s.RollingLinearRegression(4)([(1.0, 2.0), (2.0, 3.0)])
    assert isinstance(out, list) and all(isinstance(r, tuple) and len(r) == 4 for r in out)


def test_list_of_asset_events_into_one_input_op():
    got = s.RollingMean(2)([np.zeros(3), np.ones(3)])
    assert isinstance(got, list) and len(got) == 2
    _eq(got[0], [np.nan] * 3)
    _eq(got[1], [0.5] * 3)
    rows = [np.array([1.0, 2, 3]), np.array([4.0, 5, 6]), np.array([7.0, 9, 8])]
    listed = s.RollingMean(2)(rows)
    _eq(listed[1], [2.5, 3.5, 4.5])
    _eq(listed[2], [5.5, 7, 7])
    mm = s.RollingMinMax(2)(rows)
    assert all(r.shape == (3, 2) for r in mm)
    _eq(mm[2].ravel(), [4, 7, 5, 9, 6, 8])


def test_list_of_k_n_asset_events_into_n_input_op():
    evs = [np.array([[1.0, 2], [3, 4], [5, 6]]), np.array([[2.0, 2], [4, 4], [9, 6]]),
           np.array([[3.0, 1], [2, 5], [5, 7]])]
    got = s.RollingCorr(2)(evs)
    assert isinstance(got, list) and all(r.shape == (3,) for r in got)
    for g, w in zip(got, s.RollingCorr(2)(iter(evs))):
        _eq(g, w)
    _eq(got[2], [-1, -1, -1])
    one = s.RollingMean(2)([np.array([[1.0, 2], [3, 4]]), np.array([[3.0, 4], [5, 6]])])
    assert one[1].shape == (2, 2)
    _eq(one[1].ravel(), [2, 3, 4, 5])
