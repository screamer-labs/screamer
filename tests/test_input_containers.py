import numpy as np
import pandas as pd
import pytest

import screamer as s


def test_series_reads_as_values():
    x = np.arange(10.)
    np.testing.assert_array_equal(s.RollingMean(3)(pd.Series(x)), s.RollingMean(3)(x))


def test_dataframe_reads_as_values():
    X = np.random.default_rng(0).random((10, 3))
    out = s.RollingMean(3)(pd.DataFrame(X))
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, s.RollingMean(3)(X))


def test_dataframe_into_multi_input_op():
    df = pd.DataFrame({"a": np.arange(5.), "b": np.ones(5)})
    np.testing.assert_array_equal(s.Add()(df["a"], df["b"]), np.arange(5.) + 1)


def test_series_into_scalar_base_op():
    x = np.arange(6.)
    np.testing.assert_array_equal(s.CumSum()(pd.Series(x)), s.CumSum()(x))


def test_series_into_multi_output_op():
    x = np.random.default_rng(1).random(30)
    a = s.RollingMean(3)(pd.Series(x))
    assert isinstance(a, np.ndarray)


def test_dataframe_into_row_op():
    X = np.random.default_rng(2).random((20, 4))
    np.testing.assert_array_equal(s.CrossRank()(pd.DataFrame(X)), s.CrossRank()(X))


def test_series_into_pipeline_node():
    x = np.arange(6.)
    i = s.Input("x")
    p = s.Pipeline([i], [s.RollingMean(2)(i)])
    np.testing.assert_array_equal(p(pd.Series(x)), p(x))


@pytest.mark.parametrize("bad", ["abc", b"abc", {"a": 1.0}, {1.0, 2.0}, frozenset([1.0])])
@pytest.mark.parametrize("op", [lambda: s.RollingMean(3), lambda: s.Add(), lambda: s.CrossRank(),
                                lambda: s.CumSum(), lambda: s.PortfolioReport()])
def test_unsupported_container_raises_type_error_at_call(op, bad):
    o = op()
    with pytest.raises(TypeError, match="is not a valid input"):
        o(bad, bad) if isinstance(o, s.Add) else o(bad)


@pytest.mark.parametrize("bad", [bytearray(b"ab"), memoryview(b"ab")])
def test_bytearray_and_memoryview_raise(bad):
    with pytest.raises(TypeError, match="is not a valid input"):
        s.RollingMean(3)(bad)


def test_datetime_and_timedelta_pandas_raise():
    dt = pd.Series(pd.date_range("2024-01-01", periods=5))
    with pytest.raises(TypeError, match="datetime or timedelta"):
        s.RollingMean(3)(dt)
    with pytest.raises(TypeError, match="datetime or timedelta"):
        s.RollingMean(3)(pd.DataFrame({"t": dt, "x": 1.0}))
    with pytest.raises(TypeError, match="datetime or timedelta"):
        s.RollingMean(3)(dt.diff())


def test_pandas_module_name_is_exact():
    class Fake:
        __module__ = "pandas_foo"

        def to_numpy(self, dtype=None):
            return np.arange(3.)

    with pytest.raises(Exception):
        s.RollingMean(3)(Fake())
