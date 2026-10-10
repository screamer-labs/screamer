"""CombineLatest joins one-column streams and rejects wider ones with a clear error."""
import numpy as np
import pytest

import screamer as s


def test_eager_rejects_a_multi_column_input():
    with pytest.raises(ValueError, match="one-column"):
        s.CombineLatest()(np.zeros((10, 4)), np.zeros(10))


def test_pipeline_rejects_a_multi_column_input():
    x, y = s.Input("x"), s.Input("y")
    bars = s.Resample(count=2, agg="ohlc")(x)
    p = s.Pipeline([x, y], [s.CombineLatest()(bars, y)])
    with pytest.raises(ValueError, match="one-column"):
        p(np.arange(20.0), np.arange(20.0))


def test_pipeline_joins_the_columns_split_with_select():
    x, y = s.Input("x"), s.Input("y")
    bars = s.Resample(count=2, agg="ohlc")(x)
    p = s.Pipeline([x, y], [s.CombineLatest()(*[s.Select(k)(bars) for k in range(4)], y)])
    values, _ = p(np.arange(20.0), np.arange(20.0))
    assert values.shape == (20, 5)
    np.testing.assert_array_equal(values[2, :4], [2.0, 3.0, 2.0, 3.0])
