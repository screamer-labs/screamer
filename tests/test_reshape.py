"""reshape names the axes of data with a layout and returns the same container."""
import numpy as np
import pandas as pd
import pytest

import screamer as s


def test_array_cases():
    x = np.arange(12.0).reshape(4, 3)
    assert s.reshape(x, "TA").shape == (4, 3, 1)
    assert s.reshape(x[:, :, None], "TAF->TA").shape == (4, 3)
    assert s.reshape(np.zeros((4, 4)), "TF->TAF").shape == (4, 1, 4)
    np.testing.assert_array_equal(s.reshape(x.T, "AT->TA"), x)
    y = np.arange(24.0).reshape(4, 2, 3)          # (T, F, A)
    np.testing.assert_array_equal(s.reshape(y, "TFA"), y.transpose(0, 2, 1))


def test_array_result_is_a_view_without_reordering():
    x = np.arange(12.0).reshape(4, 3)
    assert np.shares_memory(s.reshape(x, "TA"), x)


def test_values_index_tuple():
    x, idx = np.zeros((4, 3)), np.arange(4)
    v, i = s.reshape((x, idx), "TA")
    assert v.shape == (4, 3, 1) and i is idx


def test_list_and_iterator_of_events():
    rows = [np.arange(3.0) + t for t in range(4)]
    out = s.reshape(rows, "TA")
    assert isinstance(out, list) and out[0].shape == (3, 1)
    gen = s.reshape(iter(rows), "TA")
    assert not isinstance(gen, list) and next(gen).shape == (3, 1)


def test_events_need_time_first():
    with pytest.raises(ValueError, match="must start with T"):
        s.reshape([np.zeros(3)], "AT->TA")


def test_dataframe_flat_to_multiindex():
    df = pd.DataFrame({"BTC": [1.0, 2.0], "ETH": [3.0, 4.0]})
    out = s.reshape(df, "TA")
    assert list(out.columns) == [("BTC", "value"), ("ETH", "value")]
    np.testing.assert_array_equal(out.to_numpy(), df.to_numpy())


def test_dataframe_multiindex_reorder():
    cols = pd.MultiIndex.from_product([["open", "close"], ["BTC", "ETH"]])   # (field, asset)
    df = pd.DataFrame(np.arange(8.0).reshape(2, 4), columns=cols)
    out = s.reshape(df, "TFA")
    assert list(out.columns) == [("BTC", "open"), ("BTC", "close"), ("ETH", "open"), ("ETH", "close")]
    assert out[("ETH", "close")].tolist() == df[("close", "ETH")].tolist()


def test_dataframe_multiindex_must_be_complete():
    cols = pd.MultiIndex.from_tuples([("BTC", "close"), ("ETH", "open")])
    with pytest.raises(ValueError, match="every"):
        s.reshape(pd.DataFrame(np.zeros((2, 2)), columns=cols), "TAF")


def test_series():
    out = s.reshape(pd.Series([1.0, 2.0]), "T")
    assert isinstance(out, pd.DataFrame) and list(out.columns) == ["value"]


def test_from_pandas_layout():
    df = pd.DataFrame({"BTC": [1.0, 2.0], "ETH": [3.0, 4.0]}, index=[10, 20])
    values, index = s.from_pandas(df, layout="TA")
    assert values.shape == (2, 2, 1) and list(index) == [10, 20]
    flat, _ = s.from_pandas(df)
    assert flat.shape == (2, 2)


def test_from_pandas_reads_two_level_columns_as_assets_and_fields():
    cols = pd.MultiIndex.from_product([["BTC", "ETH"], ["open", "close"]])
    values, _ = s.from_pandas(pd.DataFrame(np.arange(8.0).reshape(2, 4), columns=cols))
    assert values.shape == (2, 2, 2)
    assert values[1, 1, 0] == 6.0    # ETH open at t=1


def test_unsupported_container():
    with pytest.raises(TypeError):
        s.reshape("abc", "T")
