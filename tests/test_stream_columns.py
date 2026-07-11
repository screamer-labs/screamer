"""Tests for OHLC positional-column convention after Stream removal.

Resample with agg='ohlc'/'ohlcv'/'ohlcv2' returns a plain (values, index)
tuple. Columns are positional, in documented order:
  ohlc:  col0=open, col1=high, col2=low, col3=close
  ohlcv: col0=open, col1=high, col2=low, col3=close, col4=volume
  ohlcv2: col0=open, col1=high, col2=low, col3=close, col4=buy_vol, col5=sell_vol
"""
import numpy as np
import pytest
from screamer.streams import resample, Resample


def test_ohlc_returns_tuple():
    x = np.arange(20.0)
    idx = np.arange(20, dtype=np.int64)
    result = Resample(freq=5, agg="ohlc")(x, idx)
    assert isinstance(result, tuple) and len(result) == 2
    values, index = result
    assert isinstance(values, np.ndarray)
    assert values.shape[1] == 4


def test_ohlc_column_order():
    """open=col0, high=col1, low=col2, close=col3."""
    x = np.arange(20.0)
    idx = np.arange(20, dtype=np.int64)
    values, index = Resample(freq=5, agg="ohlc")(x, idx)
    # bar 0: ticks 0..4, open=0, high=4, low=0, close=4
    np.testing.assert_allclose(values[0, 0], 0.0)   # open
    np.testing.assert_allclose(values[0, 1], 4.0)   # high
    np.testing.assert_allclose(values[0, 2], 0.0)   # low
    np.testing.assert_allclose(values[0, 3], 4.0)   # close


def test_ohlc_no_columns_attribute():
    """The result is a plain tuple; there is no .columns attribute."""
    x = np.arange(10.0)
    idx = np.arange(10, dtype=np.int64)
    result = Resample(freq=5, agg="ohlc")(x, idx)
    assert not hasattr(result, "columns")


def test_scalar_agg_returns_tuple():
    x = np.arange(20.0)
    idx = np.arange(20, dtype=np.int64)
    result = Resample(freq=5, agg="sum")(x, idx)
    assert isinstance(result, tuple) and len(result) == 2
    values, index = result
    assert values.ndim == 1


def test_ohlcv_returns_tuple():
    price = np.arange(20.0)
    vol = np.ones(20)
    idx = np.arange(20, dtype=np.int64)
    result = Resample(freq=5, agg="ohlcv")(np.column_stack([price, vol]), idx)
    assert isinstance(result, tuple) and len(result) == 2
    values, index = result
    assert values.shape[1] == 5


def test_ohlcv2_returns_tuple():
    price = np.arange(20.0)
    svol = np.where(np.arange(20) % 2, 1.0, -1.0)
    idx = np.arange(20, dtype=np.int64)
    result = Resample(freq=5, agg="ohlcv2")(np.column_stack([price, svol]), idx)
    assert isinstance(result, tuple) and len(result) == 2
    values, index = result
    assert values.shape[1] == 6


def test_ohlcv_positional_columns():
    """ohlcv: open=0, high=1, low=2, close=3, volume=4."""
    price = np.arange(20.0)
    vol = np.arange(20.0) * 0.5
    idx = np.arange(20, dtype=np.int64)
    values, index = Resample(freq=5, agg="ohlcv")(np.column_stack([price, vol]), idx)
    # Access volume column positionally (col 4)
    volume_col = values[:, 4]
    ref_sum = np.array([sum(vol[i*5:(i+1)*5]) for i in range(4)])
    np.testing.assert_allclose(volume_col, ref_sum)


def test_tuple_unpacking():
    x = np.arange(10.0)
    idx = np.arange(10, dtype=np.int64)
    values, index = Resample(freq=5, agg="ohlc")(x, idx)
    assert index is not None
    assert values.shape == (2, 4)
