"""Tests for resample(agg="ohlcv" | "ohlcv2") - Task 7 multi-input bar aggs.

ohlcv : 2-column [price, volume] -> 5 columns [open, high, low, close, volume]
ohlcv2: 2-column [price, signed_volume] -> 6 columns
        [open, high, low, close, buy_vol, sell_vol]

All numeric compute stays in C++ (OHLC via the existing ohlc reducer, volume
via sum, buy/sell via PosPart/NegPart + sum). Python only orchestrates.
"""
import numpy as np
import pytest
from screamer import PosPart, NegPart
from screamer.streams import Resample


# ---------------------------------------------------------------------------
# ohlcv
# ---------------------------------------------------------------------------

def test_ohlcv_columns():
    """Resample(..., agg='ohlcv') returns a tuple with 5 positional columns."""
    price = np.arange(20.0)
    vol = np.ones(20)
    idx = np.arange(20, dtype=np.int64)
    bars = Resample(freq=5, agg="ohlcv")(np.column_stack([price, vol]), idx)
    assert isinstance(bars, tuple) and isinstance(bars[0], np.ndarray)
    assert bars[0].shape[1] == 5  # open(0),high(1),low(2),close(3),volume(4)


def test_ohlcv_ohlc_block_matches_ohlc_agg():
    """The OHLC block (first 4 columns) equals resample(price, agg='ohlc')."""
    price = np.arange(20.0)
    vol = np.ones(20)
    idx = np.arange(20, dtype=np.int64)
    bars = Resample(freq=5, agg="ohlcv")(np.column_stack([price, vol]), idx)
    ohlc, _ = Resample(freq=5, agg="ohlc")(price, idx)
    np.testing.assert_allclose(bars[0][:, :4], ohlc)


def test_ohlcv_volume_column_matches_sum_agg():
    """The volume column equals resample(volume, agg='sum')."""
    price = np.arange(20.0)
    vol = np.arange(20.0) * 0.5
    idx = np.arange(20, dtype=np.int64)
    bars = Resample(freq=5, agg="ohlcv")(np.column_stack([price, vol]), idx)
    vol_sum, _ = Resample(freq=5, agg="sum")(vol, idx)
    np.testing.assert_allclose(bars[0][:, 4], vol_sum)


# ---------------------------------------------------------------------------
# ohlcv2
# ---------------------------------------------------------------------------

def test_ohlcv2_columns():
    """Resample(..., agg='ohlcv2') returns a tuple with 6 positional columns."""
    price = np.arange(20.0)
    svol = np.where(np.arange(20) % 2, 1.0, -1.0)
    idx = np.arange(20, dtype=np.int64)
    bars = Resample(freq=5, agg="ohlcv2")(np.column_stack([price, svol]), idx)
    assert isinstance(bars, tuple) and isinstance(bars[0], np.ndarray)
    assert bars[0].shape[1] == 6  # open(0),high(1),low(2),close(3),buy_vol(4),sell_vol(5)


def test_ohlcv2_ohlc_block_matches_ohlc_agg():
    """The OHLC block (first 4 columns) equals resample(price, agg='ohlc')."""
    price = np.arange(20.0)
    svol = np.where(np.arange(20) % 2, 1.0, -1.0)
    idx = np.arange(20, dtype=np.int64)
    bars = Resample(freq=5, agg="ohlcv2")(np.column_stack([price, svol]), idx)
    ohlc, _ = Resample(freq=5, agg="ohlc")(price, idx)
    np.testing.assert_allclose(bars[0][:, :4], ohlc)


def test_ohlcv2_buy_vol_matches_pospart_sum():
    """buy_vol equals resample(PosPart(signed_vol), agg='sum')."""
    price = np.arange(20.0)
    svol = np.where(np.arange(20) % 2, 1.0, -1.0)
    idx = np.arange(20, dtype=np.int64)
    bars = Resample(freq=5, agg="ohlcv2")(np.column_stack([price, svol]), idx)
    buy_ref, _ = Resample(freq=5, agg="sum")(np.asarray(PosPart()(svol)), idx)
    np.testing.assert_allclose(bars[0][:, 4], buy_ref)


def test_ohlcv2_sell_vol_matches_negpart_sum():
    """sell_vol equals resample(NegPart(signed_vol), agg='sum')."""
    price = np.arange(20.0)
    svol = np.where(np.arange(20) % 2, 1.0, -1.0)
    idx = np.arange(20, dtype=np.int64)
    bars = Resample(freq=5, agg="ohlcv2")(np.column_stack([price, svol]), idx)
    sell_ref, _ = Resample(freq=5, agg="sum")(np.asarray(NegPart()(svol)), idx)
    np.testing.assert_allclose(bars[0][:, 5], sell_ref)


def test_ohlcv2_matches_composition():
    """Full composition check: ohlcv2 == ohlc + PosPart/NegPart sums."""
    price = np.arange(20.0)
    vol = np.where(np.arange(20) % 2, 1.0, -1.0)
    idx = np.arange(20, dtype=np.int64)
    bars = Resample(freq=5, agg="ohlcv2")(np.column_stack([price, vol]), idx)
    assert bars[0].shape[1] == 6  # open(0),high(1),low(2),close(3),buy_vol(4),sell_vol(5)
    o, _ = Resample(freq=5, agg="ohlc")(price, idx)
    np.testing.assert_allclose(bars[0][:, :4], o)
    buy, _ = Resample(freq=5, agg="sum")(np.asarray(PosPart()(vol)), idx)
    np.testing.assert_allclose(bars[0][:, 4], buy)


# ---------------------------------------------------------------------------
# Hand-computed expected-value case (2 bars, values written out by hand)
# ---------------------------------------------------------------------------

def test_ohlcv_hand_computed():
    """Small input, 2 bars, values verified by hand.

    price  = [0, 1, 2, 3, 4,  5, 6, 7,  8, 9]
    volume = [1, 2, 3, 4, 5,  6, 7, 8,  9, 10]
    every  = 5

    Bar 0  (idx 0..4): open=0, high=4, low=0, close=4, volume=1+2+3+4+5=15
    Bar 1  (idx 5..9): open=5, high=9, low=5, close=9, volume=6+7+8+9+10=40
    """
    price = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    vol   = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    idx   = np.arange(10, dtype=np.int64)
    bars  = Resample(freq=5, agg="ohlcv")(np.column_stack([price, vol]), idx)

    assert bars[0].shape == (2, 5)
    expected = np.array([
        [0.0, 4.0, 0.0, 4.0, 15.0],
        [5.0, 9.0, 5.0, 9.0, 40.0],
    ])
    np.testing.assert_allclose(bars[0], expected)


def test_ohlcv2_hand_computed():
    """Small input, 2 bars, values verified by hand.

    price       = [0, 1, 2, 3, 4,  5, 6, 7,  8, 9]
    signed_vol  = [1,-1, 1,-1, 1,  1, 1,-1,  1,-1]   (alternating)
    every       = 5

    Bar 0  (idx 0..4): open=0, high=4, low=0, close=4
             PosPart(vol) = [1, 0, 1, 0, 1] -> buy_vol  = 3
             NegPart(vol) = [0, 1, 0, 1, 0] -> sell_vol = 2
    Bar 1  (idx 5..9): open=5, high=9, low=5, close=9
             PosPart(vol) = [1, 1, 0, 1, 0] -> buy_vol  = 3
             NegPart(vol) = [0, 0, 1, 0, 1] -> sell_vol = 2
    """
    price = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    svol  = np.array([1.0,-1.0, 1.0,-1.0, 1.0, 1.0, 1.0,-1.0, 1.0,-1.0])
    idx   = np.arange(10, dtype=np.int64)
    bars  = Resample(freq=5, agg="ohlcv2")(np.column_stack([price, svol]), idx)

    assert bars[0].shape == (2, 6)
    expected = np.array([
        [0.0, 4.0, 0.0, 4.0, 3.0, 2.0],
        [5.0, 9.0, 5.0, 9.0, 3.0, 2.0],
    ])
    np.testing.assert_allclose(bars[0], expected)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_ohlcv_rejects_1d_input():
    """ohlcv requires a 2-column input; a 1-D array raises ValueError."""
    x = np.arange(10.0)
    idx = np.arange(10, dtype=np.int64)
    with pytest.raises(ValueError, match="2 columns"):
        Resample(freq=5, agg="ohlcv")(x, idx)


def test_ohlcv2_rejects_1d_input():
    """ohlcv2 requires a 2-column input; a 1-D array raises ValueError."""
    x = np.arange(10.0)
    idx = np.arange(10, dtype=np.int64)
    with pytest.raises(ValueError, match="2 columns"):
        Resample(freq=5, agg="ohlcv2")(x, idx)


def test_ohlcv_rejects_wrong_column_count():
    """ohlcv requires exactly 2 columns; 3 columns raises ValueError."""
    x = np.column_stack([np.arange(10.0)] * 3)
    idx = np.arange(10, dtype=np.int64)
    with pytest.raises(ValueError, match="2 columns"):
        Resample(freq=5, agg="ohlcv")(x, idx)


def test_ohlcv_bar_index_is_real_array():
    """ohlcv returns a tuple whose index (element 1) is a real (non-None) array."""
    price = np.arange(10.0)
    vol = np.ones(10)
    idx = np.arange(10, dtype=np.int64)
    bars = Resample(freq=5, agg="ohlcv")(np.column_stack([price, vol]), idx)
    assert bars[1] is not None
    assert len(bars[1]) == 2


def test_ohlcv_tuple_input():
    """ohlcv works when the input is a (values, index) tuple (not a raw array)."""
    price = np.arange(10.0)
    vol = np.ones(10)
    idx = np.arange(10, dtype=np.int64)
    s = (np.column_stack([price, vol]), idx)
    bars = Resample(freq=5, agg="ohlcv")(s)
    assert isinstance(bars, tuple) and isinstance(bars[0], np.ndarray)
    assert bars[0].shape[1] == 5  # open(0),high(1),low(2),close(3),volume(4)
    assert bars[0].shape == (2, 5)
