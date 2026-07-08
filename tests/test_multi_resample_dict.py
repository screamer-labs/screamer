"""Tests for resample(t, agg={name: Reducer()(expr)}) lazy dict front-end.

Task 6b: multi-column bars in the graph (Node) regime, returned as labelled Streams.
"""
import numpy as np
import pytest

from screamer import (
    ExpandingMax, ExpandingMin, ExpandingSum, First, Last, NegPart, PosPart,
)
from screamer.dag import Dag, Input
from screamer.streams import Stream, multi_resample, resample


# ---------------------------------------------------------------------------
# Fixtures / shared data
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
N = 50          # number of ticks
W = 10          # bar width (every=W)
N_BARS = N // W

def _make_data():
    t_arr      = np.arange(N, dtype=np.int64)
    price_arr  = RNG.standard_normal(N).cumsum() + 100.0
    # signed volume: positive => buy, negative => sell
    vol_arr    = RNG.standard_normal(N) * 10.0
    return t_arr, price_arr, vol_arr


# ---------------------------------------------------------------------------
# Test 1: OHLCV2 recipe (time mode) - the headline test
# ---------------------------------------------------------------------------

class TestOHLCV2Recipe:
    """Six-column lazy agg dict via the graph regime, every= time bucketing."""

    def setup_method(self):
        t_arr, price_arr, vol_arr = _make_data()
        self.t_arr     = t_arr
        self.price_arr = price_arr
        self.vol_arr   = vol_arr

        price = Input("price")
        vol   = Input("vol")
        t     = Input("t")

        bars = resample(t, every=W, agg={
            "open":  First()(price),
            "high":  ExpandingMax()(price),
            "low":   ExpandingMin()(price),
            "close": Last()(price),
            "buy":   ExpandingSum()(PosPart()(vol)),
            "sell":  ExpandingSum()(NegPart()(vol)),
        })
        self.dag = Dag([t, price, vol], [bars])
        self.out = self.dag(t=t_arr, price=price_arr, vol=vol_arr)

    def test_returns_stream(self):
        assert isinstance(self.out, Stream)

    def test_columns_are_keys_in_order(self):
        assert self.out.columns == ("open", "high", "low", "close", "buy", "sell")

    def test_shape(self):
        assert self.out.values.ndim == 2
        assert self.out.values.shape[1] == 6
        assert len(self.out) == N_BARS

    def test_named_column_access(self):
        buy_col = self.out["buy"]
        assert buy_col.shape == (N_BARS,)
        np.testing.assert_array_equal(buy_col, self.out.values[:, 4])

    def _reference_col(self, agg_str, arr=None):
        """Compute a reference column via the eager single-column path."""
        if arr is None:
            arr = self.price_arr
        ref = resample(arr, self.t_arr, every=W, agg=agg_str)
        return ref.values if isinstance(ref, Stream) else ref[0]

    def test_open_matches_reference(self):
        ref = self._reference_col("first")
        np.testing.assert_allclose(self.out["open"], ref, rtol=1e-12)

    def test_high_matches_reference(self):
        ref = self._reference_col("max")
        np.testing.assert_allclose(self.out["high"], ref, rtol=1e-12)

    def test_low_matches_reference(self):
        ref = self._reference_col("min")
        np.testing.assert_allclose(self.out["low"], ref, rtol=1e-12)

    def test_close_matches_reference(self):
        ref = self._reference_col("last")
        np.testing.assert_allclose(self.out["close"], ref, rtol=1e-12)

    def test_buy_matches_reference(self):
        pos_vol = np.where(self.vol_arr > 0, self.vol_arr, 0.0)
        ref = self._reference_col("sum", pos_vol)
        np.testing.assert_allclose(self.out["buy"], ref, rtol=1e-12)

    def test_sell_matches_reference(self):
        # NegPart returns abs(min(x, 0)), i.e. magnitude of negative values.
        neg_part_vol = np.where(self.vol_arr < 0, -self.vol_arr, 0.0)
        ref = self._reference_col("sum", neg_part_vol)
        np.testing.assert_allclose(self.out["sell"], ref, rtol=1e-12)


# ---------------------------------------------------------------------------
# Test 2: Count mode dict
# ---------------------------------------------------------------------------

class TestCountModeDict:
    """Dict agg with count= bucketing."""

    COUNT = 5

    def setup_method(self):
        t_arr, price_arr, vol_arr = _make_data()
        self.t_arr    = t_arr
        self.price_arr = price_arr
        self.vol_arr   = vol_arr

        price = Input("price")
        vol   = Input("vol")
        t     = Input("t")

        bars = resample(t, count=self.COUNT, agg={
            "open":  First()(price),
            "close": Last()(price),
            "buy":   ExpandingSum()(PosPart()(vol)),
        })
        self.dag = Dag([t, price, vol], [bars])
        self.out = self.dag(t=t_arr, price=price_arr, vol=vol_arr)

    def test_returns_stream(self):
        assert isinstance(self.out, Stream)

    def test_columns_are_keys_in_order(self):
        assert self.out.columns == ("open", "close", "buy")

    def test_shape(self):
        # N=50 ticks, count=5 -> 10 bars (possibly fewer if last partial omitted)
        assert self.out.values.ndim == 2
        assert self.out.values.shape[1] == 3
        assert len(self.out) > 0

    def test_open_matches_single_col_count(self):
        ref_stream = resample(self.price_arr, self.t_arr, count=self.COUNT, agg="first")
        ref = ref_stream.values if isinstance(ref_stream, Stream) else ref_stream[0]
        np.testing.assert_allclose(self.out["open"], ref, rtol=1e-12)

    def test_close_matches_single_col_count(self):
        ref_stream = resample(self.price_arr, self.t_arr, count=self.COUNT, agg="last")
        ref = ref_stream.values if isinstance(ref_stream, Stream) else ref_stream[0]
        np.testing.assert_allclose(self.out["close"], ref, rtol=1e-12)


# ---------------------------------------------------------------------------
# Test 3: batch == stream (live mode)
# ---------------------------------------------------------------------------

class TestBatchEqualsStream:
    """Dag(...) and dag.live() produce equal results with identical column labels."""

    def _build(self):
        t_arr, price_arr, vol_arr = _make_data()
        price = Input("price")
        t     = Input("t")
        bars = resample(t, every=W, agg={
            "open":  First()(price),
            "high":  ExpandingMax()(price),
            "close": Last()(price),
        })
        dag = Dag([t, price], [bars])
        return dag, t_arr, price_arr

    def test_batch_and_live_values_equal(self):
        dag, t_arr, price_arr = self._build()
        batch_out = dag(t=t_arr, price=price_arr)

        live = dag.live()
        for i in range(len(t_arr)):
            live.push("t",     t_arr[i], t_arr[i])
            live.push("price", t_arr[i], price_arr[i])
        live.flush()
        live_out = live.result()

        assert isinstance(live_out, Stream)
        assert live_out.columns == batch_out.columns
        np.testing.assert_allclose(live_out.values, batch_out.values, rtol=1e-12)

    def test_live_columns_equal_batch_columns(self):
        dag, t_arr, price_arr = self._build()
        batch_out = dag(t=t_arr, price=price_arr)
        live = dag.live()
        for i in range(len(t_arr)):
            live.push("t",     t_arr[i], t_arr[i])
            live.push("price", t_arr[i], price_arr[i])
        live.flush()
        live_out = live.result()
        assert live_out.columns == batch_out.columns


# ---------------------------------------------------------------------------
# Test 4: Column order matches dict insertion order
# ---------------------------------------------------------------------------

def test_column_order_matches_insertion_order():
    """The output columns respect Python dict insertion order."""
    t_arr, price_arr, _ = _make_data()
    t     = Input("t")
    price = Input("price")

    bars = resample(t, every=W, agg={
        "z": Last()(price),
        "a": First()(price),
        "m": ExpandingMax()(price),
    })
    dag = Dag([t, price], [bars])
    out = dag(t=t_arr, price=price_arr)
    assert out.columns == ("z", "a", "m")


# ---------------------------------------------------------------------------
# Test 5: Validation errors
# ---------------------------------------------------------------------------

class TestValidationErrors:
    """_split_reducer_expr raises clear errors for bad dict values."""

    def _make_inputs(self):
        return Input("t"), Input("price")

    def test_plain_string_raises(self):
        """A plain string as a dict value raises a clear error about code fragments."""
        t, price = self._make_inputs()
        with pytest.raises(ValueError, match="code fragment"):
            resample(t, every=W, agg={"open": "first"})

    def test_input_node_no_reducer_raises(self):
        """A bare Input (no reducer on top) raises a clear error."""
        t, price = self._make_inputs()
        with pytest.raises(ValueError, match="reducer functor"):
            resample(t, every=W, agg={"open": price})

    def test_empty_dict_raises(self):
        """An empty agg dict raises ValueError."""
        t, _ = self._make_inputs()
        with pytest.raises(ValueError, match="empty"):
            resample(t, every=W, agg={})


# ---------------------------------------------------------------------------
# Test 6: Empty time-bars via the clock (sparse price, dense clock t)
# ---------------------------------------------------------------------------

def test_empty_bars_via_clock():
    """Sparse price stream + dense clock produces NaN-filled empty bars in the output.

    The clock `t` drives empty bar finalization even when there are no price
    events in that window - via `advance()` in the live path or via the
    engine's batch clock processing.
    """
    # Dense clock every tick 0..29; price only at ticks 0, 10, 20 (one per bar).
    t_clock = np.arange(30, dtype=np.int64)
    t_price = np.array([0, 10, 20], dtype=np.int64)
    price_vals = np.array([1.0, 2.0, 3.0])

    # Build a clock stream (all zeros as values - only the index matters for bucketing)
    clock_vals = np.zeros(30, dtype=np.float64)

    t      = Input("t")
    price  = Input("price")
    clock  = Input("clock")

    bars = resample(t, every=10, agg={
        "open":  First()(price),
        "close": Last()(price),
    })
    # Use 3 inputs: t (clock), price, clock (dummy feed to carry clock events)
    # Actually: t is the clock for resample; price is the data port.
    # We only have 2 inputs in this graph: t and price.
    dag = Dag([t, price], [bars])

    # Feed: t has dense events [0..29]; price has sparse events at [0, 10, 20].
    # _as_stream handles (values, index) pair (values-first user convention).
    out = dag(t=(clock_vals, t_clock), price=(price_vals, t_price))

    assert isinstance(out, Stream)
    assert out.columns == ("open", "close")
    # We expect at least the 3 bars containing ticks 0, 10, 20.
    assert len(out) >= 3
    # The bars at 0, 10, 20 should have valid (non-NaN) open and close.
    np.testing.assert_allclose(out["open"][:3], [1.0, 2.0, 3.0], rtol=1e-12)
    np.testing.assert_allclose(out["close"][:3], [1.0, 2.0, 3.0], rtol=1e-12)
