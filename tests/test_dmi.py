import numpy as np
import pytest
import talib

from screamer import ADX, PlusDI, MinusDI, PlusDM, MinusDM
from tests.regime_helpers import assert_batch_equals_scalar


def _ohlc(n, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.standard_normal(n))
    high = close + np.abs(rng.standard_normal(n))
    low = close - np.abs(rng.standard_normal(n))
    return high, low, close


class TestADXUnchanged:
    @pytest.mark.parametrize("w", [14, 20])
    def test_adx_matches_talib_after_refactor(self, w):
        high, low, close = _ohlc(300, seed=w)
        out = ADX(w)(high, low, close)
        pdi, mdi, adx = out[:, 0], out[:, 1], out[:, 2]
        ref_pdi = talib.PLUS_DI(high, low, close, timeperiod=w)
        ref_mdi = talib.MINUS_DI(high, low, close, timeperiod=w)
        ref_adx = talib.ADX(high, low, close, timeperiod=w)
        for ours, ref in [(pdi, ref_pdi), (mdi, ref_mdi), (adx, ref_adx)]:
            m = np.isfinite(np.asarray(ours)) & np.isfinite(ref)
            np.testing.assert_allclose(np.asarray(ours)[m], ref[m], atol=1e-8)

    def test_adx_batch_equals_stream(self):
        high, low, close = _ohlc(200, seed=1)
        assert_batch_equals_scalar(lambda: ADX(14), high, low, close)


class TestPlusMinusDI:
    @pytest.mark.parametrize("w", [14, 20])
    def test_plus_di_matches_talib(self, w):
        high, low, close = _ohlc(300, seed=w + 1)
        ours = np.asarray(PlusDI(w)(high, low, close))
        ref = talib.PLUS_DI(high, low, close, timeperiod=w)
        m = np.isfinite(ours) & np.isfinite(ref)
        np.testing.assert_allclose(ours[m], ref[m], atol=1e-8)

    @pytest.mark.parametrize("w", [14, 20])
    def test_minus_di_matches_talib(self, w):
        high, low, close = _ohlc(300, seed=w + 2)
        ours = np.asarray(MinusDI(w)(high, low, close))
        ref = talib.MINUS_DI(high, low, close, timeperiod=w)
        m = np.isfinite(ours) & np.isfinite(ref)
        np.testing.assert_allclose(ours[m], ref[m], atol=1e-8)

    def test_di_batch_equals_stream(self):
        high, low, close = _ohlc(200, seed=3)
        assert_batch_equals_scalar(lambda: PlusDI(14), high, low, close)
        assert_batch_equals_scalar(lambda: MinusDI(14), high, low, close)


class TestPlusMinusDM:
    @pytest.mark.parametrize("w", [14, 20])
    def test_plus_dm_matches_talib_aligned(self, w):
        high, low, close = _ohlc(300, seed=w + 4)
        ours = np.asarray(PlusDM(w)(high, low))
        ref = talib.PLUS_DM(high, low, timeperiod=w)
        # screamer uses a uniform first-valid index of window_size across the
        # DMI family; talib emits PLUS_DM one bar earlier. Compare where both
        # are finite (index >= window_size), where the values are identical.
        m = np.isfinite(ours) & np.isfinite(ref)
        assert m.sum() > 0
        np.testing.assert_allclose(ours[m], ref[m], atol=1e-8)

    @pytest.mark.parametrize("w", [14, 20])
    def test_minus_dm_matches_talib_aligned(self, w):
        high, low, close = _ohlc(300, seed=w + 5)
        ours = np.asarray(MinusDM(w)(high, low))
        ref = talib.MINUS_DM(high, low, timeperiod=w)
        m = np.isfinite(ours) & np.isfinite(ref)
        assert m.sum() > 0
        np.testing.assert_allclose(ours[m], ref[m], atol=1e-8)

    def test_dm_batch_equals_stream(self):
        high, low, close = _ohlc(200, seed=6)
        assert_batch_equals_scalar(lambda: PlusDM(14), high, low)
        assert_batch_equals_scalar(lambda: MinusDM(14), high, low)
