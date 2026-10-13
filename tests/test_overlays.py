import numpy as np
import pytest

from screamer import MAMA, InstantaneousTrendline
from tests.regime_helpers import assert_batch_equals_scalar


def _series(n=600, seed=0):
    rng = np.random.default_rng(seed)
    return 100 + np.cumsum(rng.standard_normal(n))


class TestMAMA:
    def test_mama_tracks_and_leads_fama(self):
        x = _series(600)
        out = np.asarray(MAMA()(x))
        mama, fama = out[:, 0], out[:, 1]
        m = np.isfinite(mama) & np.isfinite(fama)
        assert m.sum() > 100
        # MAMA and FAMA both track the series (bounded error), MAMA closer.
        err_mama = np.mean(np.abs(mama[m] - x[m]))
        err_fama = np.mean(np.abs(fama[m] - x[m]))
        assert err_mama < err_fama  # FAMA follows MAMA with more lag.

    def test_matches_talib_reference(self):
        talib = pytest.importorskip("talib")
        x = _series(800, seed=2)
        out = np.asarray(MAMA()(x))
        ref_mama, ref_fama = talib.MAMA(x, fastlimit=0.5, slowlimit=0.05)
        # Same arctan phase and adaptation rule; only the start-up differs
        # (TA-Lib starts both averages at 0), and that fades by sample 300.
        np.testing.assert_allclose(out[300:, 0], ref_mama[300:], rtol=0, atol=1e-6)
        np.testing.assert_allclose(out[300:, 1], ref_fama[300:], rtol=0, atol=1e-6)

    def test_batch_equals_stream(self):
        x = _series(400, seed=3)
        assert_batch_equals_scalar(lambda: MAMA(), x)


class TestInstantaneousTrendline:
    def test_tracks_ramp_with_bounded_lag(self):
        n = 800
        x = np.linspace(0.0, 100.0, n)  # clean upward ramp
        it = np.asarray(InstantaneousTrendline()(x))
        tail = slice(-200, None)
        m = np.isfinite(it[tail])
        # On a ramp the trendline lags by (N - 1) / 2 + 1 samples, N the
        # measured cycle length in [6, 50], so by at most 25.5 samples.
        slope = x[1] - x[0]
        assert m.sum() > 50
        lag = (x[tail][m] - it[tail][m]) / slope
        assert np.all(lag >= 3.5 - 1e-9) and np.all(lag <= 25.5 + 1e-9)

    def test_matches_talib_reference(self):
        talib = pytest.importorskip("talib")
        x = _series(800, seed=4)
        it = np.asarray(InstantaneousTrendline()(x))
        ref = talib.HT_TRENDLINE(x)
        np.testing.assert_allclose(it[200:], ref[200:], rtol=0, atol=1e-9)

    def test_batch_equals_stream(self):
        x = _series(400, seed=5)
        assert_batch_equals_scalar(lambda: InstantaneousTrendline(), x)
