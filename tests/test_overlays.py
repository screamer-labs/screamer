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

    def test_loose_talib_reference(self):
        import talib
        x = _series(800, seed=2)
        mama = np.asarray(MAMA()(x))[:, 0]
        ref, _ = talib.MAMA(x, fastlimit=0.5, slowlimit=0.05)
        m = np.isfinite(mama) & np.isfinite(ref)
        assert m.sum() > 100
        # Different analytic-signal front end -> not identical, but correlated.
        c = np.corrcoef(mama[m], ref[m])[0, 1]
        assert c > 0.9, f"correlation {c:.3f}"

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
        # On a steady ramp the trendline tracks price with small bounded error.
        assert m.sum() > 50
        assert np.max(np.abs(it[tail][m] - x[tail][m])) < 5.0

    def test_batch_equals_stream(self):
        x = _series(400, seed=5)
        assert_batch_equals_scalar(lambda: InstantaneousTrendline(), x)
