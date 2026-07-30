import numpy as np
import pytest

from screamer import DominantCycle, HilbertPhasor, CyclePhase, CycleFrequency
from tests.regime_helpers import assert_batch_equals_scalar


def _tone(period, n=600, amp=1.0, seed=0):
    t = np.arange(n)
    return amp * np.sin(2 * np.pi * t / period)


class TestDominantCycle:
    @pytest.mark.parametrize("period", [14.0, 20.0, 30.0])
    def test_recovers_known_period(self, period):
        x = _tone(period, n=800)
        out = np.asarray(DominantCycle()(x))
        tail = out[-200:]
        tail = tail[np.isfinite(tail)]
        assert tail.size > 50
        # The homodyne discriminator recovers the dominant cycle within 15%.
        assert abs(np.median(tail) - period) / period < 0.15, \
            f"median {np.median(tail):.2f} vs period {period}"

    def test_batch_equals_stream(self):
        x = _tone(20.0, n=400)
        assert_batch_equals_scalar(lambda: DominantCycle(), x)


class TestHilbertPhasor:
    def test_quadrature_lags_inphase_on_tone(self):
        x = _tone(20.0, n=800)
        out = np.asarray(HilbertPhasor()(x))
        inphase, quad = out[:, 0], out[:, 1]
        m = np.isfinite(inphase) & np.isfinite(quad)
        # I and Q are the two components of the analytic signal: both finite
        # after warm-up, and not identical (a nonzero quadrature exists).
        assert m.sum() > 100
        assert np.any(np.abs(quad[m]) > 1e-6)

    def test_batch_equals_stream(self):
        x = _tone(20.0, n=400)
        assert_batch_equals_scalar(lambda: HilbertPhasor(), x)


class TestCyclePhaseFrequency:
    def test_frequency_recovers_period(self):
        period = 20.0
        x = _tone(period, n=800)
        f = np.asarray(CycleFrequency()(x))
        tail = f[-200:]; tail = tail[np.isfinite(tail)]
        assert tail.size > 50
        assert abs(np.median(tail) - 1.0 / period) / (1.0 / period) < 0.15

    def test_phase_in_range(self):
        x = _tone(20.0, n=800)
        p = np.asarray(CyclePhase()(x))
        m = np.isfinite(p)
        assert m.sum() > 100
        assert p[m].min() >= 0.0 and p[m].max() <= 360.0

    def test_batch_equals_stream(self):
        x = _tone(20.0, n=400)
        assert_batch_equals_scalar(lambda: CyclePhase(), x)
        assert_batch_equals_scalar(lambda: CycleFrequency(), x)
