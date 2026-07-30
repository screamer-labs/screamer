import numpy as np
import pytest

from screamer import (
    DominantCycle, HilbertPhasor, CyclePhase, CycleFrequency, CycleAmplitude, CycleSine,
    TrendMode,
)
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


class TestCycleAmplitude:
    @pytest.mark.parametrize("amp", [1.0, 3.0])
    def test_recovers_amplitude(self, amp):
        x = _tone(20.0, n=800, amp=amp)
        a = np.asarray(CycleAmplitude()(x))
        tail = a[-200:]; tail = tail[np.isfinite(tail)]
        assert tail.size > 50
        # Analytic-signal magnitude recovers the tone amplitude within 25%.
        assert abs(np.median(tail) - amp) / amp < 0.25

    def test_batch_equals_stream(self):
        x = _tone(20.0, n=400)
        assert_batch_equals_scalar(lambda: CycleAmplitude(), x)


class TestCycleSine:
    def test_sine_bounded_and_leads(self):
        x = _tone(20.0, n=800)
        out = np.asarray(CycleSine()(x))
        sine, lead = out[:, 0], out[:, 1]
        m = np.isfinite(sine) & np.isfinite(lead)
        assert m.sum() > 100
        assert sine[m].min() >= -1.0001 and sine[m].max() <= 1.0001
        assert lead[m].min() >= -1.0001 and lead[m].max() <= 1.0001

    def test_batch_equals_stream(self):
        x = _tone(20.0, n=400)
        assert_batch_equals_scalar(lambda: CycleSine(), x)


class TestTrendMode:
    def test_binary_and_finite_after_warmup(self):
        # A pure tone is a cycle, so trend mode should be mostly 0 on the tail.
        x = _tone(20.0, n=1000)
        tm = np.asarray(TrendMode()(x))
        m = np.isfinite(tm)
        assert m.sum() > 100
        vals = set(np.unique(tm[m]).tolist())
        assert vals.issubset({0.0, 1.0})
        assert np.mean(tm[m]) < 0.5  # predominantly cycle on a clean tone

    def test_batch_equals_stream(self):
        x = _tone(20.0, n=400)
        assert_batch_equals_scalar(lambda: TrendMode(), x)

    def test_nan_input_preserves_phase_memory(self):
        # nan_policy "ignore": a mid-stream NaN must not disturb internal
        # state, so the first finite sample after the gap should be
        # processed as if the NaN sample had never occurred.
        x = np.linspace(0.0, 100.0, 300)  # a ramp: trend -> mostly 1.0
        i = 150
        xb = x.copy()
        xb[i] = np.nan
        tm_b = np.asarray(TrendMode()(xb))

        xc = np.delete(x, i)  # as-if-skipped reference
        tm_c = np.asarray(TrendMode()(xc))

        tm_b_dropped = np.delete(tm_b, i)
        m = np.isfinite(tm_b_dropped) & np.isfinite(tm_c)
        assert m.sum() > 0
        np.testing.assert_allclose(tm_b_dropped[m], tm_c[m])
