"""
Tests for the signal-processing batch:

  ButterHighpass, ButterBandpass, ButterBandstop  -- IIR filters
  MovingAverage(taps)                              -- FIR filter
  KalmanFilter                                     -- 1-D scalar
"""
import numpy as np
import pytest
import scipy.signal as ss

from screamer import (
    Butter, ButterHighpass, ButterBandpass, ButterBandstop,
    MovingAverage, KalmanFilter,
)


def _x(n, seed=0):
    return np.random.default_rng(seed).standard_normal(n)


# ---------------------------------------------------------------------------
# Butter family vs scipy.signal.butter + scipy.signal.lfilter
# ---------------------------------------------------------------------------

class TestButterFamily:

    @pytest.mark.parametrize("order,cutoff", [(2, 0.1), (4, 0.2), (6, 0.4)])
    def test_lowpass_matches_scipy(self, order, cutoff):
        """Validates existing Butter (lowpass) didn't regress."""
        x = _x(500, seed=order)
        ours = Butter(order, cutoff)(x)
        b, a = ss.butter(order, cutoff, btype='lowpass')
        ref = ss.lfilter(b, a, x)
        np.testing.assert_allclose(ours, ref, atol=1e-12)

    @pytest.mark.parametrize("order,cutoff", [(2, 0.1), (4, 0.2), (6, 0.4)])
    def test_highpass_matches_scipy(self, order, cutoff):
        x = _x(500, seed=order + 100)
        ours = ButterHighpass(order, cutoff)(x)
        b, a = ss.butter(order, cutoff, btype='highpass')
        ref = ss.lfilter(b, a, x)
        np.testing.assert_allclose(ours, ref, atol=1e-12)

    @pytest.mark.parametrize("order,lo,hi", [(2, 0.1, 0.3), (4, 0.15, 0.35)])
    def test_bandpass_matches_scipy(self, order, lo, hi):
        x = _x(500, seed=order + 200)
        ours = ButterBandpass(order, lo, hi)(x)
        b, a = ss.butter(order, [lo, hi], btype='bandpass')
        ref = ss.lfilter(b, a, x)
        np.testing.assert_allclose(ours, ref, atol=1e-9)

    @pytest.mark.parametrize("order,lo,hi", [(2, 0.1, 0.3), (4, 0.15, 0.35)])
    def test_bandstop_matches_scipy(self, order, lo, hi):
        x = _x(500, seed=order + 300)
        ours = ButterBandstop(order, lo, hi)(x)
        b, a = ss.butter(order, [lo, hi], btype='bandstop')
        ref = ss.lfilter(b, a, x)
        np.testing.assert_allclose(ours, ref, atol=1e-9)

    def test_constructor_validation(self):
        with pytest.raises(ValueError):
            ButterHighpass(0, 0.2)
        with pytest.raises(ValueError):
            ButterHighpass(4, 1.5)
        with pytest.raises(ValueError):
            ButterBandpass(4, 0.4, 0.1)   # low >= high


# ---------------------------------------------------------------------------
# MovingAverage (FIR)
# ---------------------------------------------------------------------------

class TestMovingAverage:

    def test_matches_numpy_convolve_hamming(self):
        x = _x(200, seed=0)
        taps = np.hamming(11)
        taps /= taps.sum()
        ours = MovingAverage(list(taps))(x)
        ref = np.convolve(x, taps, mode='full')[:len(x)]
        ref[:10] = np.nan
        np.testing.assert_allclose(ours, ref, equal_nan=True, atol=1e-12)

    def test_uniform_taps_match_rolling_mean(self):
        """Uniform 1/n taps gives a simple moving average."""
        from screamer import RollingMean
        n = 7
        x = _x(60, seed=1)
        taps = [1.0 / n] * n
        ma = MovingAverage(taps)(x)
        rm = RollingMean(n)(x)
        np.testing.assert_allclose(ma, rm, equal_nan=True, atol=1e-12)

    def test_warmup_length(self):
        out = MovingAverage([0.5, 0.3, 0.2])(np.arange(10.0))
        assert np.all(np.isnan(out[:2]))
        assert np.all(np.isfinite(out[2:]))

    def test_constructor_validation(self):
        with pytest.raises(ValueError):
            MovingAverage([])


# ---------------------------------------------------------------------------
# KalmanFilter
# ---------------------------------------------------------------------------

class TestKalmanFilter:

    def test_extreme_observation_noise_tracks_input_loosely(self):
        """With very small observation_var, the filter trusts measurements
        completely and the output stays close to the input."""
        x = _x(200, seed=0)
        out = KalmanFilter(process_var=1.0, observation_var=1e-12)(x)
        # K -> 1, so output essentially equals input (after the first step
        # which is biased by initial_state=0).
        np.testing.assert_allclose(out[1:], x[1:], atol=1e-8)

    def test_extreme_process_noise_tracks_input_loosely(self):
        """With very large process_var, the filter forgets prior state quickly."""
        x = _x(50, seed=1)
        # With huge process_var the gain saturates at 1, filter just outputs
        # the latest observation.
        out = KalmanFilter(process_var=1e12, observation_var=1.0)(x)
        np.testing.assert_allclose(out[1:], x[1:], atol=1e-9)

    def test_zero_process_noise_converges_to_running_mean(self):
        """With process_var=0 the steady-state output approaches the
        mean of all observations seen so far."""
        x = np.full(100, 5.0)
        out = KalmanFilter(process_var=0.0, observation_var=1.0,
                            initial_state=0.0, initial_variance=1e9)(x)
        # Starting from huge initial variance, it converges quickly.
        # After enough samples, should be very close to 5.
        assert abs(out[-1] - 5.0) < 1e-3

    def test_constructor_validation(self):
        with pytest.raises(ValueError):
            KalmanFilter(process_var=-1.0, observation_var=1.0)
        with pytest.raises(ValueError):
            KalmanFilter(process_var=1.0, observation_var=0.0)

    def test_reset_clears_state(self):
        x = _x(40, seed=2)
        kf = KalmanFilter(process_var=0.1, observation_var=1.0)
        first = kf(x)
        kf.reset()
        second = kf(x)
        np.testing.assert_array_equal(first, second)


# ---------------------------------------------------------------------------
# Reset / scalar-loop parity for the IIR family
# ---------------------------------------------------------------------------

class TestParity:

    @pytest.mark.parametrize("ctor", [
        lambda: Butter(4, 0.2),
        lambda: ButterHighpass(4, 0.2),
        lambda: ButterBandpass(4, 0.1, 0.3),
        lambda: ButterBandstop(4, 0.1, 0.3),
        lambda: MovingAverage(list(np.hamming(11))),
        lambda: KalmanFilter(0.01, 1.0),
    ])
    def test_reset_clears_state(self, ctor):
        x = _x(60, seed=10)
        obj = ctor()
        first = obj(x)
        obj.reset()
        second = obj(x)
        np.testing.assert_array_equal(first, second)
