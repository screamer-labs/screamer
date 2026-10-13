"""Known-signal checks for the Ehlers cycle and filter operators (tracker F3).

Each check feeds a synthetic signal whose period, phase, amplitude or trend is
known and compares the operator's estimate with that value. The review pack
docs/superpowers/reviews/2026-10-ehlers-review.md shows the same checks as
plots and explains every tolerance. A check that fails because of the operator
is marked xfail with the reason given in the pack, so the suite stays green and
the failure stays visible. The whole module runs in well under a second.

Signals are plain numpy arrays. The evaluation window starts at sample
``SETTLE`` (200): the shared Hilbert engine reports from sample 41, and its
period loop starts at zero and can grow by at most 1.5x per sample before a
0.2/0.8 and a 0.33/0.67 smoothing pass, so the first ~100 samples still carry
the start-up transient.
"""

import numpy as np
import pytest

from screamer import (
    CycleAmplitude,
    CycleFrequency,
    CyclePhase,
    CycleSine,
    Decycler,
    DominantCycle,
    HilbertPhasor,
    InstantaneousTrendline,
    MAMA,
    RoofingFilter,
    SuperSmoother,
    TrendMode,
)

SETTLE = 200
N = 1200
PERIODS = (10.0, 20.0, 40.0)


# ---------------------------------------------------------------- signals

def sine(period, n=N, amp=1.0):
    t = np.arange(n, dtype=float)
    return amp * np.sin(2.0 * np.pi * t / period)


def chirp(p0=15.0, p1=40.0, n=1500):
    """Sine whose period moves linearly from p0 to p1; returns (x, period)."""
    period = np.linspace(p0, p1, n)
    phase = 2.0 * np.pi * np.cumsum(1.0 / period)
    return np.sin(phase), period


def sine_plus_trend(period, n=N, amp=1.0):
    """Sine plus a ramp that rises one amplitude per cycle; returns (x, trend)."""
    t = np.arange(n, dtype=float)
    trend = (amp / period) * t
    return amp * np.sin(2.0 * np.pi * t / period) + trend, trend


def best_lagged_error(est, period, settle=100):
    """Relative error of est[t] against period[t - L], best L in [0, period[t]].

    This is the chirp criterion: the estimate may trail the true period by up
    to one cycle.
    """
    out = np.empty(len(est) - settle)
    for k, i in enumerate(range(settle, len(est))):
        lag = int(np.ceil(period[i]))
        cands = period[max(0, i - lag): i + 1]
        out[k] = np.min(np.abs(est[i] / cands - 1.0))
    return out


def lag_of_max_xcorr(a, b, max_lag):
    """Shift k maximising corr(a[t], b[t + k]); k > 0 means a leads b."""
    lo, hi = SETTLE, len(a) - max_lag - 1
    ks = np.arange(-max_lag, max_lag + 1)
    c = [np.corrcoef(a[lo:hi], b[lo + k: hi + k])[0, 1] for k in ks]
    return int(ks[int(np.argmax(c))])


def gain_db(op, period, n=4000):
    """Steady-state gain in dB of op for a unit sine of the given period."""
    y = np.asarray(op(sine(period, n=n)))[n // 2:]
    return 20.0 * np.log10(np.sqrt(2.0) * np.std(y))


def unwrap_deg(d):
    return (d + 180.0) % 360.0 - 180.0


# ------------------------------------------------- period and frequency

@pytest.mark.parametrize("period", PERIODS)
def test_dominant_cycle_pure_sine(period):
    est = np.asarray(DominantCycle()(sine(period)))[SETTLE:]
    assert np.max(np.abs(est / period - 1.0)) <= 0.05


@pytest.mark.parametrize("period", PERIODS)
def test_cycle_frequency_pure_sine(period):
    est = np.asarray(CycleFrequency()(sine(period)))[SETTLE:]
    assert np.max(np.abs(est * period - 1.0)) <= 0.05


@pytest.mark.parametrize("direction", ["up", "down"])
def test_dominant_cycle_chirp(direction):
    x, period = chirp()
    if direction == "down":
        x, period = chirp(40.0, 15.0)
    est = np.asarray(DominantCycle()(x))
    assert np.max(best_lagged_error(est, period)) <= 0.10


@pytest.mark.parametrize("period", PERIODS)
def test_dominant_cycle_sine_plus_trend_after_roofing(period):
    # A raw ramp biases the homodyne discriminator (12.3 for a period of 10,
    # identical to TA-Lib HT_DCPERIOD), so Ehlers measures cycles on
    # roofing-filtered data. The check follows that recipe.
    x, _ = sine_plus_trend(period)
    roofed = np.asarray(RoofingFilter(hp_period=48.0, lp_period=8.0)(x))
    est = np.asarray(DominantCycle()(roofed))[SETTLE:]
    assert np.max(np.abs(est / period - 1.0)) <= 0.05


def test_dominant_cycle_matches_talib_reference():
    # TA-Lib HT_DCPERIOD implements the same Rocket Science homodyne
    # discriminator; after both start-up transients the two agree.
    talib = pytest.importorskip("talib")
    x = 100.0 + np.cumsum(np.random.default_rng(3).normal(size=3000))
    ours = np.asarray(DominantCycle()(x))[SETTLE:]
    ref = talib.HT_DCPERIOD(x)[SETTLE:]
    assert np.max(np.abs(ours - ref)) <= 1e-4


# ------------------------------------------------------ amplitude, I/Q

@pytest.mark.xfail(
    strict=True,
    reason="F3 pack: CycleAmplitude reports 0.41, 1.08, 1.41 for a unit sine "
           "of period 10, 20, 40; the magnitude of (I2, Q2) is not normalised "
           "to the input amplitude.")
@pytest.mark.parametrize("period", PERIODS)
def test_cycle_amplitude_pure_sine(period):
    est = np.asarray(CycleAmplitude()(sine(period)))[SETTLE:]
    assert abs(np.median(est) - 1.0) <= 0.05


@pytest.mark.parametrize("period", PERIODS)
def test_hilbert_phasor_quadrature(period):
    out = np.asarray(HilbertPhasor()(sine(period)))
    i, q = out[:, 0], out[:, 1]
    # Equal size, and Q leads I by a quarter cycle (Ehlers' convention:
    # Q1 is the Hilbert FIR, a differentiator, of the centred I1).
    assert abs(np.std(q[SETTLE:]) / np.std(i[SETTLE:]) - 1.0) <= 0.05
    lead = lag_of_max_xcorr(q, i, int(period) // 2)
    assert abs(lead - period / 4.0) <= 1.0


# ---------------------------------------------------------------- phase

@pytest.mark.parametrize("period", PERIODS)
def test_cycle_phase_rate(period):
    ph = np.asarray(CyclePhase()(sine(period)))[SETTLE:]
    d = unwrap_deg(np.diff(ph))
    # Ehlers' convention: the phase decreases by 360 / period per sample
    # (MAMA's DeltaPhase = Phase[1] - Phase is positive on a cycle).
    assert abs(-np.median(d) / (360.0 / period) - 1.0) <= 0.05


@pytest.mark.parametrize("period", PERIODS)
def test_cycle_phase_tracks_input_phase(period):
    t = np.arange(N, dtype=float)
    ph = np.asarray(CyclePhase()(sine(period)))
    offset = unwrap_deg(ph + 360.0 * t / period)[SETTLE:]
    # A constant offset (filter delay) is fine; it must not wander.
    offset = unwrap_deg(offset - offset[0])
    assert np.std(offset) <= 5.0


@pytest.mark.parametrize("period", PERIODS)
def test_cycle_sine_follows_input(period):
    sine_out = np.asarray(CycleSine()(sine(period)))[:, 0]
    s = sine_out[SETTLE:]
    crossings = np.count_nonzero(np.diff(np.signbit(s)))
    cycles = crossings / 2.0
    assert abs(len(s) / cycles / period - 1.0) <= 0.05


@pytest.mark.xfail(
    strict=True,
    reason="F3 pack: leadsine lags sine by period/8 instead of leading it, "
           "because CyclePhase decreases with time and CycleSine adds +45 "
           "degrees; TA-Lib HT_SINE leads by period/8 on the same input.")
@pytest.mark.parametrize("period", PERIODS)
def test_cycle_sine_leadsine_leads(period):
    out = np.asarray(CycleSine()(sine(period)))
    lead = lag_of_max_xcorr(out[:, 1], out[:, 0], int(period) // 2)
    assert abs(lead - period / 8.0) <= 1.0


# ----------------------------------------------------------- trend mode

@pytest.mark.parametrize("period", PERIODS)
def test_trend_mode_sine_plus_trend_is_cycle(period):
    x, _ = sine_plus_trend(period)
    tm = np.asarray(TrendMode()(x))[SETTLE:]
    assert np.mean(tm == 0.0) >= 0.9


@pytest.mark.xfail(
    strict=True,
    reason="F3 pack: on a pure linear ramp TrendMode reports trend on only "
           "64% of samples; the period/phase loop self-oscillates on a ramp. "
           "TA-Lib HT_TRENDMODE reports trend on 100%.")
def test_trend_mode_pure_ramp_is_trend():
    x = 0.05 * np.arange(N, dtype=float)
    tm = np.asarray(TrendMode()(x))[SETTLE:]
    assert np.mean(tm == 1.0) >= 0.9


# ------------------------------------------------- trendline, MAMA

def test_instantaneous_trendline_ramp_no_lag():
    x = 0.05 * np.arange(N, dtype=float)
    it = np.asarray(InstantaneousTrendline()(x))[SETTLE:]
    assert np.max(np.abs(it - x[SETTLE:])) <= 1e-6


@pytest.mark.xfail(
    strict=True,
    reason="F3 pack: with alpha = 2 / (period + 1) the 2-pole trendline "
           "passes about 57% of the dominant cycle, so the trend estimate "
           "carries most of the cycle.")
@pytest.mark.parametrize("period", PERIODS)
def test_instantaneous_trendline_matches_trend(period):
    x, trend = sine_plus_trend(period)
    it = np.asarray(InstantaneousTrendline()(x))[SETTLE:]
    assert np.max(np.abs(it - trend[SETTLE:])) <= 0.1


@pytest.mark.parametrize("period", PERIODS)
def test_mama_adaptation_law(period):
    fast, slow = 0.5, 0.05
    x = sine(period) + 5.0
    m = np.asarray(MAMA(fast, slow)(x))[:, 0]
    alpha = (np.diff(m) / (x[1:] - m[:-1]))[SETTLE:]
    expected = float(np.clip(fast / (360.0 / period), slow, fast))
    steady = alpha[alpha < 0.4]
    assert abs(np.median(steady) / expected - 1.0) <= 0.05
    # One phase wrap, hence one fast_limit step, per cycle (atan2 phase).
    n_fast = np.count_nonzero(alpha > 0.4)
    assert abs(n_fast / (len(alpha) / period) - 1.0) <= 0.05


# -------------------------------------------------------------- filters

@pytest.mark.parametrize("period", PERIODS)
def test_super_smoother_cutoff_gain(period):
    assert abs(gain_db(SuperSmoother(period=period), period) + 3.0) <= 1.0


@pytest.mark.parametrize("period", (20.0, 60.0))
def test_decycler_cutoff_gain(period):
    assert abs(gain_db(Decycler(period=period), period) + 3.0) <= 1.0


@pytest.mark.parametrize("edge", (48.0, 10.0))
def test_roofing_filter_edge_gain(edge):
    op = RoofingFilter(hp_period=48.0, lp_period=10.0)
    assert abs(gain_db(op, edge) + 3.0) <= 1.0
