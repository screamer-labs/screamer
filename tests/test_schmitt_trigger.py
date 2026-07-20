"""Tests for the Schmitt trigger.

The trigger has three documented behaviors:

1. Cross-above-upper -> latches 1.0.
2. Cross-below-lower -> latches 0.0.
3. In the dead band ``[lower, upper]`` -> output unchanged.

Plus the ``initial`` latch seed (default 0.0, the low state) that the
output holds until the first threshold crossing, and the library-wide
invariants: NaN input emits NaN with no state update, ``reset()``
returns the latched state to the initial seed, and streaming must match
batch on identical input.
"""
import math

import numpy as np
import pytest

from screamer import SchmittTrigger


def test_constructor_rejects_lower_ge_upper():
    with pytest.raises(ValueError):
        SchmittTrigger(lower=1.0, upper=1.0)
    with pytest.raises(ValueError):
        SchmittTrigger(lower=2.0, upper=1.0)


def test_constructor_rejects_non_finite_thresholds():
    with pytest.raises(ValueError):
        SchmittTrigger(lower=float("nan"), upper=1.0)
    with pytest.raises(ValueError):
        SchmittTrigger(lower=-1.0, upper=float("nan"))


def test_constructor_rejects_invalid_initial():
    with pytest.raises(ValueError):
        SchmittTrigger(lower=-1.0, upper=1.0, initial=0.5)
    with pytest.raises(ValueError):
        SchmittTrigger(lower=-1.0, upper=1.0, initial=-1.0)


def test_default_warmup_reads_low():
    """By default the latch seeds low, so a signal that starts inside the
    dead band reads 0.0 (not NaN) until it first crosses a threshold."""
    trig = SchmittTrigger(lower=-1.0, upper=1.0)
    out = trig(np.array([0.0, 0.5, -0.5, 0.9, -0.9]))
    np.testing.assert_array_equal(out, np.zeros(5))


def test_initial_high_reads_high_during_warmup():
    """initial=1.0 seeds the latch high until the first cross below lower."""
    trig = SchmittTrigger(lower=-1.0, upper=1.0, initial=1.0)
    out = trig(np.array([0.0, 0.5, -0.5, 0.9]))
    np.testing.assert_array_equal(out, np.ones(4))


def test_initial_nan_reproduces_undefined_warmup():
    """initial=NaN restores the old behavior: undefined until first crossing."""
    trig = SchmittTrigger(lower=-1.0, upper=1.0, initial=float("nan"))
    out = trig(np.array([0.0, 0.5, -0.5, 0.9, -0.9]))
    assert np.all(np.isnan(out))


def test_high_then_dead_band_latches_high():
    trig = SchmittTrigger(lower=-1.0, upper=1.0)
    x = np.array([2.0, 0.0, 0.5, -0.5, 0.9, -0.9])
    expected = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    np.testing.assert_array_equal(trig(x), expected)


def test_low_then_dead_band_latches_low():
    trig = SchmittTrigger(lower=-1.0, upper=1.0)
    x = np.array([-2.0, 0.0, 0.5, -0.5, 0.9, -0.9])
    expected = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(trig(x), expected)


def test_hysteresis_blocks_chatter_within_dead_band():
    """A signal that bounces inside the dead band must NOT toggle the output."""
    trig = SchmittTrigger(lower=-1.0, upper=1.0)
    x = np.array([2.0, 0.5, -0.5, 0.5, -0.5, 0.5])
    out = trig(x)
    # Latched high on the first cross, then no change.
    np.testing.assert_array_equal(out, np.ones_like(x))


def test_full_round_trip_high_to_low_to_high():
    trig = SchmittTrigger(lower=-1.0, upper=1.0)
    x = np.array([2.0,  # cross upper -> high
                  0.5,  # dead band   -> hold high
                  -2.0, # cross lower -> low
                  -0.5, # dead band   -> hold low
                  0.5,  # dead band   -> hold low
                  2.0,  # cross upper -> high
                  0.0]) # dead band   -> hold high
    expected = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    np.testing.assert_array_equal(trig(x), expected)


def test_exactly_at_threshold_does_not_trigger():
    """Strict inequality: x == upper is INSIDE the dead band, not above."""
    trig = SchmittTrigger(lower=-1.0, upper=1.0)
    # Every sample sits exactly on a threshold (dead band), so the latch never
    # leaves its low seed: == upper does not trigger high, == lower stays low.
    out = trig(np.array([1.0, -1.0, 1.0]))
    np.testing.assert_array_equal(out, np.zeros(3))


def test_nan_input_emits_nan_state_untouched():
    """NaN input under the library's "ignore" policy must not corrupt state."""
    trig = SchmittTrigger(lower=-1.0, upper=1.0)
    x = np.array([2.0, np.nan, 0.5, np.nan, -2.0, np.nan, 0.5])
    out = trig(x)
    # Indices 0..1: high latched; 1 is NaN-in -> NaN-out.
    assert out[0] == 1.0
    assert math.isnan(out[1])
    assert out[2] == 1.0          # latched value preserved despite the NaN
    assert math.isnan(out[3])
    assert out[4] == 0.0          # cross lower -> low
    assert math.isnan(out[5])
    assert out[5] != out[5]       # NaN
    assert out[6] == 0.0          # latched low preserved


def test_reset_returns_to_initial():
    trig = SchmittTrigger(lower=-1.0, upper=1.0)
    trig(np.array([2.0, 0.0, 0.0]))
    trig.reset()
    # After reset, the latch is back at its initial seed (default low).
    out = trig(np.array([0.5, 0.0]))
    np.testing.assert_array_equal(out, np.zeros(2))
    # With initial=NaN, reset restores the undefined warmup.
    trig_nan = SchmittTrigger(lower=-1.0, upper=1.0, initial=float("nan"))
    trig_nan(np.array([2.0, 0.0]))
    trig_nan.reset()
    assert np.all(np.isnan(trig_nan(np.array([0.5, 0.0]))))


def test_stream_matches_batch():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    a = SchmittTrigger(lower=-0.5, upper=0.5)
    b = SchmittTrigger(lower=-0.5, upper=0.5)
    out_stream = np.array([a(float(v)) for v in x])
    out_batch = b(x.copy())
    np.testing.assert_array_equal(out_stream, out_batch)


def test_asymmetric_thresholds():
    """Lower and upper don't have to be symmetric around zero."""
    trig = SchmittTrigger(lower=0.0, upper=10.0)
    x = np.array([20.0, 5.0, -1.0, 5.0, 15.0])
    expected = np.array([1.0, 1.0, 0.0, 0.0, 1.0])
    np.testing.assert_array_equal(trig(x), expected)
