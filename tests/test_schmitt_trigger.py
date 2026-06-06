"""Tests for the Schmitt trigger.

The trigger has three documented behaviors:

1. Cross-above-upper -> latches 1.0.
2. Cross-below-lower -> latches 0.0.
3. In the dead band ``[lower, upper]`` -> output unchanged.

Plus the library-wide invariants: NaN input emits NaN with no state
update, ``reset()`` returns the latched state to NaN, and streaming
must match batch on identical input.
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


def test_initial_output_is_nan_until_first_trigger():
    """Until any input crosses a threshold, the latched value is NaN."""
    trig = SchmittTrigger(lower=-1.0, upper=1.0)
    # All inside dead band -> output remains the initial NaN.
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
    # First sample sits exactly on the upper threshold: dead band -> NaN.
    out = trig(np.array([1.0, -1.0, 1.0]))
    assert np.all(np.isnan(out))


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


def test_reset_returns_to_nan():
    trig = SchmittTrigger(lower=-1.0, upper=1.0)
    trig(np.array([2.0, 0.0, 0.0]))
    trig.reset()
    # After reset, the trigger has no latched state again.
    out = trig(np.array([0.5, 0.0]))
    assert np.all(np.isnan(out))


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
