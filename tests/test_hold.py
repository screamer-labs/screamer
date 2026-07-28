import math
import numpy as np
import pytest

from screamer import Hold
from tests.regime_helpers import assert_batch_equals_scalar


def test_hold_worked_example():
    out = np.asarray(Hold(n=3)(np.array([0., 5, 0, 0, 0, -2, 0, 0])))
    np.testing.assert_array_equal(out, [0, 5, 5, 5, 0, -2, -2, -2])


def test_hold_n1_shows_only_trigger_bar():
    out = np.asarray(Hold(n=1)(np.array([0., 5, 0, 7, 0])))
    np.testing.assert_array_equal(out, [0, 5, 0, 7, 0])


def test_hold_release_value():
    out = np.asarray(Hold(n=2, release=-1.0)(np.array([0., 5, 0, 0, 0])))
    np.testing.assert_array_equal(out, [-1, 5, 5, -1, -1])


def test_hold_nan_ignored_state_untouched():
    out = np.asarray(Hold(n=3)(np.array([0., 5, np.nan, 0, 0])))
    # NaN passes through; the hold counter does NOT advance on the NaN bar
    assert math.isnan(out[2])
    np.testing.assert_array_equal(out[[0, 1, 3, 4]], [0, 5, 5, 5])


def test_hold_rejects_n_below_1():
    with pytest.raises(ValueError):
        Hold(n=0)


def test_hold_batch_equals_scalar():
    rng = np.random.default_rng(0)
    x = np.where(rng.random(200) < 0.1, rng.standard_normal(200), 0.0)
    x[::17] = np.nan
    assert_batch_equals_scalar(lambda: Hold(n=5), x)
