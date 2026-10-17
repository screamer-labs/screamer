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


import importlib.util
import pathlib

from tests.regime_helpers import assert_regimes_agree

_B = pathlib.Path(__file__).resolve().parents[1] / "devtools" / "baselines" / "Hold.py"
_spec = importlib.util.spec_from_file_location("hold_baseline", _B)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
Hold_numpy = _mod.Hold_numpy


def _old_hold(x, n, release=0.0):
    """Hold before cooldown existed: NaN skipped, a trigger restarts the hold."""
    held, remaining, out = release, 0, []
    for v in x:
        if math.isnan(v):
            out.append(math.nan)
        elif v != 0.0:
            held, remaining = v, n - 1
            out.append(v)
        elif remaining > 0:
            remaining -= 1
            out.append(held)
        else:
            out.append(release)
    return np.array(out)


def _events(seed, n=400):
    rng = np.random.default_rng(seed)
    x = np.where(rng.random(n) < 0.2, rng.standard_normal(n), 0.0)
    x[::37] = np.nan
    return x


def test_hold_cooldown_ignores_triggers_after_expiry():
    out = np.asarray(Hold(n=2, cooldown=2)(np.array([5., 0, 7, 3, 0, 4, 0])))
    np.testing.assert_array_equal(out, [5, 5, 0, 0, 0, 4, 4])


def test_hold_cooldown_trigger_during_hold_restarts():
    out = np.asarray(Hold(n=3, cooldown=1)(np.array([1., 0, 2, 0, 0, 0, 3, 0])))
    np.testing.assert_array_equal(out, [1, 1, 2, 2, 2, 0, 3, 3])


def test_hold_cooldown_after_one_step_hold_outputs_release():
    out = np.asarray(Hold(n=1, release=-1.0, cooldown=1)(np.array([5., 6, 7, 0, 8])))
    np.testing.assert_array_equal(out, [5, -1, 7, -1, 8])


def test_hold_cooldown_nan_does_not_use_up_a_cooldown_step():
    out = np.asarray(Hold(n=1, cooldown=1)(np.array([5., np.nan, 6, 0])))
    assert math.isnan(out[1])
    np.testing.assert_array_equal(out[[0, 2, 3]], [5, 0, 0])


@pytest.mark.parametrize("n", [1, 2, 5])
def test_hold_cooldown_zero_is_the_previous_behaviour(n):
    x = _events(n)
    np.testing.assert_array_equal(np.asarray(Hold(n=n, cooldown=0)(x)), _old_hold(x, n))
    np.testing.assert_array_equal(np.asarray(Hold(n=n)(x)), _old_hold(x, n))


@pytest.mark.parametrize("n,cooldown,release", [(1, 1, 0.0), (3, 2, 0.0), (5, 7, -1.0)])
def test_hold_cooldown_matches_baseline(n, cooldown, release):
    x = _events(10 + n)
    np.testing.assert_array_equal(np.asarray(Hold(n=n, release=release, cooldown=cooldown)(x)),
                                  Hold_numpy(n=n, release=release, cooldown=cooldown)(x))


def test_hold_cooldown_regimes_agree():
    assert_regimes_agree(lambda: Hold(n=3, cooldown=2), _events(3))


def test_hold_rejects_negative_cooldown():
    with pytest.raises(ValueError, match=r"^Hold: cooldown must be at least 0, got -1$"):
        Hold(n=3, cooldown=-1)
