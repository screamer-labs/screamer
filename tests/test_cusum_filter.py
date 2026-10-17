import math

import numpy as np
import pytest

from devtools.baselines.CusumFilter import CusumFilter_numpy
from screamer import CusumFilter
from tests.regime_helpers import assert_regimes_agree


def test_hand_example():
    x = np.array([0.5, 0.6, -0.2, 0.3, -1.5, 0.1, np.nan, 0.4])
    out = np.asarray(CusumFilter(threshold=1.0)(x))
    np.testing.assert_array_equal(out[[0, 1, 2, 3, 4, 5, 7]], [0, 1, 0, 0, -1, 0, 0])
    assert math.isnan(out[6])


def test_threshold_is_strict():
    # 0.5 + 0.5 = 1.0 is not above 1.0; the third step is
    out = np.asarray(CusumFilter(threshold=1.0)(np.array([0.5, 0.5, 0.5])))
    np.testing.assert_array_equal(out, [0, 0, 1])


def test_drift():
    x = np.array([0.5, 0.5, 0.5, 0.25, -1.0, -1.0])
    out = np.asarray(CusumFilter(threshold=1.0, drift=0.25)(x))
    np.testing.assert_array_equal(out, [0, 0, 0, 0, 0, -1])


def test_sum_restarts_after_an_event():
    # the +1 at step 1 resets S+; the fall then builds from 0 and fires at step 3
    out = np.asarray(CusumFilter(threshold=1.0)(np.array([-0.5, 1.6, -0.9, -0.2])))
    np.testing.assert_array_equal(out, [0, 1, 0, -1])


def test_repeated_events_after_reset():
    out = np.asarray(CusumFilter(threshold=1.0)(np.array([1.5, 0.4, 0.4, 0.4, -2.0])))
    np.testing.assert_array_equal(out, [1, 0, 0, 1, -1])


@pytest.mark.parametrize("threshold,drift", [(0.5, 0.0), (2.0, 0.1), (3.0, 0.5)])
def test_matches_baseline(threshold, drift):
    rng = np.random.default_rng(int(threshold * 10))
    x = rng.standard_normal(2000)
    x[::97] = np.nan
    np.testing.assert_array_equal(np.asarray(CusumFilter(threshold, drift)(x)),
                                  CusumFilter_numpy(threshold, drift)(x))


def test_regimes_agree():
    x = np.random.default_rng(1).standard_normal(500)
    x[[10, 200]] = np.nan
    assert_regimes_agree(lambda: CusumFilter(threshold=1.5, drift=0.1), x)


def test_settings_are_checked():
    with pytest.raises(ValueError):
        CusumFilter(threshold=0.0)
    with pytest.raises(ValueError):
        CusumFilter(threshold=1.0, drift=-0.1)
    with pytest.raises(ValueError, match="threshold must be finite"):
        CusumFilter(threshold=math.inf)


def _reset_both(x, threshold, drift):
    # reference that restarts both sums at every event
    pos = neg = 0.0
    out = np.full(len(x), np.nan)
    for t, v in enumerate(x):
        if math.isnan(v):
            continue
        pos = max(0.0, pos + v - drift)
        neg = min(0.0, neg + v + drift)
        out[t] = 0.0
        if neg < -threshold:
            out[t] = -1.0
        elif pos > threshold:
            out[t] = 1.0
        if out[t] != 0.0:
            pos = neg = 0.0
    return out


@pytest.mark.parametrize("drift", [0.0, 0.1, 0.5])
def test_resetting_only_the_fired_sum_equals_resetting_both(drift):
    rng = np.random.default_rng(42)
    x = rng.standard_normal(5000)
    x[::113] = np.nan
    out = np.asarray(CusumFilter(threshold=1.0, drift=drift)(x))
    assert (np.abs(out[~np.isnan(out)]) == 1).sum() > 50
    np.testing.assert_array_equal(out, _reset_both(x, 1.0, drift))
