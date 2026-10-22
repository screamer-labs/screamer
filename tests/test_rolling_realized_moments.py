import math

import numpy as np
import pytest

from devtools.baselines.RollingRealizedKurt import RollingRealizedKurt_numpy
from devtools.baselines.RollingRealizedSkew import RollingRealizedSkew_numpy
from screamer import RollingRealizedKurt, RollingRealizedSkew
from tests.regime_helpers import assert_regimes_agree


def test_hand_example():
    r = np.array([1.0, -2.0, 3.0])
    rv = 14.0
    skew = RollingRealizedSkew(3)(r)
    kurt = RollingRealizedKurt(3)(r)
    assert np.isnan(skew[:2]).all() and np.isnan(kurt[:2]).all()
    assert math.isclose(skew[2], math.sqrt(3) * 20.0 / rv ** 1.5, rel_tol=1e-14)
    assert math.isclose(kurt[2], 3 * 98.0 / rv ** 2, rel_tol=1e-14)


def test_symmetric_window_and_scale_invariance():
    r = np.array([1.0, -1.0, 2.0, -2.0])
    assert RollingRealizedSkew(4)(r)[-1] == 0.0
    x = np.random.default_rng(0).standard_normal(300)
    np.testing.assert_allclose(RollingRealizedSkew(50)(x * 1e-4), RollingRealizedSkew(50)(x),
                               rtol=1e-11, equal_nan=True)
    np.testing.assert_allclose(RollingRealizedKurt(50)(x * 1e-4), RollingRealizedKurt(50)(x),
                               rtol=1e-11, equal_nan=True)


def test_normal_returns_give_zero_skew_and_kurtosis_three():
    r = np.random.default_rng(1).standard_normal(400_000) * 0.001
    w = 2000
    assert abs(np.nanmean(RollingRealizedSkew(w)(r)[w - 1::w])) < 0.02
    assert abs(np.nanmean(RollingRealizedKurt(w)(r)[w - 1::w]) - 3.0) < 0.05


def test_moments_are_about_zero_not_the_mean():
    # A constant positive return: the centered skewness is undefined, the
    # realized one is sqrt(N) N c^3 / (N c^2)^1.5 = 1 and the kurtosis 1.
    r = np.full(10, 0.01)
    np.testing.assert_allclose(RollingRealizedSkew(5)(r)[4:], 1.0, rtol=1e-12)
    np.testing.assert_allclose(RollingRealizedKurt(5)(r)[4:], 1.0, rtol=1e-12)


@pytest.mark.parametrize("cls", [RollingRealizedSkew, RollingRealizedKurt])
def test_zero_window_is_nan_also_after_a_spike(cls):
    r = np.zeros(30)
    r[5] = 0.3
    out = cls(5)(r)
    assert np.isnan(out[:5]).all()
    assert np.isfinite(out[5:10]).all()
    assert np.isnan(out[10:]).all()


@pytest.mark.parametrize("cls,ref", [(RollingRealizedSkew, RollingRealizedSkew_numpy),
                                     (RollingRealizedKurt, RollingRealizedKurt_numpy)])
@pytest.mark.parametrize("start_policy", ["strict", "expanding", "zero"])
def test_matches_reference(cls, ref, start_policy):
    rng = np.random.default_rng(5)
    r = rng.standard_normal(3000) * 0.001
    r[rng.random(3000) < 0.01] *= 50               # jumps leave the window
    r[[9, 250, 1700]] = np.nan
    got = cls(40, start_policy=start_policy)(r)
    np.testing.assert_allclose(got, ref(40, start_policy)(r), rtol=1e-10, atol=1e-12,
                               equal_nan=True)


@pytest.mark.parametrize("cls", [RollingRealizedSkew, RollingRealizedKurt])
def test_small_returns_after_a_jump_keep_their_digits(cls):
    rng = np.random.default_rng(7)
    r = rng.standard_normal(400) * 1e-5
    r[100] = 1.0                                   # 1e5 times the others
    got = cls(30)(r)[140:]
    want = cls(30)(r[110:])[30:]                   # never saw the jump
    np.testing.assert_allclose(got, want, rtol=1e-12)


@pytest.mark.parametrize("cls", [RollingRealizedSkew, RollingRealizedKurt])
def test_regimes_agree(cls):
    r = np.random.default_rng(6).standard_normal(300)
    r[40] = np.nan
    assert_regimes_agree(lambda: cls(25), r)


@pytest.mark.parametrize("cls", [RollingRealizedSkew, RollingRealizedKurt])
def test_window_check(cls):
    with pytest.raises(ValueError):
        cls(1)
