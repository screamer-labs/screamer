import math

import numpy as np
import pytest

from devtools.baselines.RollingUpsideDeviation import RollingUpsideDeviation_numpy
from screamer import RollingDownsideDeviation, RollingUpsideDeviation
from tests.regime_helpers import assert_regimes_agree


def test_hand_example_strict():
    out = np.asarray(RollingUpsideDeviation(3)(np.array([1.0, -2.0, 2.0, 0.0])))
    assert np.isnan(out[:2]).all()
    np.testing.assert_allclose(out[2:], [math.sqrt(5 / 3), math.sqrt(4 / 3)], rtol=1e-12)


def test_hand_example_mar():
    out = np.asarray(RollingUpsideDeviation(3, mar=1.0)(np.array([1.0, -2.0, 2.0])))
    np.testing.assert_allclose(out[2], math.sqrt(1 / 3), rtol=1e-12)


def test_hand_example_expanding_and_zero():
    x = np.array([1.0, -2.0, 2.0])
    exp = np.asarray(RollingUpsideDeviation(3, start_policy="expanding")(x))
    np.testing.assert_allclose(exp, [1.0, math.sqrt(1 / 2), math.sqrt(5 / 3)], rtol=1e-12)
    zero = np.asarray(RollingUpsideDeviation(3, start_policy="zero")(x))
    np.testing.assert_allclose(zero, [math.sqrt(1 / 3), math.sqrt(1 / 3), math.sqrt(5 / 3)], rtol=1e-12)


@pytest.mark.parametrize("mar", [0.0, 0.3, -0.2])
def test_mirror_of_downside_deviation(mar):
    x = np.random.default_rng(0).standard_normal(300)
    np.testing.assert_allclose(np.asarray(RollingUpsideDeviation(30, mar=mar)(x)),
                               np.asarray(RollingDownsideDeviation(30, mar=-mar)(-x)),
                               rtol=1e-12, equal_nan=True)


@pytest.mark.parametrize("start_policy", ["strict", "expanding", "zero"])
def test_matches_baseline(start_policy):
    x = np.random.default_rng(1).standard_normal(400)
    x[[5, 77, 300]] = np.nan
    got = np.asarray(RollingUpsideDeviation(25, mar=0.1, start_policy=start_policy)(x))
    want = RollingUpsideDeviation_numpy(25, 0.1, start_policy)(x)
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12, equal_nan=True)


def test_regimes_agree():
    x = np.random.default_rng(2).standard_normal(300)
    x[50] = np.nan
    assert_regimes_agree(lambda: RollingUpsideDeviation(20, mar=0.05), x)
