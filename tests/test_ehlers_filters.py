import numpy as np
import pytest
from scipy.signal import lfilter

from screamer import SuperSmoother
from tests.regime_helpers import assert_batch_equals_scalar


def _x(n, seed=0):
    return np.random.default_rng(seed).standard_normal(n)


def _supersmoother_ba(period):
    a1 = np.exp(-np.sqrt(2) * np.pi / period)
    b1 = 2 * a1 * np.cos(np.sqrt(2) * np.pi / period)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3
    b = [c1 / 2, c1 / 2]
    a = [1.0, -c2, -c3]
    return b, a


class TestSuperSmoother:
    @pytest.mark.parametrize("period", [10.0, 20.0, 33.0])
    def test_matches_reference(self, period):
        x = _x(500, seed=int(period))
        ours = np.asarray(SuperSmoother(period=period)(x))
        b, a = _supersmoother_ba(period)
        ref = lfilter(b, a, x)
        np.testing.assert_allclose(ours, ref, atol=1e-12)

    def test_period_equals_cutoff(self):
        x = _x(200, seed=1)
        by_period = np.asarray(SuperSmoother(period=20.0)(x))
        by_cutoff = np.asarray(SuperSmoother(cutoff=2.0 / 20.0)(x))
        np.testing.assert_allclose(by_period, by_cutoff, atol=1e-12)

    def test_requires_exactly_one_arg(self):
        with pytest.raises(Exception):
            SuperSmoother()
        with pytest.raises(Exception):
            SuperSmoother(period=20.0, cutoff=0.1)

    def test_batch_equals_stream(self):
        x = _x(300, seed=7)
        assert_batch_equals_scalar(lambda: SuperSmoother(period=15.0), x)
