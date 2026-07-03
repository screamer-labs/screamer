import numpy as np
import pytest
from screamer import RollingCorr, Cart2Polar


def test_two_input_functor_accepts_TxN_array():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200)
    y = rng.standard_normal(200)
    aligned = np.column_stack([x, y])          # shape (200, 2)
    got = RollingCorr(20)(aligned)
    exp = RollingCorr(20)(x, y)
    np.testing.assert_array_equal(got, exp)


def test_TxN_array_wrong_width_raises():
    rng = np.random.default_rng(1)
    bad = rng.standard_normal((100, 3))        # 3 cols for a 2-input functor
    with pytest.raises((ValueError, TypeError)):
        RollingCorr(20)(bad)


def test_NtoM_functor_accepts_TxN_array():
    rng = np.random.default_rng(2)
    xy = rng.standard_normal((50, 2))
    got = Cart2Polar()(xy)                      # N=2, M=2
    exp = Cart2Polar()(xy[:, 0], xy[:, 1])
    np.testing.assert_array_equal(got, exp)
