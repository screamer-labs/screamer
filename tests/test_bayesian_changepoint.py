import math

import numpy as np
import pytest

from devtools.baselines.BayesianChangepoint import BayesianChangepoint_numpy
from screamer import BayesianChangepoint
from tests.regime_helpers import assert_regimes_agree


def test_first_value():
    # one value 0.5: run length 0, mu = 0.25, beta = 1 + 0.25 / 4, var = beta * 3 / (2 * 0.5)
    out = np.asarray(BayesianChangepoint()(np.array([0.5])))
    np.testing.assert_allclose(out[0], [0.0, 0.25, math.sqrt(3.1875)], rtol=1e-12)


def test_run_length_drops_after_a_three_sigma_shift():
    rng = np.random.default_rng(3)
    x = np.r_[rng.standard_normal(300), 3.0 + rng.standard_normal(200)]
    rl = np.asarray(BayesianChangepoint(expected_run=250.0)(x))[:, 0]
    assert rl[299] > 100
    assert rl[305] < 20


def test_run_length_is_capped_by_max_run():
    x = np.random.default_rng(4).standard_normal(400)
    rl = np.asarray(BayesianChangepoint(expected_run=1000.0, max_run=50)(x))[:, 0]
    assert np.nanmax(rl) <= 50.0


@pytest.mark.parametrize("expected_run,max_run,prior_var", [(250.0, 1000, 1.0), (20.0, 30, 0.5),
                                                            (1.0, 5, 2.0)])
def test_matches_baseline(expected_run, max_run, prior_var):
    rng = np.random.default_rng(max_run)
    x = np.r_[rng.standard_normal(150), 2.0 + 0.5 * rng.standard_normal(150)]
    x[[7, 160]] = np.nan
    got = np.asarray(BayesianChangepoint(expected_run, max_run, prior_var)(x))
    want = BayesianChangepoint_numpy(expected_run, max_run, prior_var)(x)
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-10, equal_nan=True)


def test_extreme_value_does_not_break_the_state():
    x = np.r_[np.random.default_rng(5).standard_normal(50), 1e6, np.zeros(20)]
    out = np.asarray(BayesianChangepoint()(x))
    assert np.isfinite(out).all()


def test_regimes_agree():
    x = np.random.default_rng(6).standard_normal(200)
    x[[10, 100]] = np.nan
    assert_regimes_agree(lambda: BayesianChangepoint(expected_run=30.0, max_run=60), x)


def test_settings_are_checked():
    with pytest.raises(ValueError):
        BayesianChangepoint(expected_run=0.5)
    with pytest.raises(ValueError):
        BayesianChangepoint(max_run=0)
    with pytest.raises(ValueError):
        BayesianChangepoint(prior_var=0.0)


def test_nan_gives_a_nan_row_and_leaves_the_state_unchanged():
    x = np.random.default_rng(7).standard_normal(60)
    with_nan = np.insert(x, 20, np.nan)
    got = np.asarray(BayesianChangepoint(expected_run=30.0, max_run=40)(with_nan))
    want = np.asarray(BayesianChangepoint(expected_run=30.0, max_run=40)(x))
    assert np.isnan(got[20]).all()
    np.testing.assert_array_equal(np.delete(got, 20, axis=0), want)


@pytest.mark.parametrize("x", [np.zeros(3000), np.full(3000, 0.37), np.full(500, 1e100),
                               np.r_[np.zeros(100), 1e150, np.zeros(100), -1e150, np.ones(100)]])
def test_degenerate_input_stays_finite(x):
    out = np.asarray(BayesianChangepoint(expected_run=50.0, max_run=200)(x))
    assert np.isfinite(out).all()
    assert (out[:, 0] >= 0.0).all() and (out[:, 0] <= 200.0).all()
    assert (out[:, 2] >= 0.0).all()


def test_pred_std_is_nan_when_the_variance_is_nan():
    # 1e155 squared overflows: the mixture variance is NaN, and so is pred_std
    x = np.r_[np.random.default_rng(4).standard_normal(20), 1e155, 0.5]
    out = np.asarray(BayesianChangepoint()(x))
    assert np.isnan(out[20:, 2]).all()
    np.testing.assert_array_equal(np.isnan(out[20:, 2]), np.isnan(BayesianChangepoint_numpy()(x)[20:, 2]))
