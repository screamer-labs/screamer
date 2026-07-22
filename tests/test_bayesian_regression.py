import numpy as np
import pytest
from screamer import BayesianRegression


def test_requires_exactly_one_forgetting_arg():
    with pytest.raises((ValueError, Exception)):
        BayesianRegression()                       # none
    with pytest.raises((ValueError, Exception)):
        BayesianRegression(span=20, alpha=0.1)     # two


def test_recovers_true_slope_and_intercept():
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.standard_normal(n)
    y = 1.5 - 0.8 * x + rng.standard_normal(n) * 0.5
    out = BayesianRegression(alpha=1e-4)(y, x)     # long memory
    pred_mean, pred_std, slope, intercept = out.T
    assert abs(slope[-1] - (-0.8)) < 0.05
    assert abs(intercept[-1] - 1.5) < 0.05


def test_defined_from_first_step_no_nan_warmup():
    out = BayesianRegression(alpha=0.1)(np.array([1.0, 2.0, 3.0]),
                                        np.array([0.5, -0.5, 1.0]))
    assert np.all(np.isfinite(out))                # no NaN warmup, unlike RollingLinearRegression
    assert (out[:, 1] > 0).all()                   # pred_std positive


def test_nan_input_emits_nan_row_state_untouched():
    x = np.array([0.5, np.nan, -0.5, 1.0])
    y = np.array([1.0, 5.0, 2.0, 0.5])
    out = BayesianRegression(alpha=0.1)(y, x)
    assert np.isnan(out[1]).all()                  # the NaN row is all NaN
    # the step after the NaN equals the same series with the NaN pair removed
    x2 = np.array([0.5, -0.5, 1.0]); y2 = np.array([1.0, 2.0, 0.5])
    out2 = BayesianRegression(alpha=0.1)(y2, x2)
    np.testing.assert_allclose(out[2], out2[1], rtol=1e-12, atol=1e-12)


def test_reset_restores_prior():
    op = BayesianRegression(alpha=0.1)
    rng = np.random.default_rng(1)
    x = rng.standard_normal(50); y = 2 * x + 1
    first = op(y, x)
    op.reset()
    again = op(y, x)
    np.testing.assert_allclose(first, again, rtol=1e-12, atol=1e-12)


def test_batch_equals_stream():
    rng = np.random.default_rng(2)
    n = 300
    x = rng.standard_normal(n); y = 0.7 * x - 0.3 + rng.standard_normal(n) * 0.4
    batch = BayesianRegression(alpha=0.05)(y, x)
    op = BayesianRegression(alpha=0.05)
    stream = np.array([op(float(y[i]), float(x[i])) for i in range(n)])
    np.testing.assert_allclose(np.nan_to_num(stream), np.nan_to_num(batch), rtol=1e-11)
