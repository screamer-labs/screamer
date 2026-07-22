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


def test_predictive_intervals_are_calibrated():
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.standard_normal(n)
    y = 1.5 - 0.8 * x + rng.standard_normal(n) * 0.5
    out = BayesianRegression(alpha=1e-4)(y, x)
    pm, ps = out[200:, 0], out[200:, 1]            # skip the early learning
    within1 = (np.abs(y[200:] - pm) <= ps).mean()
    within2 = (np.abs(y[200:] - pm) <= 2 * ps).mean()
    assert 0.63 < within1 < 0.73                    # ~68%
    assert 0.92 < within2 < 0.97                    # ~95%


def test_intervals_shrink_as_evidence_grows():
    rng = np.random.default_rng(3)
    n = 3000
    x = rng.standard_normal(n)
    y = 2.0 * x + rng.standard_normal(n) * 0.3
    out = BayesianRegression(alpha=1e-4)(y, x)     # long memory -> certainty grows
    early = out[10:60, 1].mean()
    late = out[-50:, 1].mean()
    assert late < early                             # predictive std tightens with data


def test_forgetting_tracks_a_slope_change():
    rng = np.random.default_rng(4)
    n = 4000; h = n // 2
    x = rng.standard_normal(n)
    y = np.empty(n)
    y[:h] = 1.5 - 0.8 * x[:h] + rng.standard_normal(h) * 0.5
    y[h:] = 1.5 + 2.0 * x[h:] + rng.standard_normal(n - h) * 0.5
    out = BayesianRegression(alpha=0.02)(y, x)     # finite memory
    assert abs(out[h - 50, 2] - (-0.8)) < 0.2       # tracked the first regime
    assert abs(out[-1, 2] - 2.0) < 0.2              # followed the change


def test_prior_regularizes_a_degenerate_sample():
    # near-constant x: OLS slope is ill-posed; the prior keeps it finite and shrunk
    x = np.full(40, 3.0) + 1e-9 * np.arange(40)
    y = np.arange(40, dtype=float)
    out = BayesianRegression(alpha=0.05, prior_precision=1.0)(y, x)
    assert np.all(np.isfinite(out))
    assert abs(out[-1, 2]) < 50.0                   # slope stays bounded, not a blow-up


def test_parity_with_rolling_linear_regression_in_the_limit():
    from screamer import RollingLinearRegression
    rng = np.random.default_rng(5)
    n = 6000
    x = rng.standard_normal(n)
    y = 1.5 - 0.8 * x + rng.standard_normal(n) * 0.5
    br = BayesianRegression(alpha=1e-4, prior_precision=1e-6)(y, x)   # long memory, weak prior
    rlr = np.asarray(RollingLinearRegression(2000)(y, x))            # OLS over a long window
    # both estimate the same stationary line; compare end slope/intercept loosely
    assert abs(br[-1, 2] - rlr[-1, 0]) < 0.05        # slope
    assert abs(br[-1, 3] - rlr[-1, 1]) < 0.05        # intercept
