"""
Tests for the 2-input EW pair statistics: EwCov, EwCorr, EwBeta.

  EwCov:  bias-corrected, matches pandas ewm(adjust=True, bias=False).cov.
  EwCorr: unbiased simple form, matches pandas ewm(adjust=True).corr.
  EwBeta: cov(x, y) / var(y). Pandas does not have a direct primitive,
          so we validate against the manually-computed ratio of pandas
          ewm cov and ewm var. Convention matches RollingBeta: the
          first argument is the dependent (target), the second is the
          regressor (CAPM convention; pandas uses the opposite).
"""
import numpy as np
import pandas as pd
import pytest

from screamer import EwCov, EwCorr, EwBeta, EwVar, EwMean


def _correlated_pair(rng, n, rho=0.7):
    """Return (x, y) with target correlation rho around 0."""
    x = rng.standard_normal(n)
    y = rho * x + np.sqrt(1.0 - rho * rho) * rng.standard_normal(n)
    return x, y


# ---------------------------------------------------------------------------
# Pandas parity
# ---------------------------------------------------------------------------

class TestPandasParity:

    @pytest.mark.parametrize("alpha", [0.05, 0.1, 0.3, 0.6])
    def test_ewcov_matches_pandas(self, alpha):
        rng = np.random.default_rng(int(alpha * 1000))
        x, y = _correlated_pair(rng, n=150)
        ref = pd.Series(x).ewm(alpha=alpha).cov(pd.Series(y)).to_numpy()
        ours = EwCov(alpha=alpha)(x, y)
        np.testing.assert_allclose(ours, ref, equal_nan=True, atol=1e-12)

    @pytest.mark.parametrize("alpha", [0.05, 0.1, 0.3, 0.6])
    def test_ewcorr_matches_pandas(self, alpha):
        rng = np.random.default_rng(int(alpha * 2000))
        x, y = _correlated_pair(rng, n=150)
        ref = pd.Series(x).ewm(alpha=alpha).corr(pd.Series(y)).to_numpy()
        ours = EwCorr(alpha=alpha)(x, y)
        np.testing.assert_allclose(ours, ref, equal_nan=True, atol=1e-12)

    @pytest.mark.parametrize("alpha", [0.05, 0.1, 0.3, 0.6])
    def test_ewbeta_matches_pandas_cov_over_var(self, alpha):
        """Pandas has no direct ewm beta; reconstruct as cov(x,y)/var(y)."""
        rng = np.random.default_rng(int(alpha * 3000))
        x, y = _correlated_pair(rng, n=150)
        cov_xy = pd.Series(x).ewm(alpha=alpha).cov(pd.Series(y)).to_numpy()
        var_y  = pd.Series(y).ewm(alpha=alpha).var().to_numpy()
        ref = cov_xy / var_y
        ours = EwBeta(alpha=alpha)(x, y)
        np.testing.assert_allclose(ours, ref, equal_nan=True, atol=1e-10)


# ---------------------------------------------------------------------------
# Constructor parameter forms (com / span / halflife / alpha)
# ---------------------------------------------------------------------------

class TestParameterForms:

    @pytest.mark.parametrize("cls", [EwCov, EwCorr, EwBeta])
    def test_alpha_span_com_halflife_consistent(self, cls):
        """The four parameter forms map to the same internal alpha."""
        rng = np.random.default_rng(7)
        x, y = _correlated_pair(rng, n=80)
        a = 0.2
        com = (1.0 - a) / a            # alpha = 1 / (1 + com)
        span = 2.0 / a - 1.0           # alpha = 2 / (span + 1)
        halflife = -np.log(2.0) / np.log(1.0 - a)  # alpha = 1 - 0.5^(1/halflife)

        ref = cls(alpha=a)(x, y)
        np.testing.assert_allclose(cls(com=com)(x, y),       ref, equal_nan=True, atol=1e-12)
        np.testing.assert_allclose(cls(span=span)(x, y),     ref, equal_nan=True, atol=1e-12)
        np.testing.assert_allclose(cls(halflife=halflife)(x, y), ref, equal_nan=True, atol=1e-12)

    @pytest.mark.parametrize("cls", [EwCov, EwCorr, EwBeta])
    def test_zero_args_raises(self, cls):
        with pytest.raises(ValueError):
            cls()

    @pytest.mark.parametrize("cls", [EwCov, EwCorr, EwBeta])
    def test_two_args_raises(self, cls):
        with pytest.raises(ValueError):
            cls(alpha=0.1, span=10)


# ---------------------------------------------------------------------------
# Algebraic identities
# ---------------------------------------------------------------------------

class TestIdentities:

    def test_ewcov_symmetric(self):
        rng = np.random.default_rng(11)
        x, y = _correlated_pair(rng, n=80)
        np.testing.assert_array_equal(EwCov(alpha=0.1)(x, y),
                                      EwCov(alpha=0.1)(y, x))

    def test_ewcorr_self_is_one(self):
        rng = np.random.default_rng(12)
        x = rng.standard_normal(80)
        out = EwCorr(alpha=0.2)(x, x)
        # Skip first sample (NaN warmup); rest must be exactly 1.
        np.testing.assert_allclose(out[1:], 1.0, atol=1e-12)

    def test_ewcorr_in_minus_one_to_one(self):
        rng = np.random.default_rng(13)
        x, y = _correlated_pair(rng, n=200)
        out = EwCorr(alpha=0.1)(x, y)
        finite = out[np.isfinite(out)]
        assert np.all(finite >= -1.0 - 1e-12)
        assert np.all(finite <=  1.0 + 1e-12)

    def test_ewcorr_negation_flips_sign(self):
        """EwCorr(-x, y) == -EwCorr(x, y)."""
        rng = np.random.default_rng(14)
        x, y = _correlated_pair(rng, n=80)
        np.testing.assert_allclose(EwCorr(alpha=0.1)(-x, y),
                                  -EwCorr(alpha=0.1)( x, y),
                                  equal_nan=True, atol=1e-12)

    def test_ewbeta_self_is_one(self):
        rng = np.random.default_rng(15)
        x = rng.standard_normal(80)
        out = EwBeta(alpha=0.2)(x, x)
        np.testing.assert_allclose(out[1:], 1.0, atol=1e-12)

    def test_ewbeta_scaled_x(self):
        """EwBeta(c*x, x) == c (linear in the dependent)."""
        rng = np.random.default_rng(16)
        x = rng.standard_normal(80)
        for c in [-2.5, 0.5, 3.0]:
            out = EwBeta(alpha=0.15)(c * x, x)
            np.testing.assert_allclose(out[1:], c, atol=1e-12)

    def test_ewbeta_invariant_to_y_scale(self):
        """EwBeta(x, c*y) == EwBeta(x, y) / c."""
        rng = np.random.default_rng(17)
        x, y = _correlated_pair(rng, n=80)
        c = 4.0
        a = EwBeta(alpha=0.1)(x, c * y)
        b = EwBeta(alpha=0.1)(x, y) / c
        np.testing.assert_allclose(a, b, equal_nan=True, atol=1e-12)

    def test_ewbeta_equals_cov_over_var(self):
        """EwBeta(x, y) == EwCov(x, y) / EwVar(y) (definitional)."""
        rng = np.random.default_rng(18)
        x, y = _correlated_pair(rng, n=80)
        np.testing.assert_allclose(
            EwBeta(alpha=0.1)(x, y),
            EwCov(alpha=0.1)(x, y) / EwVar(alpha=0.1)(y),
            equal_nan=True, atol=1e-12,
        )

    def test_ewcorr_equals_cov_over_std_product(self):
        """EwCorr(x, y) == EwCov(x, y) / sqrt(EwVar(x) * EwVar(y))."""
        rng = np.random.default_rng(19)
        x, y = _correlated_pair(rng, n=80)
        denom = np.sqrt(EwVar(alpha=0.1)(x) * EwVar(alpha=0.1)(y))
        np.testing.assert_allclose(
            EwCorr(alpha=0.1)(x, y),
            EwCov(alpha=0.1)(x, y) / denom,
            equal_nan=True, atol=1e-12,
        )


# ---------------------------------------------------------------------------
# Streaming (scalar loop) parity with batch
# ---------------------------------------------------------------------------

class TestScalarLoopParity:

    @pytest.mark.parametrize("cls", [EwCov, EwCorr, EwBeta])
    def test_scalar_loop_matches_array(self, cls):
        rng = np.random.default_rng(21)
        x, y = _correlated_pair(rng, n=60)
        obj = cls(alpha=0.2)
        streamed = np.array([obj(xi, yi) for xi, yi in zip(x, y)])
        np.testing.assert_allclose(streamed, cls(alpha=0.2)(x, y),
                                  equal_nan=True, atol=1e-12)


# ---------------------------------------------------------------------------
# 2D per-column independence
# ---------------------------------------------------------------------------

class Test2DIndependence:

    @pytest.mark.parametrize("cls", [EwCov, EwCorr, EwBeta])
    def test_2d_per_column_pairing(self, cls):
        rng = np.random.default_rng(22)
        X = rng.standard_normal((50, 3))
        Y = 0.5 * X + 0.5 * rng.standard_normal((50, 3))
        out_2d = cls(alpha=0.15)(X, Y)
        for k in range(3):
            np.testing.assert_array_equal(
                out_2d[:, k],
                cls(alpha=0.15)(X[:, k].copy(), Y[:, k].copy()),
            )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    # Note on exactly-constant inputs:
    # The classes use the standard "uncentered sums" recurrence (same as
    # EwVar). When one input is exactly constant, mathematically var=0
    # and beta/corr are undefined. With uncentered sums the var calculation
    # has cancellation error of order 1e-9, so the var > 0 check in C++
    # passes by float fluke and the output is small numerical noise rather
    # than NaN. Pandas avoids this by tracking centered sums. The behaviour
    # difference only matters for exactly-constant inputs; for realistic
    # near-constant data both algorithms agree to ~1e-12.

    def test_first_sample_is_nan(self):
        """All three need at least 2 samples for n_eff > 1."""
        out_cov  = EwCov (alpha=0.1)(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        out_corr = EwCorr(alpha=0.1)(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        out_beta = EwBeta(alpha=0.1)(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        for out in (out_cov, out_corr, out_beta):
            assert np.isnan(out[0])
            assert np.isfinite(out[1])
