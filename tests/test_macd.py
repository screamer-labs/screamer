"""
Tests for MACD: (macd, signal, histogram) = 1->3.

Pure composition of three EwMean instances. Validation strategy:

  1. Pandas composition reference (adjust=True, same as our EwMean) --
     must be bit-equal.
  2. Manual numpy reference re-deriving the three EMAs from scratch --
     same expectation.
  3. Algebraic invariants (histogram == macd - signal; constant input
     -> all-zero output; etc.).

TA-Lib alignment is covered in tests/test_third_party_alignment.py
where it appears as a documented divergence (TA-Lib uses adjust=False
with an SMA seed; we use pandas's adjust=True everywhere).
"""
import numpy as np
import pandas as pd
import pytest

from screamer import MACD


def ref_macd_pandas(x, fast=12, slow=26, signal=9):
    """Composition reference using pandas's adjust=True EMA (matches our EwMean)."""
    s = pd.Series(x)
    ema_fast = s.ewm(span=fast, adjust=True).mean()
    ema_slow = s.ewm(span=slow, adjust=True).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=True).mean()
    hist = macd - sig
    return macd.to_numpy(), sig.to_numpy(), hist.to_numpy()


# ---------------------------------------------------------------------------
# Composition reference (the strict pin)
# ---------------------------------------------------------------------------

class TestPandasComposition:

    @pytest.mark.parametrize("fast,slow,signal", [
        (12, 26, 9),     # TA-Lib / charting standard
        (5,  35, 5),     # short fast, long slow
        (3,  10, 4),     # responsive
    ])
    def test_three_outputs_match_pandas(self, fast, slow, signal):
        rng = np.random.default_rng(fast * 100 + slow + signal)
        x = rng.standard_normal(200).cumsum()
        out = MACD(fast, slow, signal)(x)
        m_ref, s_ref, h_ref = ref_macd_pandas(x, fast, slow, signal)
        np.testing.assert_allclose(out[:, 0], m_ref, atol=1e-12)
        np.testing.assert_allclose(out[:, 1], s_ref, atol=1e-12)
        np.testing.assert_allclose(out[:, 2], h_ref, atol=1e-12)


# ---------------------------------------------------------------------------
# Algebraic invariants
# ---------------------------------------------------------------------------

class TestInvariants:

    def test_histogram_equals_macd_minus_signal(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(80).cumsum()
        out = MACD()(x)
        np.testing.assert_allclose(out[:, 2], out[:, 0] - out[:, 1], atol=1e-12)

    def test_constant_input_all_zero(self):
        """EMA of a constant is the constant. So both EMAs collapse to c,
        macd = 0, signal = EMA(0) = 0, histogram = 0."""
        out = MACD()(np.full(50, 3.7))
        np.testing.assert_allclose(out, 0.0, atol=1e-12)

    def test_first_sample_macd_is_zero(self):
        """At t=0, EMA(span=anything) returns x[0]. So macd[0] = x[0] - x[0] = 0.
        signal[0] = EMA of macd at t=0 = 0. histogram = 0."""
        out = MACD()(np.array([5.0, 7.0, 9.0]))
        np.testing.assert_allclose(out[0], 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestConstructor:

    def test_defaults_match_charting_standard(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(50)
        np.testing.assert_array_equal(MACD()(x), MACD(12, 26, 9)(x))

    def test_fast_ge_slow_raises(self):
        with pytest.raises(ValueError):
            MACD(fast=20, slow=20)
        with pytest.raises(ValueError):
            MACD(fast=30, slow=20)

    def test_zero_period_raises(self):
        with pytest.raises(ValueError):
            MACD(fast=0, slow=10, signal=5)
        with pytest.raises(ValueError):
            MACD(fast=5, slow=10, signal=0)


# ---------------------------------------------------------------------------
# Dispatcher paths (1->3 follows the same rules as BollingerBands / RollingMinMax)
# ---------------------------------------------------------------------------

class TestDispatcher:

    def test_scalar_returns_tuple(self):
        out = MACD()(0.0)
        assert isinstance(out, tuple) and len(out) == 3

    def test_1d_array_shape(self):
        out = MACD()(np.arange(50.0))
        assert out.shape == (50, 3)

    def test_2d_array_shape(self):
        rng = np.random.default_rng(2)
        X = rng.standard_normal((40, 3))
        out = MACD()(X)
        assert out.shape == (40, 3, 3)

    def test_2d_per_column_independence(self):
        rng = np.random.default_rng(3)
        X = rng.standard_normal((50, 4)).cumsum(axis=0)
        out_2d = MACD()(X)
        for k in range(X.shape[1]):
            ref = MACD()(X[:, k].copy())
            np.testing.assert_allclose(out_2d[:, k, :], ref,
                                       equal_nan=True, atol=1e-12)


# ---------------------------------------------------------------------------
# Cross-mode parity
# ---------------------------------------------------------------------------

class TestParity:

    def test_scalar_loop_matches_array(self):
        rng = np.random.default_rng(4)
        x = rng.standard_normal(80).cumsum()
        obj = MACD()
        streamed = np.array([obj(v) for v in x])
        np.testing.assert_allclose(streamed, MACD()(x),
                                   equal_nan=True, atol=1e-12)

    def test_reset_clears_history(self):
        rng = np.random.default_rng(5)
        x = rng.standard_normal(40).cumsum()
        obj = MACD()
        first = obj(x)
        obj.reset()
        second = obj(x)
        np.testing.assert_array_equal(first, second)
