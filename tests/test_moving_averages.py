"""
Tests for the composite moving averages: DEMA, TEMA, TRIMA, HullMA.

All four are *defined* as compositions of existing primitives:

  DEMA(x)   = 2*EMA(x) - EMA(EMA(x))
  TEMA(x)   = 3*EMA(x) - 3*EMA(EMA(x)) + EMA(EMA(EMA(x)))
  TRIMA(x, n)  = SMA(SMA(x, n_inner), n_outer)
  HullMA(x, n) = WMA(2*WMA(x, n/2) - WMA(x, n), sqrt(n))

These tests validate the dedicated implementation against the manual
composition built from EwMean / RollingMean / WMA -- the dedicated
class is ONLY a convenience wrapper, so it must match bit-for-bit
(post-warmup, where applicable).
"""
import numpy as np
import pandas as pd
import pytest

from screamer import (
    DEMA, TEMA, TRIMA, HullMA,
    EwMean, RollingMean, WMA,
)


# ---------------------------------------------------------------------------
# DEMA / TEMA: composed from EwMean
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("span", [5, 10, 20, 50])
def test_dema_matches_composition(span):
    rng = np.random.default_rng(span)
    x = rng.standard_normal(150)
    e1 = EwMean(span=span)(x)
    e2 = EwMean(span=span)(e1)
    ref = 2 * e1 - e2
    np.testing.assert_allclose(DEMA(span=span)(x), ref, atol=1e-12)


@pytest.mark.parametrize("span", [5, 10, 20, 50])
def test_tema_matches_composition(span):
    rng = np.random.default_rng(span + 100)
    x = rng.standard_normal(150)
    e1 = EwMean(span=span)(x)
    e2 = EwMean(span=span)(e1)
    e3 = EwMean(span=span)(e2)
    ref = 3 * e1 - 3 * e2 + e3
    np.testing.assert_allclose(TEMA(span=span)(x), ref, atol=1e-12)


def test_dema_constant_input_is_constant():
    """For constant input c, EMA = c, so DEMA = 2c - c = c."""
    out = DEMA(span=5)(np.full(30, 4.2))
    np.testing.assert_allclose(out, 4.2, atol=1e-12)


def test_tema_constant_input_is_constant():
    out = TEMA(span=5)(np.full(30, 4.2))
    np.testing.assert_allclose(out, 4.2, atol=1e-12)


def test_dema_first_sample_equals_input():
    """DEMA[0] = 2*x[0] - x[0] = x[0]."""
    out = DEMA(span=10)(np.array([1.7, 0.0, 0.0]))
    assert out[0] == 1.7


def test_tema_first_sample_equals_input():
    """TEMA[0] = 3*x[0] - 3*x[0] + x[0] = x[0] (up to one ULP from the
    3-term linear combination)."""
    out = TEMA(span=10)(np.array([1.7, 0.0, 0.0]))
    assert out[0] == pytest.approx(1.7, abs=1e-15)


@pytest.mark.parametrize("cls", [DEMA, TEMA])
def test_constructor_parameter_forms_consistent(cls):
    """All four parameter forms must produce identical output."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal(60)
    a = 0.2
    com = (1.0 - a) / a
    span = 2.0 / a - 1.0
    halflife = -np.log(2.0) / np.log(1.0 - a)
    ref = cls(alpha=a)(x)
    np.testing.assert_allclose(cls(com=com)(x), ref, atol=1e-12)
    np.testing.assert_allclose(cls(span=span)(x), ref, atol=1e-12)
    np.testing.assert_allclose(cls(halflife=halflife)(x), ref, atol=1e-12)


@pytest.mark.parametrize("cls", [DEMA, TEMA])
def test_zero_args_raises(cls):
    with pytest.raises(ValueError):
        cls()


@pytest.mark.parametrize("cls", [DEMA, TEMA])
def test_two_args_raises(cls):
    with pytest.raises(ValueError):
        cls(alpha=0.1, span=10)


# ---------------------------------------------------------------------------
# TRIMA: composed from RollingMean
# ---------------------------------------------------------------------------

def _trima_inner_outer(n):
    """TA-Lib convention for the two SMA window sizes."""
    if n % 2 == 1:
        return (n + 1) // 2, (n + 1) // 2
    return n // 2 + 1, n // 2


@pytest.mark.parametrize("n", [3, 5, 6, 7, 10, 11, 20])
def test_trima_matches_composition(n):
    rng = np.random.default_rng(n)
    x = rng.standard_normal(80)
    n_inner, n_outer = _trima_inner_outer(n)
    inner = RollingMean(n_inner, "expanding")(x)
    outer = RollingMean(n_outer, "expanding")(inner)
    ours = TRIMA(n)(x)
    # Post-warmup: TRIMA emits NaN for first n-1 samples; the inner+outer
    # composition with "expanding" returns values from t=0. They must
    # agree from sample n-1 onward.
    np.testing.assert_allclose(ours[n - 1:], outer[n - 1:], atol=1e-12)


@pytest.mark.parametrize("n", [3, 5, 6, 7, 10, 20])
def test_trima_warmup_is_nan(n):
    rng = np.random.default_rng(n + 200)
    x = rng.standard_normal(50)
    out = TRIMA(n)(x)
    assert np.all(np.isnan(out[:n - 1]))
    assert np.all(np.isfinite(out[n - 1:]))


def test_trima_constant_input_is_constant():
    out = TRIMA(7)(np.full(30, 3.7))
    np.testing.assert_allclose(out[6:], 3.7, atol=1e-12)


def test_trima_inner_outer_split():
    """Sanity: n_inner + n_outer - 1 == n for both even and odd n."""
    for n in range(2, 21):
        ni, no = _trima_inner_outer(n)
        assert ni + no - 1 == n


# ---------------------------------------------------------------------------
# HullMA: composed from WMA
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [4, 9, 10, 16, 25])
def test_hull_ma_matches_composition(n):
    rng = np.random.default_rng(n)
    x = rng.standard_normal(120)
    n_half = n // 2
    n_sqrt = int(np.sqrt(n))
    w_half = WMA(n_half, "expanding")(x)
    w_full = WMA(n, "expanding")(x)
    diff = 2 * w_half - w_full
    w_outer = WMA(n_sqrt, "expanding")(diff)
    ours = HullMA(n)(x)
    warmup = n + n_sqrt - 1
    np.testing.assert_allclose(ours[warmup - 1:], w_outer[warmup - 1:], atol=1e-12)


@pytest.mark.parametrize("n", [4, 9, 10, 16, 25])
def test_hull_ma_warmup_is_nan(n):
    rng = np.random.default_rng(n + 300)
    x = rng.standard_normal(80)
    out = HullMA(n)(x)
    n_sqrt = int(np.sqrt(n))
    warmup = n + n_sqrt - 1
    assert np.all(np.isnan(out[:warmup - 1]))
    assert np.all(np.isfinite(out[warmup - 1:]))


def test_hull_ma_constant_input_is_constant():
    out = HullMA(9)(np.full(30, 4.2))
    n_sqrt = int(np.sqrt(9))
    warmup = 9 + n_sqrt - 1
    np.testing.assert_allclose(out[warmup - 1:], 4.2, atol=1e-12)


def test_hull_ma_rejects_small_window():
    with pytest.raises(ValueError):
        HullMA(3)


# ---------------------------------------------------------------------------
# Streaming / 2D parity (shared across all four)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ctor", [
    lambda: DEMA(span=10),
    lambda: TEMA(span=10),
    lambda: TRIMA(7),
    lambda: HullMA(9),
])
def test_scalar_loop_matches_array(ctor):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(50)
    obj = ctor()
    streamed = np.array([obj(v) for v in x])
    np.testing.assert_allclose(streamed, ctor()(x), equal_nan=True, atol=1e-12)


@pytest.mark.parametrize("ctor", [
    lambda: DEMA(span=10),
    lambda: TEMA(span=10),
    lambda: TRIMA(7),
    lambda: HullMA(9),
])
def test_2d_per_column_independence(ctor):
    rng = np.random.default_rng(1)
    X = rng.standard_normal((40, 3))
    out_2d = ctor()(X)
    for k in range(X.shape[1]):
        np.testing.assert_allclose(
            out_2d[:, k], ctor()(X[:, k].copy()),
            equal_nan=True, atol=1e-12,
        )


@pytest.mark.parametrize("ctor", [
    lambda: DEMA(span=10),
    lambda: TEMA(span=10),
    lambda: TRIMA(7),
    lambda: HullMA(9),
])
def test_reset_clears_history(ctor):
    rng = np.random.default_rng(2)
    x = rng.standard_normal(40)
    obj = ctor()
    first = np.array([obj(v) for v in x])
    obj.reset()
    second = np.array([obj(v) for v in x])
    np.testing.assert_array_equal(first, second)
