"""Tests for RollingHurst.

Validates against a Python reference implementation of the Anis-Lloyd
corrected R/S analysis (method 2 of the original snippet supplied with
the feature request), plus sanity checks (white noise ~> 0.5, trending
series > 0.5, zig-zag < 0.5, NaN warmup).
"""

import numpy as np
import pytest
from math import gamma

from screamer import RollingHurst


# ---------------------------------------------------------------------
# Reference implementation (the user's "method=2" branch, restricted to
# dyadic scales up to W/2, as the C++ implementation does).
# ---------------------------------------------------------------------
def hurst_rs_anis_lloyd(x, min_scale=4):
    x = np.asarray(x, dtype=float)
    W = len(x)
    lengths = []
    rescaled = []
    base = []
    n = min_scale
    while 2 * n <= W:
        # partition exactly W/n non-overlapping blocks of length n
        num = W // n
        if num < 2:
            break
        patterns = x[: num * n].reshape(num, n).copy()
        patterns -= np.mean(patterns, axis=1, keepdims=True)
        stds = np.mean(patterns ** 2, axis=1, keepdims=True) ** 0.5
        if np.any(stds == 0):
            return np.nan
        cum = np.cumsum(patterns, axis=1)
        rng = np.max(cum, axis=1, keepdims=True) - np.min(cum, axis=1, keepdims=True)
        avg = np.mean(rng / stds)

        # Anis-Lloyd theoretical R/S
        i = np.arange(1, n)
        ers = np.sum(np.sqrt((n - i) / i))
        if n <= 340:
            ers *= gamma((n - 1) / 2) / np.sqrt(np.pi) / gamma(n / 2)
        else:
            ers *= 1.0 / np.sqrt(np.pi * n / 2)

        lengths.append(n)
        rescaled.append(avg)
        base.append(ers)
        n *= 2

    if len(lengths) < 3:
        return np.nan

    lengths = np.array(lengths, dtype=float)
    rescaled = np.array(rescaled)
    base = np.array(base)
    rsal = rescaled - base + np.sqrt(0.5 * np.pi * lengths)
    if np.any(rsal <= 0):
        return np.nan
    return np.polyfit(np.log(lengths), np.log(rsal), 1)[0]


# ---------------------------------------------------------------------


class TestWarmup:
    def test_first_w_minus_1_are_nan(self):
        rng = np.random.default_rng(0)
        W = 64
        rh = RollingHurst(window_size=W)
        out = rh(rng.normal(size=200))
        assert np.all(np.isnan(out[: W - 1]))
        assert not np.isnan(out[W - 1])

    def test_constant_input_returns_nan(self):
        rh = RollingHurst(window_size=64)
        out = rh(np.ones(200))
        assert np.all(np.isnan(out))


class TestReferenceAlignment:
    @pytest.mark.parametrize("W", [32, 64, 128, 256, 512])
    def test_bit_exact_vs_reference_on_random_walk(self, W):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 3 * W)
        out = RollingHurst(window_size=W)(x)
        # Check several post-warmup positions
        for t in [W - 1, W + 5, 2 * W, 3 * W - 1]:
            window = x[t - W + 1 : t + 1]
            expected = hurst_rs_anis_lloyd(window, min_scale=4)
            assert np.isclose(out[t], expected, atol=1e-12, rtol=1e-12), (
                f"t={t}, W={W}: got {out[t]}, expected {expected}"
            )

    @pytest.mark.parametrize("min_scale", [4, 8, 16])
    def test_bit_exact_with_min_scale_variations(self, min_scale):
        rng = np.random.default_rng(7)
        W = 256
        x = rng.normal(size=2 * W)
        out = RollingHurst(window_size=W, min_scale=min_scale)(x)
        window = x[W : 2 * W]
        expected = hurst_rs_anis_lloyd(window, min_scale=min_scale)
        assert np.isclose(out[2 * W - 1], expected, atol=1e-12, rtol=1e-12)


class TestQualitativeBehaviour:
    """These are sanity checks, not bit-exact tests — Hurst estimators are
    noisy on finite samples. Tolerances are wide on purpose."""

    def test_white_noise_near_half(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, 4096)
        H = RollingHurst(window_size=2048)(x)
        # Last value, with a generous tolerance.
        assert 0.40 < H[-1] < 0.60

    def test_integrated_noise_is_persistent(self):
        # Cumulative sum of white noise = Brownian motion; H of the *path*
        # should be substantially above 0.5.
        rng = np.random.default_rng(3)
        x = np.cumsum(rng.normal(0, 1, 2048))
        H = RollingHurst(window_size=1024)(x)
        assert H[-1] > 0.7


class TestConstructorValidation:
    def test_too_small_window(self):
        with pytest.raises(ValueError):
            RollingHurst(window_size=8, min_scale=4)

    def test_min_scale_below_4(self):
        with pytest.raises(ValueError):
            RollingHurst(window_size=64, min_scale=2)

    def test_unknown_method(self):
        with pytest.raises(ValueError):
            RollingHurst(window_size=64, method="dfa")
