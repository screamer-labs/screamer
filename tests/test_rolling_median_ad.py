"""Tests for RollingMedianAD = median(|x - rolling_median|).

The robust (median-based) absolute deviation, in contrast to RollingMad which
is the mean absolute deviation. O(W) per step; validated against a plain-numpy
reference. Warmup, NaN "ignore", and the three start policies are covered here;
generic stream==batch / NaN-compliance coverage comes from param_cases.
"""
import numpy as np
import pytest

from screamer import RollingMedianAD


def _manual_median_ad(x, w, policy="strict"):
    """Reference for every start policy: median(|v - median(v)|) over the window.

    strict    -> NaN until the window is full
    expanding -> compute over the samples seen so far, from the first
    zero      -> window pre-filled with `w` zeros (missing counts as 0)
    NaN input is skipped (state untouched), emitting NaN.
    """
    out = np.full(len(x), np.nan)
    buf = [0.0] * w if policy == "zero" else []
    for i, v in enumerate(x):
        if np.isnan(v):
            out[i] = np.nan
            continue
        buf.append(v)
        if len(buf) > w:
            buf.pop(0)
        if policy == "strict" and len(buf) < w:
            out[i] = np.nan
            continue
        m = np.median(buf)
        out[i] = np.median(np.abs(np.array(buf) - m))
    return out


@pytest.mark.parametrize("w", [1, 2, 3, 5, 10, 20])
@pytest.mark.parametrize("policy", ["strict", "expanding", "zero"])
def test_matches_manual_numpy_reference(w, policy):
    rng = np.random.default_rng(w)
    x = rng.standard_normal(80)
    got = RollingMedianAD(w, start_policy=policy)(x)
    exp = _manual_median_ad(x, w, policy)
    np.testing.assert_allclose(got, exp, equal_nan=True, atol=1e-12)


def test_even_and_odd_window_median_convention():
    # Even window uses the average of the two central order statistics (numpy).
    x = np.array([0.0, 0.0, 10.0, 10.0])
    # window=4: median=5, deviations=[5,5,5,5], median-AD=5
    out = RollingMedianAD(4)(x)
    assert np.isnan(out[:3]).all()
    np.testing.assert_allclose(out[3], 5.0)


def test_constant_input_is_zero():
    out = RollingMedianAD(5)(np.full(20, 3.7))
    assert np.isnan(out[:4]).all()
    np.testing.assert_allclose(out[4:], 0.0, atol=1e-12)


def test_robust_to_a_single_spike():
    # One large spike in an otherwise constant window barely moves the median-AD,
    # unlike the mean absolute deviation.
    x = np.zeros(21)
    x[10] = 100.0
    out = RollingMedianAD(21)(x)
    # window of 21 zeros + one 100: median=0, deviations mostly 0 -> MAD=0.
    np.testing.assert_allclose(out[-1], 0.0, atol=1e-12)


@pytest.mark.parametrize("policy", ["strict", "expanding", "zero"])
def test_batch_equals_stream(policy):
    rng = np.random.default_rng(3)
    x = rng.standard_normal(200)
    batch = np.asarray(RollingMedianAD(15, start_policy=policy)(x))
    f = RollingMedianAD(15, start_policy=policy)
    stream = np.array([f(v) for v in x])
    np.testing.assert_allclose(batch, stream, equal_nan=True)


def test_nan_ignore_and_recovery():
    x = np.array([np.nan, np.nan, 1.0, 2.0, 3.0, 4.0, 5.0])
    out = np.asarray(RollingMedianAD(3)(x))
    assert np.isnan(out[0]) and np.isnan(out[1])
    # After the leading NaNs the window fills from finite samples and recovers.
    assert np.isfinite(out[-1])


def test_mid_stream_nan_is_skipped():
    base = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    withnan = np.array([1.0, 2.0, np.nan, 3.0, 4.0, 5.0, 6.0])
    clean_out = np.asarray(RollingMedianAD(3)(base))
    nan_out = np.asarray(RollingMedianAD(3)(withnan))
    # The NaN emits NaN and is skipped; every finite output matches the clean run
    # once the inserted NaN slot is removed (state is untouched by the NaN).
    assert np.isnan(nan_out[2])
    np.testing.assert_allclose(np.delete(nan_out, 2), clean_out, equal_nan=True)


def test_expanding_policy_has_no_warmup_nan():
    x = np.array([1.0, 5.0, 2.0, 8.0, 3.0])
    out = np.asarray(RollingMedianAD(4, start_policy="expanding")(x))
    assert np.isfinite(out).all()
    # first sample: single value -> deviation 0
    np.testing.assert_allclose(out[0], 0.0)


def test_zero_policy_has_no_warmup_nan():
    x = np.array([1.0, 2.0, 3.0])
    out = np.asarray(RollingMedianAD(5, start_policy="zero")(x))
    assert np.isfinite(out).all()


def test_two_dimensional_columns_independent():
    rng = np.random.default_rng(9)
    x = rng.standard_normal((50, 3))
    out = RollingMedianAD(7)(x)
    assert out.shape == (50, 3)
    for c in range(3):
        np.testing.assert_allclose(out[:, c], np.asarray(RollingMedianAD(7)(x[:, c])),
                                   equal_nan=True)


def test_reset_restores_initial_state():
    f = RollingMedianAD(4)
    a = np.asarray(f(np.array([1.0, 2.0, 3.0, 4.0, 5.0])))
    f.reset()
    b = np.asarray(f(np.array([1.0, 2.0, 3.0, 4.0, 5.0])))
    np.testing.assert_allclose(a, b, equal_nan=True)


def test_invalid_window_raises():
    with pytest.raises(Exception):
        RollingMedianAD(0)
    with pytest.raises(Exception):
        RollingMedianAD(-3)
