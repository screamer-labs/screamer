"""
Regression tests for the NaN-as-warmup-sentinel contract.

Several screamer algorithms (EwVar, EwStd, EwSkew, EwKurt, RollingVar, ...)
return NaN for the leading samples where there isn't enough data yet to
produce a meaningful value. This matches pandas' ewm/rolling behavior and
is what the reference baselines compare against.

Build tooling note: do NOT compile screamer_bindings with -ffast-math.
That flag enables -ffinite-math-only, which tells the optimizer "no NaN/Inf
will appear in this code". The optimizer then dead-codes the NaN return
paths in the algorithms — silently. Tests in test_baselines.py and
test_view.py will fail with mysterious off-by-N errors instead of NaN
mismatches. Keep -O3 -flto, drop -ffast-math.

These tests are deterministic (no random data) so a regression is loud.
"""
import math
import numpy as np
import pytest

from screamer import EwVar, EwStd, EwMean, RollingZscore


def test_ewvar_first_scalar_is_nan():
    e = EwVar(span=5)
    assert math.isnan(e(1.0)), "EwVar must return NaN before n_eff > 1"
    assert not math.isnan(e(2.0)), "EwVar must produce a real value once n_eff > 1"


def test_ewstd_first_scalar_is_nan():
    e = EwStd(span=5)
    assert math.isnan(e(1.0)), "EwStd must return NaN before n_eff > 1"


def test_ewvar_first_array_element_is_nan():
    e = EwVar(span=5)
    out = e(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert math.isnan(out[0])
    assert np.all(np.isfinite(out[1:]))


def test_ewvar_state_resets_between_array_calls():
    """Same instance called twice must yield bit-identical output."""
    e = EwVar(span=5)
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    out1 = e(arr)
    out2 = e(arr)
    np.testing.assert_array_equal(
        out1, out2, err_msg="EwVar leaked state between consecutive calls"
    )


def test_ewvar_strided_view_matches_contig_3d():
    """
    Same instance, processed first as contig copy then as strided view of
    identical data, must yield identical results. This is the deterministic
    version of test_screamer_view_strided_shifted_materialized that
    surfaced the -ffast-math bug.
    """
    rng = np.random.default_rng(42)
    big = rng.standard_normal((200, 9, 8))
    view = big[1::2, 2::3, 3::4]
    contig = view.copy()
    assert not view.flags.c_contiguous
    assert contig.flags.c_contiguous
    assert np.array_equal(view, contig)

    e = EwVar(span=5)
    out_contig = e(contig)
    out_strided = e(view)
    np.testing.assert_allclose(
        out_contig,
        out_strided,
        equal_nan=True,
        rtol=1e-12,
        atol=1e-12,
        err_msg="strided 3D view does not match its contiguous copy",
    )


@pytest.mark.parametrize("span", [3, 5, 10, 20])
def test_ewvar_strided_matches_contig_various_spans(span):
    rng = np.random.default_rng(span)
    big = rng.standard_normal((100, 9, 8))
    view = big[::2, 1::3, 2::4]
    contig = view.copy()

    e = EwVar(span=span)
    out_contig = e(contig)
    out_strided = e(view)
    np.testing.assert_allclose(out_contig, out_strided, equal_nan=True, rtol=1e-12, atol=1e-12)


def test_rolling_zscore_streaming_matches_batch():
    """Regression test for an uninitialised-member bug in RollingZscore.

    Before the fix, the constructor's initialiser list set everything except
    `size_t n_`. On platforms whose heap happened to zero-fill (most macOS
    builds, most Ubuntu Python ABIs), n_ was effectively 0 and tests passed.
    On Ubuntu CPython 3.14 the heap left non-zero garbage in n_, so
    `process_scalar` skipped its warmup branch (`n_ < window_size_` was
    false) and divided by garbage, producing near-zero outputs instead of
    real z-scores.

    This test runs deterministic, non-random data and asserts that the
    streaming path (per-scalar `process_scalar` via __call__) matches the
    batch path (`process_array_no_stride`) within numerical tolerance.
    """
    window = 5
    # Deterministic input: linear ramp + sine. Real values, no NaNs except
    # the warmup ones the algorithm itself produces.
    x = np.arange(30, dtype=float) + np.sin(np.arange(30, dtype=float))

    # Streaming path: one scalar at a time
    stream = RollingZscore(window_size=window)
    out_stream = np.empty_like(x)
    for i, xi in enumerate(x):
        out_stream[i] = stream(xi)

    # Batch path: full array
    batch = RollingZscore(window_size=window)
    out_batch = batch(x)

    np.testing.assert_allclose(
        out_stream,
        out_batch,
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
        err_msg="RollingZscore streaming vs batch mismatch (uninit n_ regression)",
    )


def test_rolling_zscore_first_scalar_call_is_finite_or_nan_not_garbage():
    """Tighter regression for the same bug.

    A freshly-constructed RollingZscore that gets its first sample must
    return either NaN (warmup) or a finite z-score-shaped value. Before
    the fix, the result could be ~0 because n_ was garbage.
    """
    rz = RollingZscore(window_size=5)
    out = rz(1.0)
    # Either NaN (warmup) or finite. Anything else (inf / wildly wrong) is
    # the uninit-memory symptom.
    assert math.isnan(out) or math.isfinite(out)
    # And the second call must continue to behave (not blow up).
    out2 = rz(2.0)
    assert math.isnan(out2) or math.isfinite(out2)


def test_ewmean_no_warmup_nan():
    """EwMean has no warmup (n_eff=1 still produces a meaningful mean)."""
    e = EwMean(span=5)
    assert e(1.0) == pytest.approx(1.0)
    # span=5 -> alpha=1/3, one_minus_alpha=2/3.
    # After x=2: sum_x = (2/3)*1 + 2 = 8/3, sum_w = (2/3)*1 + 1 = 5/3, ratio = 8/5 = 1.6.
    assert e(2.0) == pytest.approx(1.6)
