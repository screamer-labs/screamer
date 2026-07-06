"""Tests for the Hampel filter (canonical causal median+MAD despiker).

Validated against a faithful plain-Python reference that mirrors the causal
trailing-window rule, including replacing a detected outlier with the median in
the working window so a burst of spikes cannot inflate the scale.
"""
import numpy as np
import pytest

from screamer import Hampel

MAD_TO_STD = 1.4826


def _ref_hampel(x, w, n_sigma, output=0, policy="strict"):
    buf = [0.0] * w if policy == "zero" else []
    out = np.full(len(x), np.nan)
    for i, v in enumerate(x):
        if np.isnan(v):
            out[i] = np.nan
            continue
        buf.append(v)
        if len(buf) > w:
            buf.pop(0)
        if policy == "strict" and len(buf) < w:  # strict warmup
            out[i] = np.nan
            continue
        m = np.median(buf)
        scale = MAD_TO_STD * np.median(np.abs(np.array(buf) - m))
        is_outlier = scale > 0.0 and abs(v - m) > n_sigma * scale
        if is_outlier:
            buf[-1] = m  # keep the window clean for later samples
            out[i] = 1.0 if output == 1 else (np.nan if output == 2 else m)
        else:
            out[i] = 0.0 if output == 1 else v
    return out


@pytest.mark.parametrize("w", [1, 5, 11, 21])
@pytest.mark.parametrize("n_sigma", [2.0, 3.0])
@pytest.mark.parametrize("output", [0, 1, 2])
@pytest.mark.parametrize("policy", ["strict", "expanding", "zero"])
def test_matches_reference(w, n_sigma, output, policy):
    rng = np.random.default_rng(w + output)
    x = rng.standard_normal(200)
    x[30] += 8.0
    x[95] -= 7.0
    x[160] += 6.0
    got = np.asarray(Hampel(w, n_sigma, output, policy)(x))
    exp = _ref_hampel(x, w, n_sigma, output, policy)
    np.testing.assert_allclose(got, exp, equal_nan=True, atol=1e-12)


def test_spikes_flagged_and_replaced():
    rng = np.random.default_rng(1)
    n = 300
    base = np.sin(np.linspace(0, 8, n))
    x = base + 0.03 * rng.standard_normal(n)
    spikes = [40, 130, 250]
    for s in spikes:
        x[s] += 6.0
    flag = np.asarray(Hampel(21, 3.0, 1)(x))
    for s in spikes:
        assert flag[s] == 1.0
    cleaned = np.asarray(Hampel(21, 3.0, 0)(x))
    # cleaned output at the spike is far closer to the smooth base than the spike
    for s in spikes:
        assert abs(cleaned[s] - base[s]) < abs(x[s] - base[s])


def test_clean_data_mostly_unflagged():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(1000)
    flag = np.asarray(Hampel(21, 3.0, 1)(x))
    # A 3-sigma robust threshold flags only a small fraction of clean normal data.
    assert np.nansum(flag) < 0.05 * len(x)


def test_constant_window_flags_nothing():
    # MAD == 0 -> no scale -> never flag (documented degenerate case).
    x = np.concatenate([np.full(30, 5.0), [99.0], np.full(10, 5.0)])
    flag = np.asarray(Hampel(21, 3.0, 1)(x))
    assert np.nansum(flag) == 0.0


def test_output_two_marks_outliers_nan():
    x = 0.1 * np.random.default_rng(0).standard_normal(60)
    x[40] += 20.0
    out = np.asarray(Hampel(21, 3.0, 2)(x))
    assert np.isnan(out[40])                 # outlier -> NaN
    np.testing.assert_allclose(out[41], x[41])  # a clean sample passes through


@pytest.mark.parametrize("policy", ["strict", "expanding", "zero"])
def test_batch_equals_stream(policy):
    rng = np.random.default_rng(4)
    x = rng.standard_normal(300)
    x[100] += 9.0
    for out_mode in (0, 1, 2):
        batch = np.asarray(Hampel(21, 3.0, out_mode, policy)(x))
        f = Hampel(21, 3.0, out_mode, policy)
        stream = np.array([f(v) for v in x])
        np.testing.assert_allclose(batch, stream, equal_nan=True)


def test_nan_ignore_and_recovery():
    x = np.concatenate([[np.nan, np.nan, np.nan], np.zeros(60)])
    out = np.asarray(Hampel(21, 3.0)(x))
    assert np.isnan(out[:3]).all()
    assert np.isfinite(out[-1])


def test_start_policies_run():
    rng = np.random.default_rng(5)
    x = rng.standard_normal(50)
    for policy in ("strict", "expanding", "zero"):
        out = np.asarray(Hampel(10, 3.0, 0, policy)(x))
        assert out.shape == x.shape
    # expanding / zero emit a finite value from the first sample
    assert np.isfinite(np.asarray(Hampel(10, 3.0, 0, "expanding")(x))[0])
    assert np.isfinite(np.asarray(Hampel(10, 3.0, 0, "zero")(x))[0])


def test_two_dimensional_columns_independent():
    rng = np.random.default_rng(6)
    x = rng.standard_normal((80, 3))
    x[40, 1] += 10.0
    out = Hampel(21, 3.0, 1)(x)
    assert out.shape == (80, 3)
    assert out[40, 1] == 1.0
    for c in range(3):
        np.testing.assert_allclose(out[:, c], np.asarray(Hampel(21, 3.0, 1)(x[:, c])),
                                   equal_nan=True)


def test_reset_restores_initial_state():
    f = Hampel(11, 3.0)
    x = np.random.default_rng(7).standard_normal(40)
    a = np.asarray(f(x))
    f.reset()
    b = np.asarray(f(x))
    np.testing.assert_allclose(a, b, equal_nan=True)


def test_invalid_arguments_raise():
    with pytest.raises(Exception):
        Hampel(0)
    with pytest.raises(Exception):
        Hampel(20, 0.0)
    with pytest.raises(Exception):
        Hampel(20, 3.0, 5)
