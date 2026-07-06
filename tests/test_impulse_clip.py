"""Tests for ImpulseClip (causal difference-domain impulse remover).

Validated against a faithful plain-Python reference. Also pins the two
documented consequences of zero-latency detection: an isolated spike flags its
return sample too (two replacements), and a genuine level shift keeps its body
(only the onset sample is clipped).
"""
import numpy as np
import pytest

from screamer import ImpulseClip

MAD_TO_STD = 1.4826


def _ref_impulse(x, w, n_sigma, output=0):
    vbuf, dbuf = [], []
    prev = None
    out = np.full(len(x), np.nan)
    for i, v in enumerate(x):
        if np.isnan(v):
            out[i] = np.nan
            continue
        if prev is None:
            vbuf.append(v)
            if len(vbuf) > w:
                vbuf.pop(0)
            prev = v
            out[i] = np.nan  # first sample: no jump yet
            continue
        diff = v - prev
        prev = v  # raw previous value
        vbuf.append(v)
        dbuf.append(diff)
        if len(vbuf) > w:
            vbuf.pop(0)
        if len(dbuf) > w:
            dbuf.pop(0)
        if len(dbuf) < w:  # strict warmup on the differences
            out[i] = np.nan
            continue
        scale = MAD_TO_STD * np.median(np.abs(np.array(dbuf) - np.median(dbuf)))
        is_outlier = scale > 0.0 and abs(diff) > n_sigma * scale
        if is_outlier:
            m = np.median(vbuf)
            vbuf[-1] = m
            out[i] = 1.0 if output == 1 else (np.nan if output == 2 else m)
        else:
            out[i] = 0.0 if output == 1 else v
    return out


@pytest.mark.parametrize("w", [5, 11, 21])
@pytest.mark.parametrize("n_sigma", [3.0, 4.0])
@pytest.mark.parametrize("output", [0, 1, 2])
def test_matches_reference(w, n_sigma, output):
    rng = np.random.default_rng(w + output)
    t = np.linspace(0, 6 * np.pi, 220)
    x = np.sin(t) + 0.2 * rng.standard_normal(t.size)
    x[60] += 5.0
    x[130] -= 4.0
    got = np.asarray(ImpulseClip(w, n_sigma, output)(x))
    exp = _ref_impulse(x, w, n_sigma, output)
    np.testing.assert_allclose(got, exp, equal_nan=True, atol=1e-12)


def test_spikes_removed_on_oscillating_signal():
    rng = np.random.default_rng(7)
    t = np.linspace(0, 6 * np.pi, 400)
    clean = np.sin(t) + 0.35 * np.sin(2.9 * t + 0.6)
    x = clean + 0.30 * rng.standard_normal(t.size)
    hit = rng.choice(400, 14, replace=False)
    x[hit] += rng.choice([-1.0, 1.0], hit.size) * rng.uniform(2.5, 4.0, hit.size)
    cleaned = np.asarray(ImpulseClip(31, 4.0)(x))
    # despiking reduces the error to the clean signal versus the raw spiky input
    raw_rmse = np.sqrt(np.nanmean((x - clean) ** 2))
    clean_rmse = np.sqrt(np.nanmean((cleaned - clean) ** 2))
    assert clean_rmse < raw_rmse


def test_isolated_spike_flags_return_sample():
    # Documented zero-latency behavior: a +/- doublet flags the spike and the
    # sample after it.
    x = 0.1 * np.random.default_rng(0).standard_normal(60)
    x[30] += 20.0
    flag = np.asarray(ImpulseClip(21, 4.0, 1)(x))
    assert flag[30] == 1.0
    assert flag[31] == 1.0
    assert np.nansum(flag) == 2.0


def test_level_shift_body_preserved():
    # A genuine step: only the onset sample is clipped; the rest of the new level
    # is kept (jumps back to ~0 after the onset).
    x = np.concatenate([np.zeros(40), np.full(40, 5.0)])
    x += 0.001 * np.random.default_rng(0).standard_normal(x.size)
    flag = np.asarray(ImpulseClip(21, 4.0, 1)(x))
    assert flag[40] == 1.0          # onset flagged
    assert np.nansum(flag[41:]) == 0.0  # body of the step untouched


def test_batch_equals_stream():
    rng = np.random.default_rng(4)
    x = np.sin(np.linspace(0, 20, 300)) + 0.2 * rng.standard_normal(300)
    x[100] += 9.0
    for out_mode in (0, 1, 2):
        batch = np.asarray(ImpulseClip(21, 4.0, out_mode)(x))
        f = ImpulseClip(21, 4.0, out_mode)
        stream = np.array([f(v) for v in x])
        np.testing.assert_allclose(batch, stream, equal_nan=True)


def test_nan_ignore_and_recovery():
    x = np.concatenate([[np.nan, np.nan], np.sin(np.linspace(0, 10, 80))])
    out = np.asarray(ImpulseClip(21, 4.0)(x))
    assert np.isnan(out[:2]).all()
    assert np.isfinite(out[-1])


def test_start_policies_run():
    rng = np.random.default_rng(5)
    x = np.sin(np.linspace(0, 10, 60)) + 0.1 * rng.standard_normal(60)
    for policy in ("strict", "expanding", "zero"):
        out = np.asarray(ImpulseClip(10, 4.0, 0, policy)(x))
        assert out.shape == x.shape
    # zero policy has a previous value (0) from the start, so no leading NaN
    assert np.isfinite(np.asarray(ImpulseClip(10, 4.0, 0, "zero")(x))[0])


def test_two_dimensional_columns_independent():
    rng = np.random.default_rng(6)
    x = np.tile(np.sin(np.linspace(0, 10, 80)), (3, 1)).T + 0.1 * rng.standard_normal((80, 3))
    x[40, 1] += 12.0
    out = ImpulseClip(21, 4.0, 1)(x)
    assert out.shape == (80, 3)
    assert out[40, 1] == 1.0
    for c in range(3):
        np.testing.assert_allclose(out[:, c], np.asarray(ImpulseClip(21, 4.0, 1)(x[:, c])),
                                   equal_nan=True)


def test_reset_restores_initial_state():
    f = ImpulseClip(11, 4.0)
    x = np.sin(np.linspace(0, 10, 40))
    a = np.asarray(f(x))
    f.reset()
    b = np.asarray(f(x))
    np.testing.assert_allclose(a, b, equal_nan=True)


def test_invalid_arguments_raise():
    with pytest.raises(Exception):
        ImpulseClip(0)
    with pytest.raises(Exception):
        ImpulseClip(20, -1.0)
    with pytest.raises(Exception):
        ImpulseClip(20, 4.0, 9)
