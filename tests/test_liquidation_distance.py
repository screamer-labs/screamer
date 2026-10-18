"""LiquidationDistance: hand-computed segments, breach latching, NaN handling,
setting checks, batch == stream, and (T, K) columns."""
import numpy as np
import pytest

from screamer import LiquidationDistance

NAN = np.nan


def _run(pos, o, h, l, **kw):
    kw.setdefault("leverage", 10.0)
    arrays = [np.asarray(x, dtype=float) for x in (pos, o, h, l)]
    return np.asarray(LiquidationDistance(**kw)(*arrays))


def test_long_segment_by_hand():
    # leverage 10, maintenance 0.005: liq = 100 * (1 - 0.1 + 0.005) = 90.5
    out = _run([1, 1, 1], [100, 99, 97], [101, 100, 98], [98, 95, 91])
    np.testing.assert_allclose(out[:, 1], 90.5)
    np.testing.assert_allclose(out[:, 0], [(98 - 90.5) / 98, (95 - 90.5) / 95, (91 - 90.5) / 91])
    np.testing.assert_array_equal(out[:, 2], [0, 0, 0])


def test_short_segment_by_hand():
    # leverage 5, maintenance 0.01: liq = 200 * (1 + 0.2 - 0.01) = 238
    out = _run([-2, -2], [200, 210], [205, 230], [195, 205], leverage=5.0, maintenance=0.01)
    np.testing.assert_allclose(out[:, 1], 238.0)
    np.testing.assert_allclose(out[:, 0], [(238 - 205) / 205, (238 - 230) / 230])
    np.testing.assert_array_equal(out[:, 2], [0, 0])


def test_flat_bars_are_nan():
    out = _run([0, 1, 0, 0], [100, 100, 100, 100], [101] * 4, [99] * 4)
    assert np.isnan(out[[0, 2, 3]]).all()
    assert np.isfinite(out[1]).all()


def test_breach_latches_until_flat():
    pos = [1, 1, 1, 1, 0, 1]
    o = [100, 95, 92, 99, 99, 100]
    h = [101, 96, 93, 100, 100, 101]
    l = [99, 90, 92, 98, 98, 99]             # bar 1 goes through the liquidation price
    out = _run(pos, o, h, l)
    assert out[1, 0] < 0.0
    np.testing.assert_array_equal(out[:4, 2], [0, 1, 1, 1])
    assert out[2, 0] > 0 and out[3, 0] > 0   # recovered, still breached
    assert np.isnan(out[4]).all()
    assert out[5, 2] == 0.0                   # a fresh entry starts unbreached


def test_sign_flip_rebases_at_the_open():
    out = _run([1, 1, -1, -1], [100, 98, 96, 97], [101, 99, 97, 98], [99, 97, 95, 96])
    np.testing.assert_allclose(out[:2, 1], 90.5)
    np.testing.assert_allclose(out[2:, 1], 96 * 1.095)
    np.testing.assert_allclose(out[3, 0], (96 * 1.095 - 98) / 98)


def test_increase_blends_the_entry():
    # 1 at 100, then 2 more at 110: entry = (1 * 100 + 2 * 110) / 3
    out = _run([1, 1, 3, 3], [100, 105, 110, 108], [101, 106, 111, 109], [99, 104, 109, 107])
    np.testing.assert_allclose(out[:2, 1], 90.5)
    entry = (100 + 2 * 110) / 3
    np.testing.assert_allclose(out[2:, 1], entry * 0.905)
    np.testing.assert_allclose(out[3, 0], (107 - entry * 0.905) / 107)
    # a short increase blends the same way
    out = _run([-2, -3], [100, 94], [101, 95], [99, 93])
    np.testing.assert_allclose(out[1, 1], (2 * 100 + 94) / 3 * 1.095)


def test_reduce_keeps_the_entry():
    out = _run([3, 3, 1, 1], [100, 105, 110, 108], [101, 106, 111, 109], [99, 104, 109, 107])
    np.testing.assert_allclose(out[:, 1], 90.5)
    # reduce, then increase: the blend starts from the kept entry
    out = _run([2, 1, 2], [100, 120, 110], [101, 121, 111], [99, 119, 109])
    np.testing.assert_allclose(out[2, 1], (100 + 110) / 2 * 0.905)


def test_resize_keeps_the_breach_and_a_flip_clears_it():
    pos = [1, 1, 2, 1, -1]
    o = [100, 92, 93, 94, 95]
    h = [101, 93, 95, 95, 96]
    l = [99, 90, 92, 93, 94]
    out = _run(pos, o, h, l)
    np.testing.assert_array_equal(out[:, 2], [0, 1, 1, 1, 0])


def test_nan_position_holds():
    a = _run([1, NAN, NAN, 1], [100, 99, 98, 97], [101] * 4, [97, 96, 95, 94])
    b = _run([1, 1, 1, 1], [100, 99, 98, 97], [101] * 4, [97, 96, 95, 94])
    np.testing.assert_array_equal(a, b)
    assert np.isnan(_run([NAN, NAN], [100, 100], [101, 101], [99, 99])).all()


def test_nan_price_gives_nan_row_and_keeps_the_segment():
    pos = [1, 1, 1, 1]
    o = [100, 99, NAN, 97]
    h = [101, NAN, 100, 98]
    l = [99, 90, 98, 96]
    out = _run(pos, o, h, l)
    assert np.isnan(out[1]).all() and np.isnan(out[2]).all()
    np.testing.assert_allclose(out[3], [(96 - 90.5) / 96, 90.5, 0.0])   # bar 1's low is not seen


def test_nan_entry_open_fills_at_the_next_finite_open():
    out = _run([0, 1, 1, 1], [100, NAN, 98, 97], [101, 101, 99, 98], [99, 99, 97, 96])
    assert np.isnan(out[:2]).all()
    np.testing.assert_allclose(out[2:, 1], 98 * 0.905)


def test_entry_fills_at_a_finite_open_when_high_or_low_is_nan():
    out = _run([0, 1, 1], [100, 100, 98], [101, NAN, 99], [99, 99, 97])
    assert np.isnan(out[1]).all()
    np.testing.assert_allclose(out[2, 1], 90.5)


def test_increase_on_a_nan_open_fills_at_the_next_finite_open():
    out = _run([1, 2, 2], [100, NAN, 110], [101, 101, 111], [99, 99, 109])
    assert np.isnan(out[1]).all()
    np.testing.assert_allclose(out[2, 1], (100 + 110) / 2 * 0.905)


def test_going_flat_while_a_fill_is_pending():
    out = _run([0, 1, 0, 0], [100, NAN, 100, 100], [101] * 4, [99] * 4)
    assert np.isnan(out).all()
    out = _run([0, 1, 0, 1], [100, NAN, 100, 97], [101, 101, 101, 98], [99, 99, 99, 96])
    assert np.isnan(out[:3]).all()
    np.testing.assert_allclose(out[3], [(96 - 97 * 0.905) / 96, 97 * 0.905, 0.0])


def test_flip_on_a_nan_open_fills_at_the_next_finite_open():
    pos = [1, 1, -1, -1]
    o = [100, 92, NAN, 97]
    h = [101, 93, 98, 98]
    l = [99, 90, 96, 96]                     # bar 1 breaches the long
    out = _run(pos, o, h, l)
    assert out[1, 2] == 1.0
    assert np.isnan(out[2]).all()
    np.testing.assert_allclose(out[3], [(97 * 1.095 - 98) / 98, 97 * 1.095, 0.0])


@pytest.mark.parametrize("kw", [
    {"leverage": 1.0}, {"leverage": 0.5}, {"leverage": np.inf}, {"leverage": np.nan},
    {"leverage": 10.0, "maintenance": -0.001}, {"leverage": 10.0, "maintenance": 0.1},
    {"leverage": 10.0, "maintenance": 0.2}, {"leverage": 10.0, "maintenance": np.nan},
])
def test_invalid_settings_raise(kw):
    with pytest.raises(ValueError):
        LiquidationDistance(**kw)


def test_leverage_is_required():
    with pytest.raises(TypeError):
        LiquidationDistance()


def _market(T=500, K=None, seed=3):
    rng = np.random.default_rng(seed)
    shape = (T,) if K is None else (T, K)
    pos = np.repeat(rng.choice([0.0, 1.0, 2.0, -1.0], size=(T // 20 + 1,) + shape[1:]), 20, axis=0)[:T]
    pos[rng.random(shape) < 0.05] = NAN
    c = 100 * np.exp(np.cumsum(rng.standard_normal(shape) * 0.03, axis=0))
    o = np.concatenate([c[:1], c[:-1]])
    h = np.maximum(o, c) * 1.01
    l = np.minimum(o, c) * 0.99
    o[rng.random(shape) < 0.02] = NAN
    l[rng.random(shape) < 0.02] = NAN
    return pos, o, h, l


def test_stream_equals_batch_and_reset():
    arrays = _market()
    op = LiquidationDistance(leverage=15.0)
    batch = np.asarray(op(*arrays))
    assert (batch[:, 2] == 1).any() and (batch[:, 2] == 0).any()
    op.reset()
    stream = np.array([op(*(a[i] for a in arrays)) for i in range(len(arrays[0]))])
    np.testing.assert_array_equal(stream, batch)
    np.testing.assert_array_equal(np.asarray(op(*arrays)), batch)   # array call starts fresh


def test_columns_match_one_column_at_a_time():
    arrays = _market(K=3, seed=4)
    out = np.asarray(LiquidationDistance(leverage=12.0)(*arrays))
    assert out.shape == (500, 3, 3)
    for k in range(3):
        one = np.asarray(LiquidationDistance(leverage=12.0)(*(a[:, k] for a in arrays)))
        np.testing.assert_array_equal(out[:, k, :], one)
