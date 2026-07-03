import numpy as np
from screamer import RollingMean, Sub, Cart2Polar
from screamer import screamer_bindings as _b


def test_engine_1in_1out_matches_eager():
    x = np.random.default_rng(0).standard_normal(200)
    keys = np.arange(x.size, dtype=np.int64)
    out = _b._run_functor_batch(RollingMean(5), keys, x)     # width 1 -> 1
    assert out.shape == (200, 1)
    np.testing.assert_array_equal(out.reshape(-1), RollingMean(5)(x))


def test_engine_2in_1out_aligned_matches_eager():
    a = np.random.default_rng(1).standard_normal(200)
    b = np.random.default_rng(2).standard_normal(200)
    aligned = np.ascontiguousarray(np.column_stack([a, b]))   # (200, 2)
    keys = np.arange(200, dtype=np.int64)
    out = _b._run_functor_batch(Sub(), keys, aligned)         # width 2 -> 1
    np.testing.assert_array_equal(out.reshape(-1), a - b)


def test_engine_2in_2out_matches_eager():
    xy = np.ascontiguousarray(np.random.default_rng(3).standard_normal((50, 2)))
    keys = np.arange(50, dtype=np.int64)
    out = _b._run_functor_batch(Cart2Polar(), keys, xy)       # width 2 -> 2
    assert out.shape == (50, 2)
    exp = Cart2Polar()(xy[:, 0], xy[:, 1])                    # (50, 2)
    np.testing.assert_array_equal(out, exp)


def test_engine_width_mismatch_raises():
    x = np.random.default_rng(4).standard_normal(10)          # width 1
    keys = np.arange(10, dtype=np.int64)
    try:
        _b._run_functor_batch(Sub(), keys, x)                 # Sub needs width 2
        assert False, "expected a width-mismatch error"
    except Exception:
        pass
