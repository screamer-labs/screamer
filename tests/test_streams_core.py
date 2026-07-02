import numpy as np
from screamer import RollingMean
from screamer import streams


def test_single_functor_matches_batch_int_keys():
    x = np.random.default_rng(0).standard_normal(1000)
    keys = np.arange(x.size, dtype=np.int64)
    got = streams._run_chain([RollingMean(5)], x, keys=keys)
    exp = RollingMean(5)(x)
    np.testing.assert_array_equal(got, exp)


def test_row_number_keys_default():
    x = np.random.default_rng(1).standard_normal(500)
    got = streams._run_chain([RollingMean(7)], x)   # keys default to row number
    exp = RollingMean(7)(x)
    np.testing.assert_array_equal(got, exp)
