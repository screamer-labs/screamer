import numpy as np
from screamer import RollingMean, Identity, Diff
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


def test_float_seconds_keys_round_trip_without_truncation():
    x = np.random.default_rng(2).standard_normal(300)
    t = np.linspace(0.0, 30.0, x.size) + 0.5   # non-integer float seconds
    keys_out, vals = streams._run_chain([Identity()], x, keys=t, return_keys=True)
    # If float keys were (wrongly) routed to the int64 binding they'd be
    # truncated to integers; exact equality with the float input catches that.
    np.testing.assert_array_equal(keys_out, t)
    np.testing.assert_array_equal(vals, Identity()(x))


def test_int_keys_round_trip():
    x = np.random.default_rng(5).standard_normal(100)
    k = (np.arange(x.size, dtype=np.int64) * 7) + 3
    keys_out, _ = streams._run_chain([Identity()], x, keys=k, return_keys=True)
    np.testing.assert_array_equal(keys_out, k)


def test_chain_equals_nested_calls():
    x = np.random.default_rng(3).standard_normal(1000)
    # source -> RollingMean(5) -> Diff(1) -> collector  ==  Diff(1)(RollingMean(5)(x))
    got = streams._run_chain([RollingMean(5), Diff(1)], x)
    exp = Diff(1)(RollingMean(5)(x))
    np.testing.assert_array_equal(got, exp)


def test_chain_float_keys_equals_nested():
    x = np.random.default_rng(4).standard_normal(400)
    t = np.linspace(0.0, 40.0, x.size)
    got = streams._run_chain([RollingMean(3), Diff(2)], x, keys=t)
    exp = Diff(2)(RollingMean(3)(x))
    np.testing.assert_array_equal(got, exp)


def test_empty_input_returns_empty():
    x = np.empty(0, dtype=np.float64)
    got = streams._run_chain([RollingMean(5)], x)
    assert got.shape == (0,)


def test_run_chain_values_only_still_correct():
    # return_keys=False path must produce identical values to the keyed path.
    x = np.random.default_rng(11).standard_normal(256)
    t = (np.arange(x.size, dtype=np.int64) * 3) + 1
    vals_only = streams._run_chain([RollingMean(4)], x, keys=t)                 # default False
    keys_out, vals_keyed = streams._run_chain([RollingMean(4)], x, keys=t, return_keys=True)
    np.testing.assert_array_equal(vals_only, vals_keyed)
    np.testing.assert_array_equal(keys_out, t)
