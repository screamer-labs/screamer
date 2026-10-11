"""An iterator of events gives the same numbers as the batch it stacks into."""
import numpy as np
import pytest

import screamer as s

rng = np.random.default_rng(1)
T, K = 120, 3
X = rng.normal(size=(T, K)).cumsum(axis=0) + 50
y = rng.normal(size=T).cumsum() + 50


def test_fresh_copies_parameters_and_state():
    op = s.RollingMean(7, start_policy="expanding")
    op(np.arange(10.0))
    copy = op._fresh()
    assert type(copy) is type(op) and repr(copy) == repr(op)
    np.testing.assert_array_equal(copy(np.arange(5.0)), s.RollingMean(7, start_policy="expanding")(np.arange(5.0)))


def test_one_input_rows():
    got = np.array(list(s.RollingMean(10)(iter(X))))
    np.testing.assert_array_equal(got, s.RollingMean(10)(X))


def test_one_input_rows_multi_output():
    got = np.array(list(s.BollingerBands(10)(iter(X))))
    np.testing.assert_array_equal(got, s.BollingerBands(10)(X))


def test_unit_field_event_is_consumed():
    got = np.array(list(s.RollingMean(10)(iter(X[:, :, None]))))
    np.testing.assert_array_equal(got, s.RollingMean(10)(X))


def test_functorbase_one_input_rows():
    got = np.array(list(s.IsNan()(iter(X))))
    np.testing.assert_array_equal(got, s.IsNan()(X))


def test_separate_iterators_broadcast():
    got = np.array(list(s.RollingCorr(15)(iter(X), iter(y))))
    np.testing.assert_array_equal(got, s.RollingCorr(15)(X, y))


def test_packed_event_split():
    packed = np.stack([X, X * 2.0], axis=-1)             # events (K, 2)
    got = np.array(list(s.Sub()(iter(packed))))
    np.testing.assert_array_equal(got, -X)


def test_tuple_of_arrays_is_fields():
    got = np.array(list(s.Add()(iter(zip(X, y)))))
    np.testing.assert_array_equal(got, X + y[:, None])


def test_scalar_events_unchanged():
    got = list(s.RollingMean(3)(iter([1.0, 2.0, 3.0])))
    np.testing.assert_array_equal(got, s.RollingMean(3)(np.array([1.0, 2.0, 3.0])))
    assert isinstance(got[0], float)


def test_event_shape_change_raises():
    it = s.RollingMean(3)(iter([np.zeros(3), np.zeros(4)]))
    next(it)
    with pytest.raises(ValueError, match=r"same shape.*\(3,\).*\(4,\)"):
        next(it)


def test_scalar_then_array_raises():
    it = s.RollingMean(3)(iter([1.0, np.zeros(3)]))
    next(it)
    with pytest.raises(ValueError, match=r"RollingMean: event 1 has shape \(3,\), expected a number"):
        next(it)
