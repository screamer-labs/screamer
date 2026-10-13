"""Python side of the FC review fixes: one NaN-index rule on every path
(ValueError, the JavaScript RangeError texts), Resample settings near the
int64 limit and origin through the shared rule, a threshold driver without an
index, and non-finite or huge volumes that never loop (G34). The JavaScript
counterparts are in js/test/fix_round1.test.ts."""

import time

import numpy as np
import pytest

import screamer as s
from screamer import CombineLatest, Input, Merge, Pipeline, Resample, RollingMean

NAN = float("nan")
ENGINE = "index must be finite, got NaN; the streaming engine is int64-indexed"


def _p():
    x = Input("x")
    return Pipeline([x], [RollingMean(2)(x)])


@pytest.mark.parametrize("call, msg", [
    (lambda: _p()((np.array([1., 2.]), np.array([0., NAN]))), ENGINE),
    (lambda: _p().live().push("x", NAN, 1.0), ENGINE),
    (lambda: list(_p()(iter([(1., 0), (2., NAN)]))),
     "Pipeline: the index of input 0 is NaN at event 1; an index must be a number"),
    (lambda: list(CombineLatest()(iter([(1., 0), (2., NAN)]))),
     "combine_latest: the index of input 0 is NaN at event 1; an index must be a number"),
    (lambda: list(CombineLatest()(iter([(1., 0.5), (2., NAN)]))),
     "combine_latest: the index of input 0 is NaN at event 1; an index must be a number"),
    (lambda: list(Merge()(iter([(1., 0), (2., np.float64(NAN))]))),
     "merge: the index of input 0 is NaN at event 1; an index must be a number"),
])
def test_a_nan_index_is_a_value_error_on_every_path(call, msg):
    with pytest.raises(ValueError) as e:
        call()
    assert str(e.value) == msg


def test_a_lazy_run_names_itself_in_index_errors():
    with pytest.raises(ValueError) as e:
        list(_p()(iter([(1., 2), (2., 1)])))
    assert str(e.value).startswith("Pipeline: the index of input 0 decreases at event 1")


def test_int64_settings_near_the_limit_are_accepted():
    v, k = Resample(every=2**63 - 1)(np.arange(3.))
    assert list(v) == [2.0]
    Resample(np.arange(3.), np.arange(3.), clock=True, fill="carry", max_age=2**63 - 1)
    with pytest.raises(ValueError) as e:
        Resample(every=10**400)(np.arange(3.))
    assert str(e.value) == "resample: every must be within the int64 range, got an int of 401 digits"


@pytest.mark.parametrize("origin, err, msg", [
    (NAN, ValueError, "resample: origin must be an integer, got nan"),
    (1e30, ValueError, "resample: origin must be within the int64 range, got 1e+30"),
    ("1", TypeError, "resample: origin must be an integer, got str"),
])
def test_origin_goes_through_the_shared_rule(origin, err, msg):
    with pytest.raises(err) as e:
        Resample(every=2, origin=origin)(np.arange(3.))
    assert str(e.value) == msg


def test_a_threshold_driver_has_no_index():
    with pytest.raises(ValueError) as e:
        Resample((np.arange(3.), np.arange(3)), (np.ones(3), np.arange(3)), threshold=1)
    assert str(e.value) == ("resample: a threshold driver is paired with the values by position; "
                            "pass the driver without an index")


@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
def test_vpin_treats_a_non_finite_volume_as_missing(bad):
    op = s.VPIN(1.0, 3)
    for _ in range(3):
        op(1.0, 1.0)
    assert np.isnan(op(1.0, bad))
    assert op(1.0, 1.0) == 0.0          # the state is untouched


def test_vpin_huge_volumes_end_quickly():
    t0 = time.perf_counter()
    out = s.VPIN(1.0, 50)(np.array([1e308, 1., 1e308]), np.array([0., 1., 1e307]))
    np.testing.assert_allclose(out, [1.0, 0.96, 0.8181818181818182], rtol=1e-12)
    assert s.VPIN(1e-300, 50)(1e308, 0.0) == pytest.approx(1.0)
    assert time.perf_counter() - t0 < 1.0


def test_threshold_resample_treats_an_infinite_driver_as_missing():
    v, k = Resample(np.arange(5.), np.array([1., np.inf, 1., -np.inf, 1.]), threshold=2)
    assert list(v) == [2.0, 4.0] and list(k) == [0, 3]
