"""Tests for functor constructor-argument capture (screamer/_functor_params.py).

Every functor is wrapped in a same-named Python subclass that records its
constructor arguments on ``_screamer_params`` without changing behavior.
"""
import numpy as np
import pytest

import screamer
import screamer.screamer_bindings as _b
from screamer import (RollingMean, EwMean, Sub, RollingCorr, MovingAverage,
                      ImpulseClip, Input, Dag)
from screamer.streams import combine_latest
from screamer._functor_params import bind_params, format_call


def test_positional_arg_bound_to_name():
    assert RollingMean(20)._screamer_params == {"window_size": 20}


def test_keyword_arg_captured():
    assert EwMean(span=5)._screamer_params == {"span": 5}


def test_positional_binds_by_signature_order():
    # EwMean's first positional parameter is `com`, not `span`.
    assert EwMean(5)._screamer_params == {"com": 5}


def test_mixed_positional_and_keyword():
    p = ImpulseClip(31, n_sigma=5.0)._screamer_params
    assert p == {"window_size": 31, "n_sigma": 5.0}


def test_no_arg_functor_is_empty():
    assert Sub()._screamer_params == {}


def test_list_argument_captured():
    taps = [0.25, 0.5, 0.25]
    assert MovingAverage(taps)._screamer_params == {"taps": taps}


def test_repr_is_readable():
    assert repr(RollingMean(20)) == "RollingMean(window_size=20)"
    assert repr(Sub()) == "Sub()"
    assert repr(RollingCorr(10)) == "RollingCorr(window_size=10)"


def test_isinstance_against_cpp_base_holds():
    assert isinstance(RollingMean(20), _b.RollingMean)
    assert isinstance(Sub(), _b.Sub)
    assert isinstance(RollingMean(20), _b.ScreamerBase)


def test_type_name_unchanged():
    assert type(RollingMean(20)).__name__ == "RollingMean"


def test_single_input_functor_runs_in_dag():
    a = Input("a")
    dag = Dag([a], [RollingMean(3)(a)])
    got = np.asarray(dag(np.arange(6.0))[0])
    ref = np.asarray(RollingMean(3)(np.arange(6.0)))
    np.testing.assert_allclose(got, ref, equal_nan=True)


def test_multi_input_functor_runs_in_dag():
    a, b = Input("a"), Input("b")
    dag = Dag([a, b], [Sub()(combine_latest(a, b))])
    fa = (np.array([10.0, 20.0, 30.0]), np.array([1, 2, 3]))
    fb = (np.array([1.0, 2.0, 3.0]), np.array([1, 2, 3]))
    np.testing.assert_allclose(np.asarray(dag(fa, fb)[0]).reshape(-1), [9.0, 18.0, 27.0])


def test_every_exported_functor_is_wrapped():
    unwrapped = []
    for name in dir(screamer):
        obj = getattr(screamer, name)
        if (isinstance(obj, type) and issubclass(obj, _b.EvalOp)
                and obj not in (_b.EvalOp, _b.ScreamerBase)
                and not getattr(obj, "_screamer_wrapped", False)):
            unwrapped.append(name)
    assert unwrapped == []


def test_captured_params_reconstruct_equal_instance():
    # The whole point: params are enough to rebuild the same functor.
    original = RollingMean(20)
    rebuilt = RollingMean(**original._screamer_params)
    x = np.arange(30.0)
    np.testing.assert_allclose(np.asarray(original(x)), np.asarray(rebuilt(x)), equal_nan=True)


def test_bind_params_unknown_class_keeps_positional():
    # A class absent from the help.json schema falls back to positional capture.
    assert bind_params("NotARealFunctor", (1, 2), {"k": 3}) == {"args": [1, 2], "k": 3}


def test_format_call_renders_positional_fallback():
    assert format_call("F", {"args": [1, 2], "k": 3}) == "F(1, 2, k=3)"


def test_constructor_introspection_still_works():
    # devtools reads __init__.__doc__ for the pybind signature; wrapping preserves it.
    from devtools import get_constructor_arguments
    args = get_constructor_arguments(RollingMean)
    names = [n for n, _ in args]
    assert "window_size" in names
