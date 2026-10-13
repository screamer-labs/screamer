"""Value kinds, input counts and error text of operator calls (3.0.0, H2).

A time (NumPy datetime64/timedelta64 scalars and arrays, NaT, pandas Timestamp,
Timedelta and NaT) and a ctypes object are never values: each is a TypeError
wherever a value is read, never a crash and never a number. A wrong number of
inputs is a ValueError naming the operator, the text of the JavaScript
RangeError (js/test/input_count.test.ts).
"""
import array
import collections
import re
import ctypes
import types

import numpy as np
import pandas as pd
import pytest

import screamer as s

X = np.arange(5.0)

FOREIGN = {
    "datetime64": lambda: np.datetime64("2020-01-01", "ns"),
    "timedelta64": lambda: np.timedelta64(1, "s"),
    "NaT": lambda: np.datetime64("NaT", "ns"),
    "ctypes array": lambda: (ctypes.c_double * 3)(1.0, 2.0, 3.0),
    "ctypes double": lambda: ctypes.c_double(1.0),
    "Timestamp": lambda: pd.Timestamp("2020-01-01"),
    "Timedelta": lambda: pd.Timedelta(1),
    "pandas NaT": lambda: pd.NaT,
}


def _pipeline():
    i = s.Input("a")
    return s.Pipeline([i], [s.RollingMean(2)(i)])


CALLS = {
    "one input": lambda b: s.RollingMean(2)(b),
    "N inputs": lambda b: s.RollingCorr(2)(b, b),
    "N inputs beside a number": lambda b: s.Add()(b, 1.0),
    "list event": lambda b: s.RollingMean(2)([b]),
    "iterator event": lambda b: list(s.RollingMean(2)(iter([b]))),
    "N iterators": lambda b: list(s.Add()(iter([b]), iter([1.0]))),
    "row op": lambda b: s.CrossRank()(b),
    "Pipeline feed": lambda b: _pipeline()(b),
    "Pipeline iterator": lambda b: list(_pipeline()(iter([(b, 0)]))),
    "live push": lambda b: _pipeline().live().push("a", 0, b),
    "merge": lambda b: s.Merge()(b, X),
    "merge iterator": lambda b: list(s.Merge()(iter([(b, 0)]), iter([(1.0, 1)]))),
    "combine_latest iterator": lambda b: list(s.CombineLatest()(iter([(b, 0)]), iter([(1.0, 1)]))),
}


@pytest.mark.parametrize("make", FOREIGN.values(), ids=FOREIGN.keys())
@pytest.mark.parametrize("call", CALLS.values(), ids=CALLS.keys())
def test_a_time_or_ctypes_value_is_a_type_error(call, make):
    with pytest.raises(TypeError):
        call(make())


@pytest.mark.parametrize("name", ["datetime64", "timedelta64", "NaT", "ctypes array",
                                  "ctypes double", "Timestamp", "Timedelta", "pandas NaT"])
def test_the_message_names_the_operator_and_the_kind(name):
    kind = type(FOREIGN[name]()).__name__
    with pytest.raises(TypeError, match=rf"^RollingMean: a {kind} is not a valid input"):
        s.RollingMean(2)(FOREIGN[name]())
    with pytest.raises(TypeError, match=rf"^RollingMean: event 0 is an? {kind}, expected a number$"):
        s.RollingMean(2)([1.0, FOREIGN[name]()][1:])


@pytest.mark.parametrize("unit", ["datetime64[ns]", "timedelta64[s]"])
def test_a_time_array_as_values_is_a_type_error(unit):
    t = np.array([1, 2, 3]).astype(unit)
    msg = re.escape(f'a value must be a number, got an array of times (dtype {unit}); '
                    'pass .astype("int64") for its tick counts')
    with pytest.raises(TypeError, match="^RollingMean: " + msg):
        s.RollingMean(2)(t)
    with pytest.raises(TypeError, match="^RollingCorr: " + msg):
        s.RollingCorr(2)(t, t)
    with pytest.raises(TypeError, match="^RollingMean: " + msg):
        s.RollingMean(2)(t[0:1].reshape(()))          # a 0-d array
    with pytest.raises(TypeError, match="^RollingMean: event 0 is a"):
        s.RollingMean(2)(list(t))
    for call in (lambda: _pipeline()(t), lambda: s.Merge()(t, X), lambda: s.Dropna()(t),
                 lambda: s.Select(0)(t), lambda: s.CombineLatest()(t, t[:]),
                 lambda: s.Resample(every=2)(t)):
        with pytest.raises(TypeError, match=msg):
            call()
    # The integer ticks are one cast away.
    np.testing.assert_array_equal(s.RollingMean(1)(t.astype("int64")), [1.0, 2.0, 3.0])


def test_a_time_index_keeps_working():
    t = np.array(["2020-01-01", "2020-01-02", "2020-01-03"], "datetime64[ns]")
    v, k = _pipeline()((X[:3], t))
    np.testing.assert_array_equal(v, [np.nan, 0.5, 1.5])
    np.testing.assert_array_equal(k, t.astype("int64"))
    out, idx = s.Resample(every=2 * 86400 * 10**9)((X[:3], t))
    assert len(out) == len(idx) == 2


def test_pandas_scalars_and_from_pandas():
    for x in (pd.Timestamp("2020"), pd.Timedelta(1), pd.NaT):
        with pytest.raises(TypeError, match="^Add: a"):
            s.Add()(x, 1.0)
    with pytest.raises(TypeError, match="^from_pandas: pass a pandas Series or DataFrame, got a list$"):
        s.from_pandas([1.0, 2.0])
    with pytest.raises(TypeError, match="^from_pandas: pass a pandas Series or DataFrame, got a ndarray$"):
        s.from_pandas(X)


# ---- Input counts (rule A4) --------------------------------------------------

@pytest.mark.parametrize("call, text", [
    (lambda: s.RollingCorr(3)(X), "RollingCorr expects 2 inputs, got 1"),
    (lambda: s.RollingCorr(3)(X, X, X), "RollingCorr expects 2 inputs, got 3"),
    (lambda: s.RollingCorr(3)(), "RollingCorr expects 2 inputs, got 0"),
    (lambda: s.RollingCorr(3)((1.0, 2.0, 3.0)), "RollingCorr expects 2 inputs, got 3"),
    (lambda: s.RollingCorr(3)(1.0), "RollingCorr expects 2 inputs, got 1"),
    (lambda: s.RollingMean(3)(), "RollingMean expects 1 input, got 0"),
    (lambda: s.RollingMean(3)(1.0, 2.0), "RollingMean expects 1 input, got 2"),
    (lambda: s.Add()(iter([1.0]), iter([1.0]), iter([1.0])),
     "Add expects 2 inputs, or one stream of events holding every field, got 3"),
])
def test_a_wrong_count_is_a_value_error_naming_the_operator(call, text):
    with pytest.raises(ValueError, match="^" + text.replace("(", r"\(") + "$"):
        call()


def test_a_keyword_an_operator_does_not_take_is_a_type_error():
    with pytest.raises(TypeError, match=r"^RollingCorr\(\) got an unexpected keyword argument 'x'$"):
        s.RollingCorr(3)(x=1.0)
    with pytest.raises(TypeError):                   # nanobind names the signatures
        s.RollingMean(3)(1.0, value=2.0)
    assert s.RollingMean(1)(value=2.0) == 2.0


def test_an_empty_list_of_events_gives_an_empty_list():
    assert s.Add()([]) == []
    assert s.RollingMean(2)([]) == []


# ---- Error text names the operator (finding 12) ------------------------------

@pytest.mark.parametrize("call, pattern, kind", [
    (lambda: s.RollingMean(3)(np.array([1 + 1j])),
     r"^RollingMean: an array of complex128 values is not a valid input; pass real numbers$", TypeError),
    (lambda: s.RollingMean(3)(np.array(["a"])),
     r"^RollingMean: a value must be a number, got an array of strings", TypeError),
    (lambda: s.RollingMean(2)(np.array("x")),
     r"^RollingMean: a value must be a number, got an array of strings", TypeError),
    (lambda: s.RollingMean(2)([np.ones(2), np.ones(3)]),
     r"^RollingMean: every event must have the same shape; the first was \(2,\), event 1 is \(3,\)$",
     ValueError),
    (lambda: s.RollingMean(2)(pd.DataFrame({"a": [object(), object()]})),
     r"^RollingMean: an element is an object, expected a number$", TypeError),
    (lambda: s.RollingMean(2)(np.array([1.0, object()], dtype=object)),
     r"^RollingMean: an element is an object, expected a number$", TypeError),
    (lambda: s.Add()(1.0, pd.Series([1.0, 2.0])),
     r"^Add: cannot combine inputs of types \(float, Series\)", TypeError),
])
def test_messages_name_the_operator(call, pattern, kind):
    with pytest.raises(kind, match=pattern):
        call()


# ---- memoryview and mappings ---------------------------------------------------

def test_a_numeric_memoryview_is_an_array():
    m = memoryview(array.array("d", [1.0, 2.0, 4.0]))
    np.testing.assert_array_equal(s.RollingMean(2)(m), s.RollingMean(2)(np.asarray(m)))
    np.testing.assert_array_equal(s.RollingSum(2)(memoryview(np.arange(6.0).reshape(3, 2))),
                                  s.RollingSum(2)(np.arange(6.0).reshape(3, 2)))
    np.testing.assert_array_equal(s.Add()(m, m), [2.0, 4.0, 8.0])
    for text in (memoryview(b"12"), memoryview(bytearray(b"12"))):
        with pytest.raises(TypeError, match="^RollingMean: a memoryview of bytes is not a valid input"):
            s.RollingMean(2)(text)


@pytest.mark.parametrize("mapping", [
    types.MappingProxyType({1: 2.0}), collections.UserDict({1: 2.0}),
    collections.ChainMap({1: 2.0}), collections.OrderedDict({1: 2.0}),
], ids=["mappingproxy", "UserDict", "ChainMap", "OrderedDict"])
def test_a_mapping_is_refused_like_a_dict(mapping):
    with pytest.raises(TypeError, match=r"^RollingMean: an? \w+ is not a valid input"):
        s.RollingMean(2)(mapping)
    with pytest.raises(TypeError, match="^Add: "):
        s.Add()(mapping, mapping)


# ---- An eager call leaves the operator reset, also when it raises (D8) ----------

@pytest.mark.parametrize("make", [lambda: s.RollingMean(3), lambda: s.EwMean(span=3.0)])
def test_an_operator_is_reset_after_a_batch_call_that_raises(make):
    op = make()
    op(1.0)
    op(5.0)
    with pytest.raises(TypeError):
        op([1.0, 2.0, object()])
    np.testing.assert_array_equal(op(X), make()(X))
    fresh = make()
    np.testing.assert_array_equal([op(7.0), op(8.0), op(9.0)],
                                  [fresh(7.0), fresh(8.0), fresh(9.0)])
    add = s.Add()
    with pytest.raises(TypeError):
        add([1.0, 2.0], [1.0, "x"])
    assert add(1.0, 2.0) == 3.0
