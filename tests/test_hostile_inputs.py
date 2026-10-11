"""Objects with unusual class metadata must raise, never crash the process.

Each case runs in a subprocess, so a segmentation fault fails the test instead
of killing pytest.
"""
import subprocess
import sys
import textwrap

import pytest

# Class headers that define a class M whose type metadata is hostile.
HEADERS = {
    "module_none": "class M:\n    __module__ = None\n",
    "module_int": "class M:\n    __module__ = 5\n",
    "module_bytes": "class M:\n    __module__ = b'x'\n",
    "module_surrogate": "class M:\n    __module__ = '\\udc80'\n",
    "module_raises": (
        "class Meta(type):\n"
        "    @property\n"
        "    def __module__(cls):\n"
        "        raise RuntimeError('boom')\n"
        "class M(metaclass=Meta):\n"
        "    pass\n"
    ),
}

CALLS = [
    "s.RollingMean(2)(M())",
    "s.Add()(M(), M())",
    "s.Add()([1.0, 2.0], M())",
    "s.RollingCorr(3)(M(), [1.0])",
    "s.RollingMinMax(3)(M())",
    "s.CrossRank()(M())",
    "s.PortfolioReport()(M())",
]


def _run(header, call):
    code = "import screamer as s\n" + header + textwrap.dedent(f"""
        try:
            {call}
        except Exception as e:
            print(type(e).__name__)
            print(e)
        else:
            print("no error")
    """)
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert p.returncode == 0, f"rc={p.returncode}\n{p.stderr}"
    return p.stdout.splitlines()


@pytest.mark.parametrize("call", CALLS)
@pytest.mark.parametrize("case", sorted(HEADERS))
def test_hostile_module_raises_type_error(case, call):
    out = _run(HEADERS[case], call)
    # A wrong kind is a TypeError that names the offending type.
    assert out[0] == "TypeError", out
    assert "'M'" in out[1] or " a M" in out[1], out


@pytest.mark.parametrize("case", sorted(HEADERS))
def test_rejected_str_subclass_with_hostile_module(case):
    # A str subclass is rejected by name before any routing.
    header = HEADERS[case].replace("class M:", "class M(str):").replace(
        "class M(metaclass=Meta):", "class M(str, metaclass=Meta):")
    for call in ["s.RollingMean(2)(M('a'))", "s.Add()(M('a'), 1.0)", "s.RollingMinMax(3)(M('a'))"]:
        out = _run(header, call)
        assert out[0] == "TypeError" and " a M " in out[1], (call, out)


def test_mixed_inputs_name_their_types():
    out = _run("", "s.Add()(1.0, [1.0])")
    assert out == ["TypeError", "Add: cannot combine inputs of types (float, list); "
                   "pass all numbers, all arrays, or all sequences"]


# Calls that fail inside nanobind itself: overload resolution (every operator
# constructor, wrong arity, an unknown keyword, a typed ndarray binding) and
# subclassing a bound type. nanobind before v2.13.0 read a non-str __module__
# unchecked here and crashed.
NANOBIND_CALLS = [
    "s.RollingMean(2)(M(), M())",                                   # overload error message
    "s.RollingMean(2)(1.0, x=M())",                                 # unknown keyword
    "s.RollingMean(M())",                                           # constructor overload error
    "s.RollingMean(window_size=M())",                               # constructor overload error
    "s.RollingMean(M(), 1, 2, 3)",                                  # constructor wrong arity
    "__import__('screamer.screamer_bindings', fromlist=['_'])"
    "._circular_shift(M(), 1, 1, 0, 1)",                            # typed ndarray binding
]


@pytest.mark.parametrize("call", NANOBIND_CALLS)
@pytest.mark.parametrize("case", sorted(set(HEADERS) - {"module_surrogate"}))
def test_nanobind_internal_hostile_module(case, call):
    out = _run(HEADERS[case], call)
    assert out[0] == "TypeError" and "incompatible function arguments" in out[1], out


@pytest.mark.parametrize("module", ["None", "5", "b'x'"])
def test_subclass_bound_type_with_hostile_module(module):
    out = _run("", f"type('X', (s.RollingMean,), {{'__module__': {module}}})(2)")
    assert out == ["no error"], out


# nanobind (v2.13.0 and its master branch as of 2026-10) still crashes on a
# __module__ that is a str but cannot be encoded as UTF-8: an overload error
# passes a null c_str() to put_dstr, and subclassing a bound type passes a null
# name to strdup_check.
@pytest.mark.xfail(strict=True, reason="nanobind crashes on an unencodable __module__")
@pytest.mark.parametrize("call", NANOBIND_CALLS + [
    "type('X', (s.RollingMean,), {'__module__': '\\udc80'})(2)",  # subclass a bound type
])
def test_nanobind_surrogate_module(call):
    out = _run(HEADERS["module_surrogate"], call)
    assert out[0] in ("TypeError", "no error"), out


# ---- G7: a wrong-kind element raises TypeError, never std::bad_cast --------

import asyncio  # noqa: E402

import screamer as s  # noqa: E402
from screamer.streams import merge  # noqa: E402


class M:
    pass


def _pipeline_iter():
    a = s.Input("a")
    p = s.Pipeline([a], [s.RollingMean(2)(a)])
    return list(p(iter([M()])))


def _pipeline_iter_value():
    a = s.Input("a")
    p = s.Pipeline([a], [s.RollingMean(2)(a)])
    return list(p(iter([(1.0, 0), (M(), 1)])))


def _pipeline_iter_index():
    a = s.Input("a")
    p = s.Pipeline([a], [s.RollingMean(2)(a)])
    return list(p(iter([(1.0, M())])))


def _pipeline_iter_wide():
    a = s.Input("a")
    p = s.Pipeline([a], [s.RollingMean(2)(a)])
    return list(p(iter([((1.0, M()), 0)])))


def _pipeline_live():
    a = s.Input("a")
    p = s.Pipeline([a], [s.RollingMean(2)(a)])
    return p.live().push("a", 0, M())


async def _agen(items):
    for x in items:
        yield x


def _async(op, items):
    async def run():
        return [x async for x in op(_agen(items))]
    return asyncio.run(run())


# (call, text the message must contain); "M" is always required as well.
WRONG_KIND = {
    "list_element": (lambda: s.RollingMean(2)([1.0, M()]), "RollingMean: event 1 is a M"),
    "list_object": (lambda: s.RollingMean(2)([object()]), "RollingMean: event 0 is a object"),
    "tuple_element": (lambda: s.RollingMean(2)((1.0, M())), "RollingMean: event 1"),
    "list_1i_Mo": (lambda: s.RollingMinMax(2)([1.0, M()]), "RollingMinMax: event 1"),
    "list_two_inputs": (lambda: s.Add()([1.0, M()], [1.0, 2.0]), "Add: event 1 of input 0"),
    "list_two_inputs_second": (lambda: s.Add()([1.0, 2.0], [1.0, M()]), "Add: event 1 of input 1"),
    "list_2i_Mo": (lambda: s.RollingLinearRegression(3)([1.0, M()], [1.0, 2.0]),
                   "RollingLinearRegression: event 1 of input 0"),
    "events_2i_Mo": (lambda: s.RollingLinearRegression(3)([(1.0, M())]),
                     "RollingLinearRegression: item [1] of event 0"),
    "events_tuple": (lambda: s.Add()([(1.0, M())]), "Add: item [1] of event 0"),
    "events_list": (lambda: s.Add()([[1.0, M()]]), "Add: item [1] of event 0"),
    "events_tuple_later": (lambda: s.Add()([(1.0, 2.0), (M(), 1.0)]), "Add: item [0] of event 1"),
    "tuple_stream": (lambda: s.Add()((M(), M())), "Add: a M"),
    "list_of_n_objects": (lambda: s.Add()([M(), M()]), "Add: a M"),
    "iter_second": (lambda: list(s.RollingMean(2)(iter([1.0, M()]))), "RollingMean: event 1"),
    "iter_first": (lambda: list(s.RollingMean(2)(iter([M()]))), "RollingMean: event 0"),
    "iter_1i_Mo": (lambda: list(s.RollingMinMax(2)(iter([1.0, M()]))), "RollingMinMax: event 1"),
    "iter_two_inputs": (lambda: list(s.Add()(iter([1.0, M()]), iter([1.0, 2.0]))),
                        "Add: event 1 of input 0"),
    "iter_events": (lambda: list(s.Add()(iter([(1.0, 2.0), (1.0, M())]))),
                    "Add: item [1] of event 1"),
    "iter_after_arrays": (lambda: list(s.RollingMean(2)(iter([[1.0, 2.0], M()]))),
                          "RollingMean: event 1"),
    "async_item": (lambda: _async(s.RollingMean(2), [1.0, M()]), "RollingMean: event 1"),
    "async_first": (lambda: _async(s.RollingMean(2), [M()]), "RollingMean: event 0"),
    "row_list": (lambda: s.CrossRank()([[1.0, 2.0], [1.0, M()]]), "CrossRank: item [1][1]"),
    "row_list_row": (lambda: s.CrossRank()([[1.0, 2.0], M()]), "CrossRank: item [1]"),
    "row_iter": (lambda: list(s.CrossRank()(iter([[1.0, 2.0], [1.0, M()]]))),
                 "CrossRank: item [1] of event 1"),
    "row_iter_row": (lambda: list(s.CrossRank()(iter([[1.0, 2.0], M()]))), "CrossRank: event 1"),
    "row_async": (lambda: _async(s.CrossRank(), [[1.0, 2.0], M()]), "CrossRank: event 1"),
    "report_list": (lambda: s.PortfolioReport()([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, M()]]),
                    "PortfolioReport: item [1][3]"),
    "report_list_row": (lambda: s.PortfolioReport()([[1.0, 2.0, 3.0, 4.0], M()]),
                        "PortfolioReport: item [1]"),
    "report_iter": (lambda: list(s.PortfolioReport()(iter([[1.0, 2.0, 3.0, 4.0],
                                                           [1.0, 2.0, 3.0, M()]]))),
                    "PortfolioReport: item [3] of event 1"),
    "report_iter_row": (lambda: list(s.PortfolioReport()(iter([[1.0, 2.0, 3.0, 4.0], M()]))),
                        "PortfolioReport: event 1"),
    "pipeline_iter": (_pipeline_iter, "Pipeline: event 0 of input 0 is a M"),
    "pipeline_iter_value": (_pipeline_iter_value, "Pipeline: the value of event 1 of input 0"),
    "pipeline_iter_index": (_pipeline_iter_index, "Pipeline: the index of event 0 of input 0"),
    "pipeline_iter_wide": (_pipeline_iter_wide, "Pipeline: item [1] of the value of event 0"),
    "pipeline_live": (_pipeline_live, ""),
    "merge_positional": (lambda: list(merge(iter([1.0, M()]), iter([2.0]))),
                         "merge: event 1 of input 0 is a M"),
    "merge_indexed": (lambda: list(merge(iter([(1.0, 0), (M(), 1)]), iter([(2.0, 0)]))),
                      "merge: the value of event 1 of input 0 is a M"),
}


@pytest.mark.parametrize("case", sorted(WRONG_KIND))
def test_wrong_kind_element_raises_type_error(case):
    call, text = WRONG_KIND[case]
    with pytest.raises(TypeError) as info:
        call()
    msg = str(info.value)
    assert "M" in msg or "object" in msg, msg
    assert text in msg, msg


def test_wrong_kind_element_hostile_module():
    # The element's type name is read safely, also for a hostile __module__.
    header = HEADERS["module_none"]
    for call in ["s.RollingMean(2)([1.0, M()])", "list(s.RollingMean(2)(iter([1.0, M()])))",
                 "s.Add()([(1.0, M())])", "s.CrossRank()([[1.0, 2.0], M()])"]:
        out = _run(header, call)
        assert out[0] == "TypeError" and "M" in out[1], (call, out)


def test_list_of_wrong_count_is_not_bad_cast():
    # A one-item list for a two-input operator was a std::bad_cast.
    with pytest.raises(TypeError):
        s.Add()([M()])


def test_wrong_kind_leaves_shape_errors_alone():
    # An event of the wrong shape (not the wrong kind) is still a ValueError.
    with pytest.raises(ValueError):
        list(s.RollingMean(2)(iter([1.0, [1.0, 2.0]])))
    with pytest.raises(ValueError):
        s.CrossRank()([[1.0, 2.0], [1.0]])


# ---- G7 review round 1 ------------------------------------------------------

_CYCLES = """
import screamer as s
l = []; l.append(l); l.append(l)
x = [1.0]
for _ in range(40):
    x = [x, x]
for call in [lambda: s.RollingMean(2)([1.0, l]), lambda: s.RollingMean(2)([1.0, x]),
             lambda: list(s.RollingMean(2)(iter([1.0, l]))), lambda: s.Add()([(1.0, l)]),
             lambda: s.Add()([1.0, l], [1.0, 2.0])]:
    try:
        call()
    except (TypeError, ValueError) as e:
        print(type(e).__name__)
    else:
        print("no error")
"""


def test_self_containing_lists_return_promptly():
    # A list that holds itself twice, or one shared list nested 40 deep, has
    # exponentially many paths; the search enters each list once.
    p = subprocess.run([sys.executable, "-c", _CYCLES], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    assert p.stdout.split() == ["ValueError"] * 5, p.stdout


async def _ag(items):
    for x in items:
        yield x


def _arun(op, items):
    async def run():
        return [x async for x in op(_ag(items))]
    return asyncio.run(run())


# (call, message): an element that holds only numbers but is not a number is a
# shape mismatch, a ValueError on every path.
WRONG_SHAPE = {
    "list": (lambda: s.RollingMean(2)([1.0, [1.0, 2.0]]), "RollingMean: event 1 has shape (2,), expected a number"),
    "iterator": (lambda: list(s.RollingMean(2)(iter([1.0, [1.0, 2.0]]))), "RollingMean: event 1 has shape (2,)"),
    "async": (lambda: _arun(s.RollingMean(2), [1.0, [1.0, 2.0]]), "RollingMean: event 1 has shape (2,)"),
    "1i_Mo_list": (lambda: s.RollingMinMax(2)([1.0, [1.0, 2.0]]), "RollingMinMax: event 1 has shape (2,)"),
    "events_list": (lambda: s.Add()([(1.0, [1.0, 2.0])]), "Add: item [1] of event 0 has shape (2,)"),
    "events_iter": (lambda: list(s.Add()(iter([(1.0, 2.0), (1.0, [1.0, 2.0])]))),
                    "Add: item [1] of event 1 has shape (2,)"),
    "events_async": (lambda: _arun(s.Add(), [(1.0, 2.0), (1.0, [1.0, 2.0])]),
                     "Add: item [1] of event 1 has shape (2,)"),
    "two_lists": (lambda: s.Add()([1.0, [1.0, 2.0]], [1.0, 2.0]), "Add: event 1 of input 0 has shape (2,)"),
    "two_iters": (lambda: list(s.Add()(iter([1.0, [1.0, 2.0]]), iter([1.0, 2.0]))),
                  "Add: event 1 of input 0 has shape (2,)"),
    "event_length": (lambda: list(s.Add()(iter([(1.0, 2.0), (1.0, 2.0, 3.0)]))),
                     "Add: event 1 has shape (3,), expected 2 numbers"),
    "pipeline_wide_item": (lambda: _pipeline_wide_shape(), "Pipeline: item [1] of the value of event 0"),
}


def _pipeline_wide_shape():
    a = s.Input("a")
    p = s.Pipeline([a], [s.RollingMean(2)(a)])
    return list(p(iter([((1.0, [2.0, 3.0]), 0)])))


@pytest.mark.parametrize("case", sorted(WRONG_SHAPE))
def test_wrong_shape_element_raises_value_error(case):
    call, text = WRONG_SHAPE[case]
    with pytest.raises(ValueError) as info:
        call()
    assert text in str(info.value), str(info.value)


class _Boom:
    def __float__(self):
        raise MemoryError("boom")


@pytest.mark.parametrize("call", [
    lambda: s.CrossRank()([[1.0, _Boom()]]),
    lambda: list(s.CrossRank()(iter([[1.0, _Boom()]]))),
    lambda: list(s.PortfolioReport()(iter([[1.0, 2.0, 3.0, _Boom()]]))),
    lambda: s.PortfolioReport()([[1.0, 2.0, 3.0, _Boom()]]),
])
def test_other_errors_are_not_relabelled(call):
    # Only a conversion's TypeError or ValueError is relabelled.
    with pytest.raises(MemoryError):
        call()


def test_lazy_pipeline_keeps_accepted_values():
    # Everything a lazy Pipeline accepted before still runs: ints, bools, numpy
    # scalars, and (value, index) tuples with extra items.
    import numpy as np
    a = s.Input("a")
    p = s.Pipeline([a], [s.RollingMean(1)(a)])
    rows = list(p(iter([(1, 0), (True, 1), (np.float32(2.5), 2), (np.int64(4), np.int64(3))])))
    assert [r[0] for r in rows] == [1.0, 1.0, 2.5, 4.0]
    assert [r[1] for r in rows] == [0, 1, 2, 3]


def test_search_budget_exhausted_keeps_the_shape_error():
    # Above the search budget (1,000,000 elements) with no wrong kind inside,
    # the shape error stands; it is never a false TypeError.
    big = [[1.0] * 3 for _ in range(400_000)]
    with pytest.raises(ValueError, match="RollingMean: event 1 has shape"):
        s.RollingMean(2)([1.0, big])
    with pytest.raises(ValueError):
        s.CrossRank()([[1.0, 2.0] for _ in range(400_000)] + [[1.0]])
