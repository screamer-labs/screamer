"""The five microstructure operators that reuse another operator's kernel:
C++ classes with named inputs (Op::kInputNames), every call path."""
import asyncio
import timeit

import numpy as np
import pytest

import screamer as s
from screamer import screamer_bindings as b

rng = np.random.default_rng(3)
T, K = 30, 3
F = rng.normal(size=(T, K))
R = 0.7 * F + rng.normal(scale=0.2, size=(T, K))
F[[4, 11], [0, 2]] = np.nan                    # NaN inputs, skipped by every kernel
R[[7, 11], [1, 0]] = np.nan
f, r = F[:, 0], R[:, 0]
POLICIES = ["strict", "expanding", "zero"]

# (operator, its kernel with the inputs in kernel order, input names)
CASES = [
    (lambda: s.SignedVolume(), lambda x, y: s.Mul()(x, y), ("sign", "volume")),
    (lambda: s.QueueImbalance(), lambda x, y: s.OFI()(x, y), ("bid_size", "ask_size")),
    *[(lambda p=p: s.RollingKyleLambda(6, p), lambda x, y, p=p: s.RollingBeta(6, p)(y, x),
       ("signed_flow", "return_")) for p in POLICIES],
    (lambda: s.EwKyleLambda(span=5.0), lambda x, y: s.EwBeta(span=5.0)(y, x), ("signed_flow", "return_")),
]
IDS = ["SignedVolume", "QueueImbalance", *[f"RollingKyleLambda-{p}" for p in POLICIES], "EwKyleLambda"]


def same(a, b):
    np.testing.assert_array_equal(np.asarray(a, dtype=float), np.asarray(b, dtype=float))


async def _agen(xs):
    for x in xs:
        yield x


def _adrain(it):
    async def go():
        return [x async for x in it]
    return asyncio.run(go())


def test_the_five_are_compiled_classes():
    for name in ["SignedVolume", "QueueImbalance", "RollingOrderImbalance",
                 "RollingKyleLambda", "EwKyleLambda"]:
        assert issubclass(getattr(s, name), getattr(b, name))
    from screamer.microstructure import RollingKyleLambda
    assert RollingKyleLambda is s.RollingKyleLambda


def test_call_signature_names_the_inputs():
    assert "__call__(self, sign, volume)" in s.SignedVolume.__call__.__doc__
    assert "__call__(self, bid_size, ask_size)" in s.QueueImbalance.__call__.__doc__
    assert "__call__(self, signed_flow, return_)" in s.RollingKyleLambda.__call__.__doc__
    assert "__call__(self, signed_flow, return_)" in s.EwKyleLambda.__call__.__doc__
    assert "__call__(self, signed_flow" in s.RollingOrderImbalance.__call__.__doc__


@pytest.mark.parametrize("make, kernel, names", CASES, ids=IDS)
def test_kernel_results_in_every_form(make, kernel, names):
    ka, kb = names
    want = kernel(F, R)
    same(make()(F, R), want)                                   # separate (T, K)
    same(make()(**{ka: F, kb: R}), want)                       # keywords
    same(make()(F, **{kb: R}), want)                           # mixed
    same(make()(np.stack([F, R], axis=-1)), want)              # packed (T, K, 2)
    same(make()(f, r), kernel(f, r))                           # 1-D
    same(make()(np.column_stack([f, r])), kernel(f, r))        # packed (T, 2)
    events = list(zip(f, r))
    same(make()(events), kernel(f, r))                         # list of events
    same(list(make()(iter(events))), kernel(f, r))             # iterator of events
    same(list(make()(iter(f), iter(r))), kernel(f, r))         # two iterators
    same(list(make()(**{ka: iter(f), kb: iter(r)})), kernel(f, r))
    same(_adrain(make()(**{ka: _agen(f), kb: _agen(r)})), kernel(f, r))


@pytest.mark.parametrize("make, kernel, names", CASES, ids=IDS)
def test_scalar_events_by_position_and_keyword(make, kernel, names):
    ka, kb = names
    op_pos, op_kw = make(), make()
    pos = [op_pos(float(x), float(y)) for x, y in zip(f, r)]
    kw = [op_kw(**{ka: float(x), kb: float(y)}) for x, y in zip(f, r)]
    same(pos, kernel(f, r))
    same(kw, pos)


@pytest.mark.parametrize("make, kernel, names", CASES, ids=IDS)
def test_async_iterator_of_events(make, kernel, names):
    same(_adrain(make()(_agen(list(zip(f, r))))), kernel(f, r))


@pytest.mark.parametrize("make, kernel, names", CASES, ids=IDS)
def test_keyword_errors(make, kernel, names):
    ka, kb = names
    who = type(make()).__name__
    with pytest.raises(TypeError, match=rf"^{who}\(\) got an unexpected keyword argument 'x'$"):
        make()(f, r, x=f)
    with pytest.raises(TypeError, match=rf"^{who}\(\) got multiple values for argument '{ka}'$"):
        make()(f, **{ka: r})
    # A count is a ValueError (rule A4), a keyword the call cannot bind a TypeError.
    with pytest.raises(ValueError, match=rf"^{who} expects 2 inputs, got 3$"):
        make()(f, r, f)
    # Without keywords a missing input is the count text of every operator.
    with pytest.raises(ValueError, match=rf"^{who} expects 2 inputs, got 1$"):
        make()(f)
    with pytest.raises(ValueError, match=rf"^{who} expects 2 inputs, got 0$"):
        make()()
    with pytest.raises(ValueError, match=rf"^{who}\(\) missing 1 required input: '{kb}'$"):
        make()(**{ka: f})


@pytest.mark.parametrize("one", [
    [1.0, 2.0], [1, 2], [np.float64(1.0), np.float32(2.0)], [], 3.0, np.arange(4.0),
], ids=["floats", "ints", "numpy-scalars", "empty", "number", "1-D"])
def test_one_input_without_every_field_is_a_missing_input(one):
    with pytest.raises(ValueError, match=r"^SignedVolume expects 2 inputs, got 1$"):
        s.SignedVolume()(one)


def test_one_input_with_every_field_passes_through():
    want = [2.0, -3.0]
    same(s.SignedVolume()([(1.0, 2.0), (-1.0, 3.0)]), want)          # list of events
    same(s.SignedVolume()([[1.0, 2.0], [-1.0, 3.0]]), want)          # list of list events
    same(s.SignedVolume()(np.array([[1.0, 2.0], [-1.0, 3.0]])), want)
    assert s.SignedVolume()((1.0, 2.0)) == 2.0                        # one tuple event


@pytest.mark.parametrize("policy", POLICIES)
def test_rolling_order_imbalance_named_input(policy):
    kernel = lambda x: s.RollingSum(5, policy)(x)                    # noqa: E731
    roi = lambda: s.RollingOrderImbalance(5, policy)                 # noqa: E731
    same(roi()(F), kernel(F))
    same(roi()(signed_flow=F), kernel(F))
    same(list(roi()(signed_flow=iter(f))), kernel(f))
    same(_adrain(roi()(signed_flow=_agen(f))), kernel(f))
    op = roi()
    same([op(signed_flow=float(x)) for x in f], kernel(f))
    op = roi()
    same([op(float(x)) for x in f], kernel(f))


def test_rolling_order_imbalance_input_errors():
    # A wrong count is a ValueError (rule A4); a keyword the call cannot bind is
    # a TypeError, as Python names it.
    for call, got in [(lambda op: op(f, r), 2), (lambda op: op(), 0)]:
        with pytest.raises(ValueError, match=rf"^RollingOrderImbalance expects 1 input, got {got}$"):
            call(s.RollingOrderImbalance(5))
    for call in [lambda op: op(f, signed_flow=f), lambda op: op(extra=f)]:
        with pytest.raises(TypeError, match="signed_flow"):
            call(s.RollingOrderImbalance(5))


def test_pipeline_nodes_by_keyword_and_packed():
    a, c = s.Input("a"), s.Input("c")
    joined = s.CombineLatest()(a, c)
    swapped = s.CombineLatest()(c, a)
    out = s.Pipeline([a, c], [
        s.RollingKyleLambda(6)(joined),                              # one packed node
        s.RollingBeta(6)(swapped),
        s.RollingOrderImbalance(4)(signed_flow=a),                   # a node by keyword
        s.RollingSum(4)(a),
    ])(f, r)
    same(out[0], out[1])
    same(out[2], out[3])


def test_pipeline_nodes_by_keyword_build_the_positional_node():
    # Two separate streams into a two-input operator raise the same error by
    # keyword as by position (a Pipeline joins them with CombineLatest).
    def run(build):
        a, c = s.Input("a"), s.Input("c")
        try:
            return s.Pipeline([a, c], [build(a, c)])(f, r)
        except ValueError as e:
            return str(e)
    assert run(lambda a, c: s.SignedVolume()(sign=a, volume=c)) == \
        run(lambda a, c: s.SignedVolume()(a, c))
    assert run(lambda a, c: s.RollingKyleLambda(6)(a, return_=c)) == \
        run(lambda a, c: s.RollingKyleLambda(6)(a, c))


def _per_call(fn, n=100_000):
    return min(timeit.repeat(fn, number=n, repeat=7)) / n


@pytest.mark.perf
@pytest.mark.parametrize("make, kernel", [
    (lambda: s.SignedVolume(), lambda: s.Mul()),
    (lambda: s.QueueImbalance(), lambda: s.OFI()),
    (lambda: s.RollingKyleLambda(20), lambda: s.RollingBeta(20)),
    (lambda: s.EwKyleLambda(span=20.0), lambda: s.EwBeta(span=20.0)),
], ids=["SignedVolume", "QueueImbalance", "RollingKyleLambda", "EwKyleLambda"])
def test_positional_scalar_call_costs_what_the_kernel_costs(make, kernel):
    op, k = make(), kernel()
    t_k = _per_call(lambda: k(1.0, 2.0))
    t_op = _per_call(lambda: op(1.0, 2.0))
    assert t_op < 1.15 * t_k + 0.01e-6, (t_k, t_op)


@pytest.mark.perf
def test_rolling_order_imbalance_scalar_call_costs_what_rolling_sum_costs():
    op, k = s.RollingOrderImbalance(20), s.RollingSum(20)
    t_k = _per_call(lambda: k(1.0))
    t_op = _per_call(lambda: op(1.0))
    assert t_op < 1.15 * t_k + 0.01e-6, (t_k, t_op)
