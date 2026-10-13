"""The five microstructure operators that reuse another operator's kernel:
C++ classes with named inputs (Op::kInputNames), every call path."""
import asyncio

import numpy as np
import pytest

import screamer as s
from screamer import screamer_bindings as b

rng = np.random.default_rng(3)
T, K = 30, 3
F = rng.normal(size=(T, K))
R = 0.7 * F + rng.normal(scale=0.2, size=(T, K))
f, r = F[:, 0], R[:, 0]

# (operator, its kernel with the inputs in kernel order, input names)
CASES = [
    (lambda: s.SignedVolume(), lambda x, y: s.Mul()(x, y), ("sign", "volume")),
    (lambda: s.QueueImbalance(), lambda x, y: s.OFI()(x, y), ("bid_size", "ask_size")),
    (lambda: s.RollingKyleLambda(6), lambda x, y: s.RollingBeta(6)(y, x), ("signed_flow", "return_")),
    (lambda: s.EwKyleLambda(span=5.0), lambda x, y: s.EwBeta(span=5.0)(y, x), ("signed_flow", "return_")),
]
IDS = ["SignedVolume", "QueueImbalance", "RollingKyleLambda", "EwKyleLambda"]


def same(a, b):
    np.testing.assert_array_equal(np.asarray(a, dtype=float), np.asarray(b, dtype=float))


def test_the_five_are_compiled_classes():
    for name in ["SignedVolume", "QueueImbalance", "RollingOrderImbalance",
                 "RollingKyleLambda", "EwKyleLambda"]:
        assert issubclass(getattr(s, name), getattr(b, name))
    from screamer.microstructure import RollingKyleLambda
    assert RollingKyleLambda is s.RollingKyleLambda


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
    async def agen():
        for e in zip(f, r):
            yield e

    async def drain():
        return [x async for x in make()(agen())]

    same(asyncio.run(drain()), kernel(f, r))


@pytest.mark.parametrize("make, kernel, names", CASES, ids=IDS)
def test_keyword_errors(make, kernel, names):
    ka, kb = names
    who = type(make()).__name__
    with pytest.raises(TypeError, match=rf"^{who}\(\) got an unexpected keyword argument 'x'$"):
        make()(f, r, x=f)
    with pytest.raises(TypeError, match=rf"^{who}\(\) got multiple values for argument '{ka}'$"):
        make()(f, **{ka: r})
    with pytest.raises(TypeError, match=rf"^{who}\(\) takes 2 inputs, got 3 positional$"):
        make()(f, r, f)
    with pytest.raises(TypeError, match=rf"^{who}\(\) missing 1 required input: '{kb}'$"):
        make()(f)
    with pytest.raises(TypeError, match=rf"^{who}\(\) missing 2 required inputs: '{ka}', '{kb}'$"):
        make()()


def test_rolling_order_imbalance_named_input():
    want = s.RollingSum(5)(F)
    same(s.RollingOrderImbalance(5)(F), want)
    same(s.RollingOrderImbalance(5)(signed_flow=F), want)
    op = s.RollingOrderImbalance(5)
    same([op(signed_flow=float(x)) for x in f], s.RollingSum(5)(f))
    with pytest.raises(TypeError, match=r"^RollingOrderImbalance\(\) takes 1 input, got 2 positional$"):
        s.RollingOrderImbalance(5)(f, r)
    with pytest.raises(TypeError, match=r"missing 1 required input: 'signed_flow'$"):
        s.RollingOrderImbalance(5)()


def test_pipeline_nodes_by_keyword_and_packed():
    a, c = s.Input("a"), s.Input("c")
    joined = s.CombineLatest()(a, c)
    swapped = s.CombineLatest()(c, a)
    out = s.Pipeline([a, c], [s.RollingKyleLambda(6)(joined), s.RollingBeta(6)(swapped)])(f, r)
    same(out[0], out[1])
    node = s.SignedVolume()(sign=joined, volume=joined)
    assert node is not None
