"""Resume and per-column panel state.

A default batch call starts from a reset operator and leaves it reset.
``resume(data)`` starts from the operator's current state and keeps the end
state, so a series split in two and resumed gives the uninterrupted batch
result bit for bit, also through a save and load between the parts. A
``resume`` on a (T, K) array keeps one state per column (a panel), which the
save file holds as ``"columns"``.

The tests run every saveable functor of the registry
(``tests/save_cases.py``), so operators that gain save support join them.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import screamer
from tests.regime_helpers import assert_bitwise
from tests.save_cases import FUNCTOR_NAMES, KNOWN_UNSAVEABLE, case, kwargs_for, splits

SAVEABLE = [n for n in FUNCTOR_NAMES if n not in KNOWN_UNSAVEABLE]
K = 5


def _split(name):
    """A split point after warm-up, inside the series."""
    return splits(name, kwargs_for(name))[2]


def _panel(x):
    """K related columns of a series: column j is x scaled and shifted."""
    return np.stack([x * (1.0 + 0.1 * j) + j for j in range(K)], axis=1)


def _part(inputs, start, stop):
    return [x[start:stop] for x in inputs]


def _state(op):
    return json.loads(op.dumps())["object"]["state"]


def _same(a, b):
    return json.dumps(a) == json.dumps(b)


@pytest.mark.parametrize("name", SAVEABLE)
def test_resume_1d(name):
    make, inputs = case(name)
    k = _split(name)
    want = np.asarray(make()(*inputs))
    a = make()
    assert_bitwise(a.resume(*_part(inputs, 0, k)), want[:k])
    b = screamer.loads(a.dumps())
    c = screamer.loads(a.dumps(format="cbor"))
    rest = _part(inputs, k, len(inputs[0]))
    assert_bitwise(b.resume(*rest), want[k:])
    assert_bitwise(c.resume(*rest), want[k:])
    assert_bitwise(a.resume(*rest), want[k:])


@pytest.mark.parametrize("name", SAVEABLE)
def test_resume_2d(name):
    make, inputs = case(name)
    k = _split(name)
    x2d = [_panel(x) for x in inputs]
    want = np.asarray(make()(*x2d))
    # The default 2D call runs each column from a fresh state, as before.
    for j in range(K):
        assert_bitwise(want[:, j], make()(*[x[:, j] for x in x2d]))
    a = make()
    assert_bitwise(a.resume(*_part(x2d, 0, k)), want[:k])
    b = screamer.loads(a.dumps())
    c = screamer.loads(a.dumps(format="cbor"))
    rest = _part(x2d, k, len(x2d[0]))
    assert_bitwise(b.resume(*rest), want[k:])
    assert_bitwise(c.resume(*rest), want[k:])
    assert_bitwise(a.resume(*rest), want[k:])


@pytest.mark.parametrize("name", SAVEABLE)
def test_resume_2d_strided_long(name):
    """Long columns of a wide panel take the column driver's tiled path."""
    make, inputs = case(name)
    t = len(inputs[0])
    reps = -(-1200 // t)
    long = [np.concatenate([x] * reps) for x in inputs]
    x2d = [np.stack([x * (1.0 + 0.01 * j) + j for j in range(64)], axis=1) for x in long]
    want = np.asarray(make()(*x2d))
    a = make()
    k = 700
    got = np.concatenate([np.asarray(a.resume(*_part(x2d, 0, k))),
                          np.asarray(a.resume(*_part(x2d, k, len(x2d[0]))))])
    assert_bitwise(got, want)


def test_panel_mode_switch():
    x = np.cumsum(np.random.default_rng(1).standard_normal(60))
    x2d = _panel(x)
    f = screamer.RollingMean(4)
    fresh = _state(screamer.RollingMean(4))
    f.resume(x2d[:30])
    state = _state(f)
    assert "values" not in state
    assert len(state["columns"]) == K
    with pytest.raises(ValueError, match="holds 5 columns, got 3"):
        f.resume(x2d[:, :3])
    with pytest.raises(ValueError, match="holds 5 columns, got a single series"):
        f.resume(x)
    with pytest.raises(ValueError, match="per-column state"):
        f(1.0)
    with pytest.raises(ValueError, match="per-column state"):
        f(np.float64(1.0))
    # The failed calls left the panel as it was.
    assert _same(_state(f), state)
    # A default batch call drops the panel and leaves the object reset.
    f(x)
    assert _same(_state(f), fresh)
    f.resume(x2d[:30])
    f.reset()
    assert _same(_state(f), fresh)
    assert np.isnan(f(1.0))     # value calls work again, from the reset state


def test_panel_value_call_multi_input():
    make, inputs = case("Segment")
    f = make()
    f.resume(*[_panel(x)[:20] for x in inputs])
    with pytest.raises(ValueError, match="per-column state for 5 columns"):
        f(1.0, 2.0)
    with pytest.raises(ValueError, match="per-column state"):
        f((1.0, 2.0))


def test_panel_load_keeps_columns():
    x2d = _panel(np.cumsum(np.random.default_rng(2).standard_normal(40)))
    f = screamer.EwMean(span=5)
    f.resume(x2d[:25])
    g = screamer.loads(f.dumps())
    assert len(_state(g)["columns"]) == K
    with pytest.raises(ValueError, match="per-column state"):
        g(1.0)
    # A single-series state loaded into a panel object replaces the panel.
    h = screamer.EwMean(span=5)
    h.resume(x2d[:25, 0])
    f.set_state(h.state())
    assert "values" in _state(f)
    assert_bitwise(f.resume(x2d[25:, 0]), screamer.EwMean(span=5)(x2d[:, 0])[25:])


def test_panel_state_refuses_bad_column():
    x2d = _panel(np.cumsum(np.random.default_rng(3).standard_normal(40)))
    f = screamer.RollingMean(4)
    f.resume(x2d[:25])
    state = f.state()
    state["columns"][2]["count"] = 9
    g = screamer.RollingMean(4)
    before = g.dumps()
    with pytest.raises(screamer.StateMismatch, match=r"columns\[2\]"):
        g.set_state(state)
    assert g.dumps() == before


def test_default_2d_creates_no_panel():
    x2d = _panel(np.cumsum(np.random.default_rng(4).standard_normal(40)))
    for name in SAVEABLE:
        make, inputs = case(name)
        f = make()
        f(*[_panel(x) for x in inputs])
        assert _same(_state(f), _state(make())), name
    f = screamer.RollingMean(4)
    f(x2d)
    assert "values" in _state(f)


@pytest.mark.parametrize("name", SAVEABLE)
def test_resume_on_fresh_equals_batch(name):
    make, inputs = case(name)
    assert_bitwise(make().resume(*inputs), make()(*inputs))


@pytest.mark.parametrize("name", SAVEABLE)
def test_default_batch_unchanged(name):
    make, inputs = case(name)
    f = make()
    first = f(*inputs)
    assert_bitwise(f(*inputs), first)
    # A batch call after a resume still starts from a reset object.
    g = make()
    g.resume(*inputs)
    assert_bitwise(g(*inputs), first)


def test_tuple_event_keeps_state():
    """One event given as a tuple of the inputs continues the state."""
    x = np.cumsum(np.random.default_rng(6).standard_normal(30))
    y = np.cumsum(np.random.default_rng(7).standard_normal(30))
    want = screamer.RollingCorr(5)(x, y)
    f = screamer.RollingCorr(5)
    got = [f((float(a), float(b))) for a, b in zip(x, y)]
    assert_bitwise(got, want)


def test_resume_keeps_state_for_value_calls():
    x = np.cumsum(np.random.default_rng(5).standard_normal(30))
    f = screamer.RollingMean(4)
    f.resume(x[:20])
    want = screamer.RollingMean(4)(x)
    for t in range(20, 30):
        assert_bitwise(f(float(x[t])), want[t])


def test_resume_rejects_non_series():
    f = screamer.RollingMean(4)
    with pytest.raises(TypeError, match="resume"):
        f.resume(1.0)
    with pytest.raises(TypeError, match="resume"):
        f.resume(np.float64(1.0))


def test_resume_lists_read_as_arrays():
    x = [1.0, 2.0, 4.0, 8.0, 16.0]
    f = screamer.RollingMean(2)
    assert_bitwise(f.resume(x), screamer.RollingMean(2)(np.asarray(x)))


def test_resume_input_count():
    make, inputs = case("Segment")
    with pytest.raises(ValueError):
        make().resume(inputs[0])
