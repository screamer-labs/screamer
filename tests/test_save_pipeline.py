"""Save and load of Pipelines and live sessions.

A Pipeline saves its description only (``"state": null``) and loads as a
Pipeline. A live session saves the description and everything it holds: the
operators' and stream nodes' state, events that wait in a join or a window,
watermarks, and outputs ``result()`` has not returned; it loads as a live
session that continues bit for bit. The state is keyed by the node ids of the
description, so an edited graph refuses the state (or starts fresh).
"""
import json
import pathlib
import warnings

import numpy as np
import pytest

import screamer
from screamer import Input, Pipeline
from screamer import screamer_bindings as _b
from screamer.streams import CombineLatest, Delay, Dropna, Resample

from devtools.pipeline_io_cases import cases as _io_cases
from tests._oracle_fresh import _same as _oracle_same, run_generator

ROOT = pathlib.Path(__file__).resolve().parent.parent
SESSION_ORACLE = ROOT / "wasm" / "smoke" / "session_oracle.json"
SESSION_GEN = ROOT / "devtools" / "wasm" / "gen_session_oracle.py"


# ---- Driving a session ------------------------------------------------------

def _events(feeds, order):
    """The pushes of `feeds` as (input, index, value) in two halves.

    "merged": the events in index order, ties in input order, as a batch call
    takes them; the test advances the session at the save. "blocked": the
    first half of input 0's events, then the first quarter of input 1's, and
    after the save the rest of input 1's, then of input 0's: a join holds the
    events of one input until the other catches up, across the save.
    """
    per = []
    for k, (values, index) in enumerate(feeds):
        per.append([(k, int(index[r]), values[r]) for r in range(len(index))])
    if order == "merged":
        merged = sorted((e for p in per for e in p), key=lambda e: (e[1], e[0]))
        h = len(merged) // 2
        return merged[:h], merged[h:]
    if len(per) == 1:
        h = len(per[0]) // 2
        return per[0][:h], per[0][h:]
    h0, h1 = len(per[0]) // 2, len(per[1]) // 4
    return per[0][:h0] + per[1][:h1], per[1][h1:] + per[0][h0:]


def _push(session, events):
    for k, t, v in events:
        session.push(k, t, v)


def _same(a, b):
    if isinstance(a, tuple) and a and isinstance(a[0], tuple):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _same(x, y)
        return
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def _floor(events):
    """An advance time no later event of any input comes below: the lowest of
    the inputs' last indexes."""
    last = {}
    for k, t, _ in events:
        last[k] = t
    return min(last.values())


def _cases():
    out = dict(_io_cases())
    x, y = Input("x"), Input("y")
    k = np.arange(1, 41, dtype=np.int64) * 2
    p = np.cos(np.arange(40.0)) + 3.0
    # A Delay feeding a join: the delayed events wait in the CombineLatest.
    out["Delay into CombineLatest"] = (
        Pipeline([x, y], [CombineLatest()(Delay(5)(x), y)]), [(p, k), (p * 2.0, k + 1)])
    out["Resample freq into CombineLatest"] = (
        Pipeline([x, y], [CombineLatest()(Resample(freq=7, agg="ohlc")(x), y)]),
        [(p, k), (p - 1.0, k)])
    # An operator and a Dropna given two streams: their events wait in a merge.
    out["operator of two streams"] = (
        Pipeline([x, y], [screamer.Sub()(CombineLatest()(x, y), CombineLatest()(y, x))]),
        [(p, k), (p + 1.0, k + 1)])
    out["Dropna of two streams"] = (
        Pipeline([x, y], [Dropna()(x, y)]), [(p, k), (p + 1.0, k + 1)])
    return out


CASES = _cases()


@pytest.mark.parametrize("fmt", ["json", "cbor"])
@pytest.mark.parametrize("order", ["merged", "blocked"])
@pytest.mark.parametrize("name", sorted(CASES))
def test_live_session_continuation(name, order, fmt):
    pipe, feeds = CASES[name]
    first, rest = _events(feeds, order)
    advance = order == "merged"
    whole = pipe.live()
    try:
        _push(whole, first)
        if advance:
            whole.advance(_floor(first))
        _push(whole, rest)
        whole.flush()
    except (ValueError, RuntimeError, TypeError):
        pytest.skip("the session refuses this push order")
    expected = whole.result()

    a = pipe.live()
    _push(a, first)
    if advance:
        a.advance(_floor(first))
    data = a.dumps(format=fmt)
    b = screamer.loads(data)
    assert b.dumps(format=fmt) == data          # a loaded session saves the same file
    _push(b, rest)
    b.flush()
    _same(b.result(), expected)


def test_uncollected_outputs_come_with_the_session():
    x = Input("x")
    pipe = Pipeline([x], [screamer.RollingMean(2)(x)])
    s = pipe.live()
    for t in range(5):
        s.push("x", t, float(t))
    loaded = screamer.loads(s.dumps())
    values, index = loaded.result()
    np.testing.assert_array_equal(index, np.arange(5))
    np.testing.assert_array_equal(values, [np.nan, 0.5, 1.5, 2.5, 3.5])


def test_aligned_outputs_continue_after_a_drain():
    x, y = Input("x"), Input("y")
    pipe = Pipeline([x, y], [screamer.RollingMean(2)(x), Resample(freq=4, agg="sum")(y)])
    events = [(0, t, float(t)) for t in range(0, 30, 3)] + [(1, t, 1.0) for t in range(0, 30, 2)]
    events.sort(key=lambda e: (e[1], e[0]))
    first, rest = events[:12], events[12:]

    whole = pipe.live()
    _push(whole, first)
    before = whole.result()
    _push(whole, rest)
    whole.flush()
    after = whole.result()

    a = pipe.live()
    _push(a, first)
    _same(a.result(), before)
    b = screamer.loads(a.dumps())
    _push(b, rest)
    b.flush()
    _same(b.result(), after)


# ---- The file ----------------------------------------------------------------

def _pipe():
    x, y = Input("x"), Input("y")
    m = screamer.RollingMean(2)(x)
    return Pipeline([x, y], [CombineLatest()(m, y)])


def test_session_file_layout():
    pipe = _pipe()
    s = pipe.live()
    s.push("x", 1, 1.0).push("x", 2, 2.0).push("y", 1, 5.0)
    f = json.loads(s.dumps())
    assert f["screamer"] == "save/1"
    node = f["object"]
    assert node["type"] == "LiveSession"
    assert node["settings"] == json.loads(pipe.to_json())
    state = node["state"]
    assert list(state) == ["version", "engine", "nodes"]
    assert state["version"] == 1
    assert list(state["nodes"]) == [str(n["id"]) for n in node["settings"]["nodes"]]
    assert state["nodes"]["0"] == {"version": 1, "values": {}}       # an input holds nothing
    mean = state["nodes"]["1"]["values"]["op"]
    assert mean["version"] == 1 and mean["values"]["count"] == 2
    join = state["nodes"]["3"]["values"]["combine_latest"]["values"]
    assert join["pending_index"] == [2] and join["pending_values"] == [1.5]


def test_pipeline_save_is_settings_only(tmp_path):
    pipe = _pipe()
    text = pipe.dumps()
    f = json.loads(text)
    assert f["object"]["type"] == "Pipeline"
    assert f["object"]["state"] is None
    settings = f["object"]["settings"]
    # The settings are the description to_json writes, byte for byte.
    assert _b._graph_to_json(settings, 2) == pipe.to_json()
    loaded = screamer.loads(text)
    assert isinstance(loaded, Pipeline)
    assert loaded.to_json() == pipe.to_json()
    assert screamer.loads(pipe.dumps(format="cbor")).to_json() == pipe.to_json()
    for name in ("p.json", "p.cbor"):
        pipe.save(tmp_path / name)
        assert screamer.load(tmp_path / name).to_json() == pipe.to_json()
    feeds = ([1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0])
    _same(loaded(*feeds), pipe(*feeds))


def test_session_files(tmp_path):
    pipe = _pipe()
    s = pipe.live()
    s.push("x", 1, 1.0).push("y", 1, 5.0).push("x", 2, 2.0)
    for name in ("s.json", "s.cbor"):
        s.save(tmp_path / name)
        b = screamer.load(tmp_path / name)
        assert b.dumps() == s.dumps()


def _session_file():
    x = Input("x")
    pipe = Pipeline([x], [screamer.Clip(lower=0.0)(screamer.RollingMean(3)(x))])
    s = pipe.live()
    for t in range(4):
        s.push("x", t, float(t))
    return pipe, json.loads(s.dumps())


def test_structure_edit_refuses_or_fresh():
    pipe, f = _session_file()
    # Remove the Clip node: the graph has one node fewer than the state.
    settings = f["object"]["settings"]
    settings["nodes"] = settings["nodes"][:2]
    settings["outputs"] = [1]
    text = json.dumps(f)
    with pytest.raises(screamer.StateMismatch, match=r"LiveSession\.nodes\.2: unknown field"):
        screamer.loads(text)
    with pytest.warns(UserWarning, match="LiveSession starts fresh"):
        s = screamer.loads(text, on_mismatch="fresh")
    s.push("x", 10, 1.0)
    values, _ = s.result()
    np.testing.assert_array_equal(values, [np.nan])      # fresh: one event of a window of 3


def test_edited_setting_refuses_or_fresh():
    pipe, f = _session_file()
    f["object"]["settings"]["nodes"][1]["params"]["window_size"] = 5
    text = json.dumps(f)
    with pytest.raises(screamer.StateMismatch, match=r"LiveSession\.nodes\.1\.values\.op\.buffer"):
        screamer.loads(text)
    with pytest.warns(UserWarning):
        s = screamer.loads(text, on_mismatch="fresh")
    assert json.loads(s.dumps())["object"]["state"]["engine"]["input_count"] == [0]


def test_bad_session_states_named():
    pipe, f = _session_file()

    def edited(change):
        g = json.loads(json.dumps(f))
        change(g["object"]["state"])
        return json.dumps(g)

    with pytest.raises(screamer.StateVersionError, match=r"^LiveSession: state version 2"):
        screamer.loads(edited(lambda s: s.update(version=2)))
    with pytest.raises(screamer.StateVersionError, match=r"^LiveSession\.nodes\.1: state version 9"):
        screamer.loads(edited(lambda s: s["nodes"]["1"].update(version=9)))
    with pytest.raises(screamer.StateMismatch, match=r"LiveSession\.engine\.extra: unknown field"):
        screamer.loads(edited(lambda s: s["engine"].update(extra=1)))
    with pytest.raises(screamer.StateMismatch, match=r"LiveSession\.engine: the input widths"):
        screamer.loads(edited(lambda s: s["engine"].update(input_widths=[3])))
    with pytest.raises(screamer.StateMismatch, match=r"LiveSession\.engine\.outputs\[0\]\.values"):
        screamer.loads(edited(lambda s: s["engine"]["outputs"][0].update(values=[1.0])))
    with pytest.raises(screamer.StateMismatch, match=r"LiveSession\.engine: index_unit"):
        screamer.loads(edited(lambda s: s["engine"].update(index_unit="weeks")))
    with pytest.raises(screamer.SaveError, match=r"LiveSession\.settings"):
        g = json.loads(json.dumps(f))
        g["object"]["settings"]["nodes"][1]["cls"] = "NoSuchOp"
        screamer.loads(json.dumps(g))


def test_poisoned_session_save_raises():
    x, y = Input("x"), Input("y")
    s = Pipeline([x, y], [CombineLatest(max_pending=2)(x, y)]).live()
    with pytest.raises(RuntimeError):
        for t in range(5):
            s.push("x", t, 1.0)
    with pytest.raises(screamer.NotSaveable, match=r"^LiveSession: this session rejected an event"):
        s.dumps()


class MyMean(screamer.RollingMean):
    pass


def test_callable_node_save_raises():
    x = Input("x")
    pipe = Pipeline([x], [MyMean(3)(x)])
    with pytest.raises(screamer.NotSaveable, match=r"^Pipeline\.nodes\.1: MyMean is a subclass of RollingMean"):
        pipe.dumps()
    with pytest.raises(screamer.NotSaveable, match=r"^LiveSession\.nodes\.1: MyMean"):
        pipe.live().dumps()


def test_nested_callable_save_raises():
    lab, x = Input("label"), Input("x")
    pipe = Pipeline([lab, x], [screamer.Segment(MyMean(3))(CombineLatest()(lab, x))])
    with pytest.raises(screamer.NotSaveable, match=r"^LiveSession\.nodes\.3\.op: MyMean"):
        pipe.live().dumps()
    x = Input("x")
    pipe = Pipeline([x], [Resample(freq=3, agg=MyMean(2))(x)])
    with pytest.raises(screamer.NotSaveable, match=r"^Pipeline\.nodes\.1\.agg: MyMean"):
        pipe.dumps()


def test_lazy_call_not_saveable():
    x = Input("x")
    pipe = Pipeline([x], [screamer.RollingMean(2)(x)])
    it = pipe(iter([(1.0, 0), (2.0, 1)]))
    next(it)
    for call in (lambda: it.save("lazy.json"), lambda: it.dumps()):
        with pytest.raises(screamer.NotSaveable, match=r"Pipeline\.live\(\)"):
            call()


def test_time_index_unit_comes_back():
    x = Input("x")
    pipe = Pipeline([x], [screamer.RollingMean(2)(x)])
    s = pipe.live()
    s.push("x", np.datetime64("2026-01-01T00:00", "ns"), 1.0)
    f = json.loads(s.dumps())
    assert f["object"]["state"]["engine"]["index_unit"] == "datetime64[ns]"
    b = screamer.loads(s.dumps())
    with pytest.raises(TypeError, match="earlier indexes were a datetime64"):
        b.push("x", 5, 1.0)
    b.push("x", np.datetime64("2026-01-02T00:00", "ns"), 2.0)
    # A session of numbers keeps refusing time indexes.
    s = pipe.live()
    s.push("x", 1, 1.0)
    assert json.loads(s.dumps())["object"]["state"]["engine"]["index_unit"] is None
    with pytest.raises(TypeError, match="earlier indexes were a number"):
        screamer.loads(s.dumps()).push("x", np.datetime64("2026-01-01", "ns"), 1.0)


def test_flushed_session_stays_flushed():
    x = Input("x")
    s = Pipeline([x], [Resample(freq=3, agg="last")(x)]).live()
    s.push("x", 1, 1.0).push("x", 4, 2.0)
    s.flush()
    b = screamer.loads(s.dumps())
    _same(b.result(), s.result())
    with pytest.raises(ValueError, match="flush"):
        b.push("x", 9, 1.0)


def test_session_oracle_for_javascript_is_fresh():
    # The files are compared as trees, so their numbers compare to 1e-9 as
    # the other oracles' do (tests/_oracle_fresh.py).
    def trees(text):
        cases = json.loads(text)
        for c in cases:
            c["file"] = json.loads(c["file"])
        return cases
    diff = _oracle_same(trees(run_generator(SESSION_GEN, ROOT)), trees(SESSION_ORACLE.read_text("utf-8")))
    assert diff is None, (f"{SESSION_ORACLE.relative_to(ROOT)} is stale ({diff}): re-run python "
                          f"{SESSION_GEN.relative_to(ROOT)} and commit the result.")
