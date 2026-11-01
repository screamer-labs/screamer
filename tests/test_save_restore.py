"""Save and load of operators: settings, state and the save file.

An operator saved after any number of events and loaded back continues bit for
bit: feeding A, saving, loading and feeding B gives what feeding A then B
gives. The registry test runs every functor of help.json at three save points
(at construction, mid warm-up and after two windows), through JSON and CBOR.
Functors without save support yet are listed in
``tests/save_cases.py::KNOWN_UNSAVEABLE`` and expected to fail.

The golden files in ``tests/data/state_golden`` fix each operator's state
layout per state version, so a layout change without a version bump fails.
"""
from __future__ import annotations

import json
import pathlib
import re

import numpy as np
import pytest

import screamer
from screamer import screamer_bindings as _b
from tests._oracle_fresh import _same as _oracle_same, run_generator
from tests.regime_helpers import assert_bitwise
from tests import test_batch_stream_bitwise as _sweep
from tests.save_cases import (FUNCTOR_NAMES, HELP, KNOWN_UNSAVEABLE, REST, case, feed,
                              kwargs_for, splits)

ROOT = pathlib.Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "tests" / "data" / "state_golden"

SAVEABLE = [n for n in FUNCTOR_NAMES if n not in KNOWN_UNSAVEABLE]


def _unsaveable_mark(name):
    if name in KNOWN_UNSAVEABLE:
        return pytest.mark.xfail(strict=True, reason=f"{name} has no save support yet")
    return ()


def _same(a, b):
    """Two JSON-native trees are equal, NaN equal to NaN and -0.0 apart from 0.0."""
    return json.dumps(a) == json.dumps(b)


# ---- Continuation ------------------------------------------------------------

@pytest.mark.parametrize("split", ["0", "3", "2w"])
@pytest.mark.parametrize(
    "name", [pytest.param(n, marks=_unsaveable_mark(n)) for n in FUNCTOR_NAMES])
def test_continuation(name, split):
    make, inputs = case(name)
    k = splits(name, kwargs_for(name))[("0", "3", "2w").index(split)]
    a = make()
    feed(a, inputs, 0, k)
    text = a.dumps()
    blob = a.dumps(format="cbor")
    assert isinstance(text, str) and isinstance(blob, bytes)
    b = screamer.loads(text)
    c = screamer.loads(blob)
    assert type(b).__name__ == name and type(c).__name__ == name
    assert b.dumps() == text and c.dumps() == text
    end = k + REST
    want = feed(a, inputs, k, end)
    assert_bitwise(feed(b, inputs, k, end), want)
    assert_bitwise(feed(c, inputs, k, end), want)


@pytest.mark.parametrize("name", SAVEABLE)
def test_settings_match_help(name):
    make, _ = case(name)
    op = make()
    settings = op.settings()
    assert list(settings) == [p["name"] for p in HELP[name].get("parameters") or []]
    again = screamer.from_settings({"type": name, "settings": settings})
    assert type(again).__name__ == name
    assert _same(again.settings(), settings)


@pytest.mark.parametrize("name", SAVEABLE)
def test_state_is_reset_state(name):
    make, inputs = case(name)
    op = make()
    fresh = op.state()
    feed(op, inputs, 0, 10)
    op.reset()
    assert _same(op.state(), fresh)


@pytest.mark.parametrize("name", SAVEABLE)
def test_from_settings_is_fresh(name):
    make, inputs = case(name)
    op = make()
    feed(op, inputs, 0, 10)
    built = screamer.from_settings(json.loads(op.dumps())["object"])
    assert _same(built.state(), make().state())


def test_save_and_load_files(tmp_path):
    x = np.linspace(0.0, 1.0, 40)
    a = screamer.RollingMean(5)
    for v in x[:20]:
        a(v)
    for path in (tmp_path / "f.json", tmp_path / "f.cbor"):
        a.save(path)
        b = screamer.load(path)
        assert b.dumps() == a.dumps()
    assert (tmp_path / "f.json").read_text(encoding="utf-8") == a.dumps()
    assert (tmp_path / "f.cbor").read_bytes() == a.dumps(format="cbor")


def test_file_layout():
    a = screamer.RollingMean(3)
    for v in (1.0, 2.0, float("nan"), 4.0):
        a(v)
    doc = json.loads(a.dumps())
    assert list(doc) == ["screamer", "library_version", "object"]
    assert doc["screamer"] == "save/1" and doc["library_version"] == screamer.__version__
    node = doc["object"]
    assert list(node) == ["type", "settings", "state"]
    assert node["type"] == "RollingMean"
    assert node["settings"] == {"window_size": 3, "start_policy": "strict"}
    assert node["state"]["version"] == 1
    assert node["state"]["values"]["sum"] == 7.0


def test_ewmean_keeps_given_smoothing():
    assert screamer.EwMean(span=10).settings() == {
        "com": None, "span": 10.0, "halflife": None, "alpha": None}
    assert screamer.EwMean(halflife=3).settings()["halflife"] == 3.0
    b = screamer.from_settings({"type": "EwMean", "settings": screamer.EwMean(alpha=0.25).settings()})
    assert b.settings() == {"com": None, "span": None, "halflife": None, "alpha": 0.25}
    assert repr(b) == "EwMean(alpha=0.25)"


# ---- Edited and bad files ----------------------------------------------------

def _rolling_mean_20():
    a = screamer.RollingMean(20)
    for v in np.arange(30.0):
        a(v)
    return a


def test_edited_setting_refuses_or_fresh():
    text = _rolling_mean_20().dumps()
    edited = text.replace('"window_size": 20', '"window_size": 50')
    assert edited != text
    with pytest.raises(screamer.StateMismatch, match="window_size|buffer"):
        screamer.loads(edited)
    with pytest.warns(UserWarning, match="RollingMean"):
        b = screamer.loads(edited, on_mismatch="fresh")
    x = np.arange(100.0)
    fresh = screamer.RollingMean(50)
    assert_bitwise([b(v) for v in x], [fresh(v) for v in x])


def _edit(text, fn):
    doc = json.loads(text)
    fn(doc)
    return json.dumps(doc)


def _unknown_type(d):
    d["object"]["type"] = "NoSuchOperator"


def _unknown_field(d):
    d["object"]["state"]["values"]["extra"] = 1


def _missing_field(d):
    del d["object"]["state"]["values"]["sum"]


def _string_for_number(d):
    d["object"]["state"]["values"]["sum"] = "a lot"


def _wrong_length(d):
    d["object"]["state"]["values"]["buffer"] = [1.0, 2.0]


# (edit of a saved RollingMean(20), or a whole bad file; the path the message names)
STATE_EDITS = [
    (_unknown_field, "RollingMean.extra"),
    (_missing_field, "RollingMean.sum"),
    (_string_for_number, "RollingMean.sum"),
    (_wrong_length, "RollingMean.buffer"),
]
BAD_FILES = [(_unknown_type, "object.type")] + STATE_EDITS + [
    ("{not json", "save JSON"),
    (b"\xa1\x61", "CBOR"),
    ('{"screamer": "save/2", "object": {}}', "save/2"),
]


def _ids(cases):
    return [c.__name__.lstrip("_") if callable(c) else repr(c)[:12] for c, _ in cases]


@pytest.mark.parametrize("bad, where", BAD_FILES, ids=_ids(BAD_FILES))
def test_bad_files_named_errors(bad, where):
    text = _edit(_rolling_mean_20().dumps(), bad) if callable(bad) else bad
    with pytest.raises(screamer.SaveError, match=re.escape(where)):
        screamer.loads(text)


@pytest.mark.parametrize("bad, where", STATE_EDITS, ids=_ids(STATE_EDITS))
def test_failed_set_state_leaves_object_unchanged(bad, where):
    a = _rolling_mean_20()
    before = a.state()
    with pytest.raises(screamer.StateMismatch, match=re.escape(where)):
        a.set_state(json.loads(_edit(a.dumps(), bad))["object"]["state"])
    assert _same(a.state(), before)
    b = _rolling_mean_20()
    x = np.arange(30.0, 60.0)
    assert_bitwise([a(v) for v in x], [b(v) for v in x])


def test_bad_state_invariants_refuse():
    a = _rolling_mean_20()
    state = a.state()
    state["values"]["count"] = 21
    with pytest.raises(screamer.StateMismatch, match="RollingMean"):
        a.set_state(state)
    q = screamer.RollingQuantile(4, 0.5)
    for v in (1.0, 2.0, 3.0):
        q(v)
    state = q.state()
    state["values"]["window"][1] = 9.0
    with pytest.raises(screamer.StateMismatch, match="RollingQuantile"):
        q.set_state(state)
    m = screamer.RollingMax(4)
    for v in (3.0, 1.0, 2.0):
        m(v)
    state = m.state()
    state["values"]["values"] = state["values"]["values"][::-1]
    with pytest.raises(screamer.StateMismatch, match="RollingMax"):
        m.set_state(state)


def test_rolling_mean_index_must_equal_count_in_warm_up():
    a = screamer.RollingMean(5)
    for v in (1.0, 2.0):
        a(v)
    state = a.state()
    state["values"]["index"] = 4
    with pytest.raises(screamer.SaveError, match="RollingMean: index must equal count"):
        a.set_state(state)


def test_rolling_max_index_not_negative():
    m = screamer.RollingMax(4)
    state = m.state()
    state["values"]["index"] = -1
    with pytest.raises(screamer.SaveError, match="RollingMax: index must not be negative"):
        m.set_state(state)


def test_rolling_quantile_fresh_value_by_value_equals_batch():
    """A fresh RollingQuantile starts with an empty (NaN) window, so a -1.0
    in the data stays in the window (it started filled with -1 before)."""
    x = [-1.0, 5.0, 7.0]
    q = screamer.RollingQuantile(3, 0.5)
    got = [q(v) for v in x]
    assert_bitwise(got, screamer.RollingQuantile(3, 0.5)(np.array(x)))
    assert_bitwise(got, [np.nan, np.nan, 5.0])


def test_unknown_setting_named():
    with pytest.raises(screamer.SaveError, match="RollingMean.settings.window"):
        screamer.from_settings({"type": "RollingMean", "settings": {"window": 3}})


def test_invalid_setting_named():
    with pytest.raises(screamer.SaveError, match="RollingMean.settings"):
        screamer.from_settings({"type": "RollingMean", "settings": {"window_size": 0}})


def test_state_version_mismatch():
    text = _rolling_mean_20().dumps().replace('"version": 1', '"version": 99')
    with pytest.raises(screamer.StateVersionError, match=r"RollingMean.*99.*\b1\b"):
        screamer.loads(text)
    with pytest.warns(UserWarning, match="RollingMean"):
        b = screamer.loads(text, on_mismatch="fresh")
    assert _same(b.state(), screamer.RollingMean(20).state())


def test_on_mismatch_checked():
    with pytest.raises(ValueError, match="on_mismatch"):
        screamer.loads(_rolling_mean_20().dumps(), on_mismatch="ignore")


def test_errors_are_save_errors():
    for cls in (screamer.StateVersionError, screamer.NotSaveable, screamer.StateMismatch):
        assert issubclass(cls, screamer.SaveError)
    assert issubclass(screamer.SaveError, ValueError)


def test_not_saveable_message():
    if not KNOWN_UNSAVEABLE:
        pytest.skip("every operator can be saved")
    name = sorted(KNOWN_UNSAVEABLE)[0]
    make, _ = case(name)
    with pytest.raises(screamer.NotSaveable, match=name):
        make().dumps()
    with pytest.raises(screamer.NotSaveable, match="Segment.op"):
        screamer.Segment(screamer.RollingStd(5)).dumps()


# ---- Nested operators --------------------------------------------------------

def test_nested_op_roundtrip():
    rng = np.random.default_rng(3)
    labels = np.repeat(np.arange(8.0), 9)
    x = rng.standard_normal(labels.size)
    a = screamer.Segment(screamer.RollingMean(5))
    for t in range(40):
        a(labels[t], x[t])
    text = a.dumps()
    node = json.loads(text)["object"]
    assert node["settings"] == {
        "op": {"type": "RollingMean", "settings": {"window_size": 5, "start_policy": "strict"}}}
    assert node["state"]["values"]["op"]["version"] == 1
    loaded = [screamer.loads(text), screamer.loads(a.dumps(format="cbor"))]
    want = [a(labels[t], x[t]) for t in range(40, labels.size)]
    for b in loaded:
        assert b.dumps() == text
        assert_bitwise([b(labels[t], x[t]) for t in range(40, labels.size)], want)


# ---- Golden files ------------------------------------------------------------

def _golden_cases():
    return sorted(p.name[:-len(".case.json")] for p in GOLDEN.glob("*.case.json"))


def _with_library_version(text):
    return re.sub(r'"library_version": "[^"]*"',
                  f'"library_version": "{screamer.__version__}"', text, count=1)


@pytest.mark.parametrize("stem", _golden_cases())
def test_golden(stem):
    name, version = re.fullmatch(r"(\w+)\.v(\d+)", stem).groups()
    scenario = json.loads((GOLDEN / f"{stem}.case.json").read_text(encoding="utf-8"))
    text = _with_library_version((GOLDEN / f"{stem}.json").read_text(encoding="utf-8"))
    prefix = [np.asarray(x, dtype=np.float64) for x in scenario["prefix"]]
    rest = [np.asarray(x, dtype=np.float64) for x in scenario["rest"]]
    current = screamer.from_settings(json.loads(text)["object"]).state()["version"]
    if int(version) != current:
        with pytest.raises(screamer.StateVersionError):
            screamer.loads(text)
        return
    a = screamer.from_settings(json.loads(text)["object"])
    feed(a, prefix, 0, len(prefix[0]))
    assert a.dumps() == text, (
        f"{name}: the state layout differs from the golden of state version {version}; "
        "bump kStateVersion and add a golden with devtools/gen_state_golden.py")
    b = screamer.loads(text)
    assert b.dumps() == text
    want = np.asarray(scenario["rest_output"], dtype=np.float64)
    assert_bitwise(feed(b, rest, 0, len(rest[0])).reshape(want.shape), want)


def test_goldens_cover_saveable():
    """Every saveable operator has a golden of its current state version."""
    missing = []
    for name in SAVEABLE:
        version = case(name)[0]().state()["version"]
        if not (GOLDEN / f"{name}.v{version}.json").exists():
            missing.append(f"{name}.v{version}")
    assert not missing, f"no golden for {missing}; run devtools/gen_state_golden.py"


# ---- The C++ registry and the JavaScript files ---------------------------------

def _settings_value(v):
    """A constructor argument of tests/save_cases.py as a settings value."""
    if isinstance(v, _b.EvalOp):
        return {"type": type(v).__name__, "settings": v.settings()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_settings_value(x) for x in v]
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v)
    return v


@pytest.mark.parametrize("name", FUNCTOR_NAMES)
def test_make_op_builds_every_functor(name):
    """serial::make_op (the JavaScript loader's builder) builds each functor
    as its Python class does: the same outputs, and the same file."""
    make, inputs = case(name)
    settings = {k: _settings_value(v) for k, v in kwargs_for(name).items()}
    built = _b._serial_make_op(name, json.dumps(settings))
    assert_bitwise(np.asarray(built(*inputs)), np.asarray(make()(*inputs)))
    if name not in KNOWN_UNSAVEABLE:
        assert built._dumps(False) == make().dumps()


@pytest.mark.parametrize("settings, where, error", [
    ('{"window": 3}', "RollingMean.settings.window: unknown parameter", screamer.SaveError),
    ('{"window_size": 2.5}', "RollingMean.settings.window_size: expected an integer", screamer.SaveError),
    ('{"window_size": "3"}', "RollingMean.settings.window_size: expected an integer", screamer.SaveError),
    ('{"window_size": 0}', "RollingMean.settings: ", screamer.SaveError),
    ('[]', "RollingMean.settings: expected an object, got an array", screamer.SaveError),
])
def test_make_op_errors_named(settings, where, error):
    with pytest.raises(error, match=re.escape(where)):
        _b._serial_make_op("RollingMean", settings)


def test_make_op_unknown_type_and_nested():
    with pytest.raises(screamer.SaveError, match=re.escape("object.type: unknown type 'Nope'")):
        _b._serial_make_op("Nope", "{}")
    with pytest.raises(screamer.SaveError,
                       match=re.escape("Segment.settings.op.type: unknown type 'Nope'")):
        _b._serial_make_op("Segment", '{"op": {"type": "Nope", "settings": {}}}')
    with pytest.raises(screamer.SaveError, match=re.escape("Segment.settings.op: missing required")):
        _b._serial_make_op("Segment", "{}")
    seg = _b._serial_make_op("Segment", '{"op": {"type": "RollingMean", "settings": {"window_size": 5}}}')
    assert seg._dumps(False) == screamer.Segment(screamer.RollingMean(5)).dumps()


@pytest.fixture
def rename_window():
    """A test row of the rename table: RollingMean's window_size was 'window'."""
    _b._serial_test_rename("RollingMean", "window", "window_size")
    yield
    _b._serial_test_rename_clear()


def test_settings_compat_renames(rename_window):
    a = screamer.from_settings({"type": "RollingMean", "settings": {"window": 7}})
    assert a.settings() == {"window_size": 7, "start_policy": "strict"}
    old = screamer.RollingMean(7).dumps().replace('"window_size"', '"window"')
    assert screamer.loads(old).dumps() == screamer.RollingMean(7).dumps()
    built = _b._serial_make_op("RollingMean", '{"window": 7}')
    assert built._dumps(False) == screamer.RollingMean(7).dumps()
    for both in ({"window": 7, "window_size": 7},):
        with pytest.raises(screamer.SaveError, match="RollingMean.settings.window: renamed to window_size"):
            screamer.from_settings({"type": "RollingMean", "settings": both})
        with pytest.raises(screamer.SaveError, match="RollingMean.settings.window: renamed to window_size"):
            _b._serial_make_op("RollingMean", json.dumps(both))


def test_settings_compat_table_shipped_empty():
    with pytest.raises(screamer.SaveError, match="RollingMean.settings.window: unknown parameter"):
        screamer.from_settings({"type": "RollingMean", "settings": {"window": 7}})


def test_op_registry_fresh():
    gen = ROOT / "devtools" / "gen_op_registry.py"
    committed = (ROOT / "src" / "screamer" / "serial" / "op_registry.cpp").read_text(encoding="utf-8")
    assert run_generator(gen, ROOT) == committed, (
        "src/screamer/serial/op_registry.cpp is stale: run python devtools/gen_op_registry.py")


def _file_tree(text):
    doc = json.loads(text)
    doc["library_version"] = "*"
    return doc


def test_js_oracle_fresh():
    """wasm/smoke/save_oracle.json is what devtools/wasm/gen_save_oracle.py
    writes today. The numbers compare to 1e-9 (tests/_oracle_fresh.py) and
    the library version is ignored."""
    gen = ROOT / "devtools" / "wasm" / "gen_save_oracle.py"
    fixture = ROOT / "wasm" / "smoke" / "save_oracle.json"
    now = json.loads(run_generator(gen, ROOT))
    committed = json.loads(fixture.read_text(encoding="utf-8"))

    def comparable(doc):
        for c in doc["cases"]:
            c["json_text"] = _file_tree(c["json_text"])
            c["cbor_hex"] = _file_tree(_b._serial_cbor_decode(bytes.fromhex(c["cbor_hex"])))
        return doc

    diff = _oracle_same(comparable(now), comparable(committed))
    assert diff is None, (f"wasm/smoke/save_oracle.json is stale ({diff}): run python "
                          "devtools/wasm/gen_save_oracle.py and commit the result")
    assert [c["name"] for c in now["cases"]] == SAVEABLE


def test_from_settings_takes_python_values():
    """Settings built in Python (NumPy integers, an operator object where a
    {type, settings} node belongs) are read as before, without a JSON pass."""
    a = screamer.from_settings({"type": "RollingMean", "settings": {"window_size": np.int64(5)}})
    assert a.settings()["window_size"] == 5
    with pytest.raises(screamer.SaveError, match=re.escape("Segment.settings.op: expected")):
        screamer.from_settings({"type": "Segment", "settings": {"op": screamer.RollingMean(3)}})


def test_nested_setting_errors_name_the_path():
    node = {"type": "Segment", "settings": {"op": {"type": "RollingMean", "settings": {"window": 3}}}}
    where = "Segment.settings.op.settings.window: unknown parameter"
    with pytest.raises(screamer.SaveError, match=re.escape(where)):
        screamer.from_settings(node)
    with pytest.raises(screamer.SaveError, match=re.escape(where)):
        _b._serial_make_op("Segment", json.dumps(node["settings"]))
    bad = '{"op": {"type": "RollingMean", "settings": {"window_size": 0}}}'
    with pytest.raises(screamer.SaveError, match=re.escape("Segment.settings.op.settings: ")):
        _b._serial_make_op("Segment", bad)


def _js_fixture():
    return json.loads((ROOT / "wasm" / "smoke" / "save_js.json").read_text(encoding="utf-8"))


_VERSION_KEY = b"\x6flibrary_version"   # CBOR: a text string of 15 bytes


def _cbor_with_library_version(blob, version):
    """``blob`` (a CBOR save file) with its library_version set to ``version``."""
    at = blob.index(_VERSION_KEY) + len(_VERSION_KEY)
    n = blob[at] - 0x60                      # a text string shorter than 24 bytes
    assert 0 <= n < 24 and len(version) < 24
    return blob[:at] + bytes([0x60 + len(version)]) + version.encode() + blob[at + 1 + n:]


def _check_js_file(c):
    """One file JavaScript wrote: Python loads it and writes the same bytes,
    in JSON and CBOR, but for the library version, which is this library's."""
    text, blob = c["json_text"], bytes.fromhex(c["cbor_hex"])
    want_text = _with_library_version(text)
    want_blob = _cbor_with_library_version(blob, screamer.__version__)
    a, b = screamer.loads(text), screamer.loads(blob)
    assert a.dumps() == want_text and b.dumps() == want_text
    assert a.dumps(format="cbor") == want_blob and b.dumps(format="cbor") == want_blob
    return a


def test_js_files_load_byte_for_byte():
    """The files JavaScript wrote (js/scripts/gen_save_js.ts) load in Python
    and write back the same bytes, in JSON and CBOR, and continue as
    Python's own file does (to 1e-9: D4, see js/test/save_restore.test.ts)."""
    fixture = _js_fixture()
    assert [c["name"] for c in fixture] == SAVEABLE, (
        "wasm/smoke/save_js.json is stale: run node --import tsx js/scripts/gen_save_js.ts")
    for c in fixture:
        a = _check_js_file(c)
        make, inputs = case(c["name"])
        k = splits(c["name"], kwargs_for(c["name"]))[2]
        ref = make()
        feed(ref, inputs, 0, k)
        np.testing.assert_allclose(feed(a, inputs, k, k + REST), feed(ref, inputs, k, k + REST),
                                   rtol=1e-9, atol=1e-9, equal_nan=True)


def test_js_files_of_another_library_version():
    """A fixture written by another library version still compares byte for
    byte (the version is the one field that differs)."""
    for c in _js_fixture():
        old = {"json_text": re.sub(r'"library_version": "[^"]*"', '"library_version": "0.0.1"',
                                   c["json_text"], count=1),
               "cbor_hex": _cbor_with_library_version(bytes.fromhex(c["cbor_hex"]), "0.0.1").hex()}
        assert '"0.0.1"' in old["json_text"] and b"0.0.1" in bytes.fromhex(old["cbor_hex"])
        _check_js_file(old)


# ---- Calls with one event on an object holding per-column state ---------------

def _panel_inputs(name, inputs):
    """Inputs of a (T, K) resume: two columns per input, or the block itself."""
    if name in _sweep.BLOCK_OPS:
        return list(inputs)
    if any(np.ndim(x) != 1 for x in inputs):
        return None
    return [np.stack([x, x[::-1]], axis=1) for x in inputs]


@pytest.mark.parametrize("name", FUNCTOR_NAMES)
def test_event_calls_on_a_panel_raise(name):
    """After a resume on a (T, K) array every call with one event, and every
    stream, raises (decision 9) and keeps the per-column state."""
    make, inputs = case(name)
    panel = _panel_inputs(name, inputs)
    op = make()
    try:
        op.resume(*(panel or []))
    except (TypeError, ValueError):
        pytest.skip(f"{name} keeps no per-column state")
    if panel is None:
        pytest.skip(f"{name} keeps no per-column state")
    k = panel[0].shape[1]
    if name in _sweep.BLOCK_OPS:
        row = panel[0][0]
        calls = [lambda: op(*[float(v) for v in row[0]]), lambda: op(row),
                 lambda: op(row, np.ones(k)), lambda: op(iter([row])),
                 lambda: op(iter([row]), iter([np.ones(k)]))]
    else:
        ev = [float(x[0, 0]) for x in panel]
        calls = [lambda: op(*ev), lambda: op(*[iter([v]) for v in ev])]
        if len(ev) > 1:
            calls += [lambda: op(tuple(ev)), lambda: op(iter([tuple(ev)]))]

        async def agen(v):
            yield v
        calls.append(lambda: op(*[agen(v) for v in ev]))
    for call in calls:
        with pytest.raises(ValueError, match=f"per-column state for {k} columns"):
            call()
    with pytest.raises(ValueError, match=f"holds {k} columns, got {k + 1}"):
        op.resume(*[np.concatenate([x, x[:, :1]], axis=1) for x in panel])
    op.resume(*panel)    # the columns are still there
