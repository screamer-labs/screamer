"""B8: the saved form of a Pipeline is one C++ schema, reader and writer
(include/screamer/dag/graph_json.h), shared with JavaScript.

- to_json writes the text json.dumps writes, at every indent;
- graphs JavaScript saved (wasm/smoke/pipeline_io_js.json, written by
  js/test/pipeline_io.test.ts) load here and give JavaScript's results;
- wasm/smoke/pipeline_io_oracle.json, which the JavaScript suite loads, is
  fresh;
- a malformed text or description raises ValueError naming the problem.
"""
import json
import math
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from screamer import Input, Pipeline, RollingMean
from screamer import screamer_bindings as _b

ROOT = pathlib.Path(__file__).resolve().parent.parent
JS_FIXTURE = ROOT / "wasm" / "smoke" / "pipeline_io_js.json"
ORACLE = ROOT / "wasm" / "smoke" / "pipeline_io_oracle.json"
GEN = ROOT / "devtools" / "wasm" / "gen_pipeline_io_oracle.py"


def test_the_oracle_for_javascript_is_fresh():
    r = subprocess.run([sys.executable, str(GEN), "--stdout"], cwd=ROOT, capture_output=True)
    assert r.returncode == 0, r.stderr.decode(errors="replace")
    assert r.stdout == ORACLE.read_bytes(), (
        "wasm/smoke/pipeline_io_oracle.json is stale: re-run "
        "python devtools/wasm/gen_pipeline_io_oracle.py and commit the result.")


def _description(params):
    return {"screamer_dag": 1, "inputs": ["x"], "align_outputs": True, "max_pending": 7,
            "nodes": [{"id": 0, "kind": "input", "name": "x"},
                      {"id": 1, "kind": "functor", "cls": "Linear", "params": params, "in": [0]}],
            "outputs": [1]}


RNG = np.random.default_rng(8)
FLOATS = ([0.0, -0.0, 1.0, 0.1, 1 / 3, 1e-5, 1e-4, 0.0001234, 1e16, 1e15, 123456789012345678.0,
           1.5e-300, 5e-324, 1.7976931348623157e308, 2.5, -7.25, math.pi, 100.0, 1e22, 1e-7]
          + list(RNG.standard_normal(200) * 10.0 ** RNG.integers(-12, 12, 200)))


@pytest.mark.parametrize("indent", [None, 0, 1, 2, 4])
def test_to_json_writes_what_json_dumps_writes(indent):
    params = {"scale": FLOATS, "shift": [1, -2, 2 ** 62, True, False, None],
              "text": ["a\"b\\c\n\t\x01", "é ü 日本 😀", ""], "nested": {"k": [[], {}], "e": {}},
              "odd": [math.inf, -math.inf, math.nan]}
    d = _description(params)
    assert _b._graph_to_json(d, indent) == json.dumps(d, indent=indent)
    back = _b._graph_from_json(json.dumps(d, indent=indent))
    assert json.dumps(back) == json.dumps(d)


def _plain_outputs(result):
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], np.ndarray) \
            and not isinstance(result[1], tuple):
        result = (result,)
    return [(np.asarray(v, dtype=float), np.asarray(i)) for v, i in result]


def _feeds(entry):
    out = []
    for f in entry["feeds"]:
        v = np.array([[np.nan if x is None else x for x in r] if isinstance(r, list) else
                      (np.nan if r is None else r) for r in f["values"]], dtype=float)
        out.append((v, np.asarray(f["index"], dtype=np.int64)))
    return out


def _close(got, want, what):
    assert len(got) == len(want), what
    for (gv, gi), w in zip(got, want):
        wv = np.array([[np.nan if x is None else x for x in r] if isinstance(r, list) else
                       (np.nan if r is None else r) for r in w["values"]], dtype=float)
        np.testing.assert_array_equal(gi, np.asarray(w["index"]), err_msg=what)
        np.testing.assert_allclose(gv.reshape(wv.shape), wv, rtol=1e-9, atol=1e-12, equal_nan=True,
                                   err_msg=what)


JS = json.loads(JS_FIXTURE.read_text())
PY = {e["name"]: e for e in json.loads(ORACLE.read_text())}


@pytest.mark.parametrize("entry", JS, ids=[e["name"] for e in JS])
def test_a_graph_javascript_saved_runs_to_its_results(entry):
    pipe = Pipeline.from_json(entry["json"])
    feeds = _feeds(entry)
    got = _plain_outputs(pipe(*feeds))
    _close(got, entry["outputs"], entry["name"])
    if entry["name"] in PY:
        # The same description as the graph Python saved, and its results.
        same = _b._graph_from_json(PY[entry["name"]]["json"])
        assert _equal_numbers(_b._graph_from_json(entry["json"]), same)
        _close(got, PY[entry["name"]]["outputs"], entry["name"] + " (Python)")
    # Saved again by Python, the text loads to the same description.
    assert _equal_numbers(_b._graph_from_json(pipe.to_json()), _b._graph_from_json(entry["json"]))


def test_javascript_native_graphs_are_covered():
    names = {e["name"] for e in JS}
    assert {"js: chain", "js: filter and clip", "js: resample threshold and clock"} <= names


def _equal_numbers(a, b):
    """Equal descriptions: 3 equals 3.0 (JavaScript has one number type), NaN equals NaN."""
    if isinstance(a, dict) and isinstance(b, dict):
        return list(a) == list(b) and all(_equal_numbers(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_equal_numbers(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) \
            and not isinstance(a, bool) and not isinstance(b, bool):
        return a == b or (math.isnan(a) and math.isnan(b))
    return a == b and type(a) is type(b)


def _graph():
    x = Input("x")
    return Pipeline([x], [RollingMean(3)(x)])


@pytest.mark.parametrize("text, message", [
    ("{", r"^invalid Pipeline JSON: expected a property name in double quotes at line 1, column 2$"),
    ('{"a": 1} x', r"^invalid Pipeline JSON: extra data after the JSON value at line 1, column 10$"),
    ('{"a": [1, 2', r"^invalid Pipeline JSON: expected ',' or '\]' in an array at line 1, column 12$"),
    ('{"a": "\x01"}', r"^invalid Pipeline JSON: a control character in a string at line 1, column 8$"),
    ("[]", r"^Pipeline description: a Pipeline description is a JSON object, got an array$"),
    ('{"screamer_dag": 2}', r"^unsupported screamer_dag version: 2 \(expected 1\)$"),
    ('{"inputs": []}', r"^unsupported screamer_dag version: none \(expected 1\)$"),
])
def test_a_malformed_text_raises_value_error(text, message):
    with pytest.raises(ValueError, match=message):
        Pipeline.from_json(text)


def _with(change):
    d = _graph().to_dict()
    change(d)
    return d


@pytest.mark.parametrize("change, message", [
    (lambda d: d["nodes"][1].update(kind="x"), r"^unknown node kind: 'x'$"),
    (lambda d: d["nodes"][1].update(cls="NoSuchOp"), r"^unknown functor class: 'NoSuchOp'$"),
    (lambda d: d["nodes"][1].update(kind="operator", op="Nope"), r"^unknown DAG operator: 'Nope'$"),
    (lambda d: d["nodes"][1].update(**{"in": [7]}),
     r"^Pipeline description: node 1: input 7 is not an earlier node$"),
    (lambda d: d["nodes"][1].update(kind="operator", op="Delay", params={"k": 1}),
     r"^Pipeline description: node 1 \(Delay\): unknown parameter 'k'$"),
    (lambda d: d["nodes"][1].update(kind="operator", op="Filter", params={}),
     r"^Pipeline description: node 1 \(Filter\) takes 2 inputs, got 1$"),
    (lambda d: d.update(inputs=["y"]),
     r"""^Pipeline description: "inputs" names 'y', which is not an input node$"""),
    (lambda d: d.update(outputs=[9]), r"^Pipeline description: output 9 is not a node$"),
    (lambda d: d.update(align_outputs="yes"),
     r"""^Pipeline description: "align_outputs" must be true or false, got 'yes'$"""),
    (lambda d: d["nodes"].append(dict(d["nodes"][0])),
     r"^Pipeline description: node 2: the id 0 is used twice$"),
])
def test_a_malformed_description_raises_value_error(change, message):
    d = _with(change)
    with pytest.raises(ValueError, match=message):
        Pipeline.from_dict(d)
    with pytest.raises(ValueError, match=message):
        Pipeline.from_json(json.dumps(d))


def test_a_setting_its_constructor_refuses_raises_as_the_constructor_does():
    d = _with(lambda d: d["nodes"][1].update(params={"window_size": "3"}))
    with pytest.raises(TypeError, match="^RollingMean: window_size must be an integer, got str$"):
        Pipeline.from_dict(d)


def test_from_json_takes_text():
    with pytest.raises(TypeError, match="takes the JSON text"):
        Pipeline.from_json(_graph().to_dict())


def test_legacy_operator_names_load():
    x = Input("x")
    d = _with(lambda d: None)
    d["nodes"][1] = {"id": 1, "kind": "operator", "op": "resample",
                     "params": {"every": 2, "agg": "last"}, "in": [0]}
    p = Pipeline.from_dict(d)
    assert p.to_dict()["nodes"][1]["op"] == "Resample"
