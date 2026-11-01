"""The serial core of save and load (include/screamer/serial, common/json.h):
the tree text, the CBOR codec, the save envelope and the archives.

The archives have no operator users here; they run on a private demo object
(bindings_core.cpp, ``_SerialDemo``) with a double, an integer, a vector, a
fixed-length vector, a nested block and a nested object saved through its own
write_state/read_state, like a nested operator.
"""
import json
import math
import pathlib
import struct
import sys

import pytest

import screamer
from screamer import screamer_bindings as _b

ROOT = pathlib.Path(__file__).resolve().parent.parent
TO_JSON_FIXTURE = ROOT / "tests" / "data" / "pipeline_to_json.json"


def _bits(x):
    return struct.pack("<d", x)


def _same_double(got, want):
    if math.isnan(want):
        return math.isnan(got)
    return _bits(got) == _bits(want)


# ---- Tree text ---------------------------------------------------------------

def test_dump_save_layout():
    assert _b._serial_dump_save('{"b":[1.0,2.0],"a":{"x":1}}') == \
        '{\n  "b": [1.0, 2.0],\n  "a": {\n    "x": 1\n  }\n}'


def test_dump_save_nested_arrays_and_empties():
    text = '{"m":[[1,2],[3.5]],"e":[],"o":{},"s":"a\\"b","n":null,"t":true}'
    assert _b._serial_dump_save(text) == (
        '{\n  "m": [\n    [1, 2],\n    [3.5]\n  ],\n  "e": [],\n  "o": {},\n'
        '  "s": "a\\"b",\n  "n": null,\n  "t": true\n}')


VALUES = [float("nan"), float("inf"), -float("inf"), -0.0, 5e-324, 0.1, 1.7976931348623157e308]


@pytest.mark.parametrize("x", VALUES, ids=repr)
def test_nonfinite_roundtrip(x):
    src = json.dumps({"v": [x], "w": x})
    for out in (_b._serial_dump_save(src), _b._serial_cbor_roundtrip(src)):
        back = json.loads(out)
        assert _same_double(back["v"][0], x), out
        assert _same_double(back["w"], x), out


def test_cbor_preserves_key_order_and_ints():
    src = '{"z":1,"a":-2,"m":[true,null,"s"]}'
    back = json.loads(_b._serial_cbor_roundtrip(src))
    assert list(back) == ["z", "a", "m"]
    assert back == {"z": 1, "a": -2, "m": [True, None, "s"]}
    assert type(back["z"]) is int and type(back["a"]) is int


@pytest.mark.parametrize("text, hexbytes", [
    ("0", "00"), ("23", "17"), ("24", "1818"), ("1000", "1903e8"), ("-1", "20"),
    ("-1000", "3903e7"), ("9223372036854775807", "1b7fffffffffffffff"),
    ("18446744073709551615", "1bffffffffffffffff"),
    ("-9223372036854775808", "3b7fffffffffffffff"),
    ("1.0", "fb3ff0000000000000"), ("-0.0", "fb8000000000000000"),
    ("Infinity", "fb7ff0000000000000"),
    ("true", "f5"), ("false", "f4"), ("null", "f6"),
    ('"a"', "6161"), ('"\\u00fc"', "62c3bc"),
    ("[1, [2, 3]]", "8201820203"), ('{"a": 1, "b": [2, 3]}', "a26161016162820203"),
])
def test_cbor_bytes_follow_rfc8949(text, hexbytes):
    assert _b._serial_cbor_encode(text).hex() == hexbytes
    assert json.loads(_b._serial_cbor_decode(bytes.fromhex(hexbytes))) == json.loads(text)


def test_truncated_cbor_raises():
    data = _b._serial_cbor_encode('{"z":1,"a":-2.5,"m":[true,null,"string"],"k":{"x":[1e300]}}')
    for n in range(len(data)):
        with pytest.raises(ValueError, match="CBOR"):
            _b._serial_cbor_decode(data[:n])
    assert json.loads(_b._serial_cbor_decode(data))["k"] == {"x": [1e300]}


@pytest.mark.parametrize("hexbytes, what", [
    ("c000", "tag"), ("5f", "indefinite"), ("9f01ff", "indefinite"), ("4161", "byte string"),
    ("f7", "simple value"), ("a10101", "key"), ("0000", "extra data"),
    ("7b7fffffffffffffff", "truncated"), ("1c", "additional information"),
])
def test_unsupported_cbor_raises(hexbytes, what):
    with pytest.raises(ValueError, match=f"CBOR.*{what}"):
        _b._serial_cbor_decode(bytes.fromhex(hexbytes))


@pytest.mark.parametrize("hexbytes, want", [
    ("f93c00", 1.0), ("f97c00", math.inf), ("f98000", -0.0), ("f90001", 2.0 ** -24),
    ("fa47c35000", 100000.0),
])
def test_cbor_reads_float16_and_float32(hexbytes, want):
    got = json.loads(_b._serial_cbor_decode(bytes.fromhex(hexbytes)))
    assert _bits(got) == _bits(want)


@pytest.mark.parametrize("hexbytes", ["62c328", "61ff", "63eda080", "62c080", "64f4908080", "62e282"])
def test_cbor_refuses_invalid_utf8(hexbytes):
    with pytest.raises(ValueError, match="CBOR.*invalid UTF-8"):
        _b._serial_cbor_decode(bytes.fromhex(hexbytes))


def test_cbor_nested_huge_lengths_raise():
    # 512 nested arrays, each claiming as many elements as the bytes left: a
    # reservation by the claimed length would ask for tens of gigabytes.
    n = 1 << 20
    data = bytearray()
    for _ in range(512):
        data += b"\x9b" + (n - len(data) - 9).to_bytes(8, "big")
    data += b"\x00" * (n - len(data))
    with pytest.raises(ValueError, match="CBOR"):
        _b._serial_cbor_decode(bytes(data))


def test_cbor_raises_a_save_error():
    with pytest.raises(_b._SaveError):
        _b._serial_cbor_decode(b"\xa1")
    assert issubclass(_b._SaveError, ValueError)
    for name in ("StateVersionError", "NotSaveable", "StateMismatch"):
        assert issubclass(getattr(_b, "_" + name), _b._SaveError)


def test_cbor_deep_nesting_raises():
    with pytest.raises(ValueError, match="CBOR.*deeply"):
        _b._serial_cbor_decode(b"\x81" * 100000)


# ---- Envelope ----------------------------------------------------------------

def test_envelope_wraps_with_schema_and_library_version():
    wrapped = json.loads(_b._serial_wrap('{"type": "Demo"}'))
    assert list(wrapped) == ["screamer", "library_version", "object"]
    assert wrapped["screamer"] == "save/1"
    assert wrapped["library_version"] == screamer.__version__
    assert json.loads(_b._serial_check_envelope(json.dumps(wrapped))) == {"type": "Demo"}


def test_envelope_rejects_other_schema():
    text = json.dumps({"screamer": "save/2", "library_version": "9.0.0", "object": {}})
    with pytest.raises(_b._SaveError, match="save/2"):
        _b._serial_check_envelope(text)


@pytest.mark.parametrize("text, what", [
    ("[1]", "an array"), ('{"object": {}}', '"screamer"'), ('{"screamer": 1, "object": {}}', "a number"),
    ('{"screamer": "save/1"}', '"object"'),
])
def test_envelope_rejects_other_files(text, what):
    with pytest.raises(_b._SaveError, match=what):
        _b._serial_check_envelope(text)


# ---- Archives ----------------------------------------------------------------

def _demo_state():
    d = _b._SerialDemo(3)
    return d, json.loads(d.state())


def test_archive_writes_every_field_in_order():
    d, s = _demo_state()
    assert list(s) == ["x", "n", "v", "fixed", "block", "inner", "flag", "label", "opt"]
    assert s["x"] == 1.5 and s["n"] == 1 and s["v"] == [1.0, 2.0]
    assert s["fixed"] == [0.0, 0.0, 0.0]
    assert s["block"] == {"a": -0.0, "buf": [1.0, 2.0, 3.0], "count": 2}
    assert s["inner"] == {"version": 1, "values": {"total": 10.0, "seen": 4}}
    assert s["opt"] is None


def test_archive_reads_a_changed_tree():
    d, s = _demo_state()
    s.update(x=math.nan, n=-3, v=[], fixed=[1.0, math.inf, -0.0], flag=True, label="é", opt=2.0)
    s["block"].update(buf=[5.0], count=1)
    s["inner"]["values"]["total"] = 99.0
    d.set_state(json.dumps(s))
    got = json.loads(d.state())
    assert math.isnan(got.pop("x"))
    s.pop("x")
    assert got == s
    assert _bits(got["fixed"][2]) == _bits(-0.0)


def _edit(fn):
    d, s = _demo_state()
    fn(s)
    return d, json.dumps(s)


BAD = [
    ("unknown field", lambda s: s.update(zz=1), r"^demo\.zz: unknown field$"),
    ("missing field", lambda s: s.pop("n"), r"^demo\.n: missing field$"),
    ("wrong type", lambda s: s.update(x="1.5"), r"^demo\.x: expected a number, got a string$"),
    ("int as float", lambda s: s.update(n=1.0), r"^demo\.n: expected an integer, got 1\.0$"),
    ("int out of range", lambda s: s.update(n=2 ** 40),
     r"^demo\.n: expected an integer from -2147483648 to 2147483647, got 1099511627776$"),
    ("bool type", lambda s: s.update(flag=1), r"^demo\.flag: expected a boolean, got 1$"),
    ("vector type", lambda s: s.update(v={"a": 1}), r"^demo\.v: expected an array, got an object$"),
    ("element type", lambda s: s.update(v=[1.0, None]), r"^demo\.v\[1\]: expected a number, got null$"),
    ("wrong length", lambda s: s.update(fixed=[1.0, 2.0]), r"^demo\.fixed: expected 3 values, got 2$"),
    ("nested unknown", lambda s: s["block"].update(q=1), r"^demo\.block\.q: unknown field$"),
    ("nested missing", lambda s: s["block"].pop("buf"), r"^demo\.block\.buf: missing field$"),
    ("block type", lambda s: s.update(block=[1]), r"^demo\.block: expected an object, got an array$"),
    ("check", lambda s: s["block"].update(count=5),
     r"^demo\.block: count must be at most the buffer length$"),
    ("top check", lambda s: s.update(n=-1, x=-2.0), r"^demo: x must not be below n$"),
    ("inner field", lambda s: s["inner"]["values"].pop("seen"), r"^demo\.inner\.seen: missing field$"),
    ("inner version", lambda s: s["inner"].update(version=99),
     r"^demo\.inner: state version 99 in the file, this library reads version 1$"),
]


@pytest.mark.parametrize("fn, message", [b[1:] for b in BAD], ids=[b[0] for b in BAD])
def test_archive_refuses_with_the_path_and_leaves_the_object_unchanged(fn, message):
    d, text = _edit(fn)
    # Change the object first, so "unchanged" is not just the constructor's state.
    s = json.loads(d.state())
    s.update(x=4.25, v=[9.0], fixed=[1.0, 2.0, 3.0])
    s["block"]["count"] = 3
    s["inner"]["values"]["seen"] = 11
    d.set_state(json.dumps(s))
    before = d.state()
    # A bad field late in the tree: earlier fields were already read and must be undone.
    with pytest.raises(_b._SaveError, match=message):
        d.set_state(text)
    assert d.state() == before


def test_archive_errors_have_their_class():
    d, text = _edit(lambda s: s["inner"].update(version=2))
    with pytest.raises(_b._StateVersionError):
        d.set_state(text)
    d, text = _edit(lambda s: s.update(zz=1))
    with pytest.raises(_b._StateMismatch):
        d.set_state(text)


# ---- Pipeline.to_json ------------------------------------------------------------

def _cases():
    sys.path.insert(0, str(ROOT / "devtools"))
    try:
        import pipeline_io_cases
    finally:
        sys.path.pop(0)
    return pipeline_io_cases.cases()


def test_pipeline_to_json_unchanged():
    # The to_json text of every graph of devtools/pipeline_io_cases.py, as the
    # code before the serial core wrote it.
    want = json.loads(TO_JSON_FIXTURE.read_text(encoding="utf-8"))
    cases = _cases()
    assert set(cases) == set(want)
    for name, (pipe, _) in cases.items():
        assert pipe.to_json(indent=None) == want[name]["none"], name
        assert pipe.to_json(indent=2) == want[name]["2"], name


# ---- Nesting depth -------------------------------------------------------------

@pytest.mark.parametrize("n", [128, 129, 600])
def test_parse_depth_limit(n):
    """Both readers take values nested 128 deep (json.h kMaxDepth) and refuse
    a deeper file, which would overflow the WebAssembly stack."""
    text = "[" * n + "0" + "]" * n
    blob = b"\x81" * n + b"\x00"
    if n <= 128:
        assert json.loads(_b._serial_dump_save(text)) == json.loads(text)
        assert json.loads(_b._serial_cbor_decode(blob)) == json.loads(text)
        return
    with pytest.raises(ValueError, match="JSON.*nested too deeply"):
        _b._serial_dump_save(text)
    with pytest.raises(ValueError, match="CBOR.*nested too deeply"):
        _b._serial_cbor_decode(blob)


def test_envelope_refuses_unknown_fields():
    text = json.dumps({"screamer": "save/1", "library_version": "1", "object": {}, "note": 1})
    with pytest.raises(_b._SaveError, match="^note: unknown field"):
        _b._serial_check_envelope(text)
