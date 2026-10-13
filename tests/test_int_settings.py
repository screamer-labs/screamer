"""An integer setting below its smallest value raises ValueError with one
message form, "<Op>: <setting> must be at least <min>, got <v>" (C++
int_settings.h), never the raw text of the operator's buffers ("vector")."""
import json
import pathlib

import pytest

import screamer as s
from screamer import screamer_bindings as _b

ROOT = pathlib.Path(__file__).resolve().parent.parent
MANIFEST = json.loads((ROOT / "devtools" / "wasm" / "wasm_manifest.json").read_text())
ORACLE = {e["name"]: e for e in json.loads((ROOT / "wasm" / "smoke" / "null_oracle.json").read_text())}
INTS = [(op["name"], a["name"]) for op in MANIFEST
        for t, a in zip(op["ctor"], op["ctor_args"]) if t == "int"]


def test_every_integer_setting_has_a_smallest_value():
    assert len(INTS) > 120
    missing = [x for x in INTS if _b._int_setting_min(*x) is None]
    assert not missing


@pytest.mark.parametrize("op, name", INTS, ids=[f"{o}.{n}" for o, n in INTS])
def test_below_the_smallest_value_raises_one_message(op, name):
    low = _b._int_setting_min(op, name)
    base = dict(ORACLE.get(op, {}).get("base", {}))
    for v in (low - 1, -1 if low > 0 else low - 2, -10 ** 9):
        with pytest.raises(ValueError) as e:
            getattr(s, op)(**dict(base, **{name: v}))
        assert str(e.value) == f"{op}: {name} must be at least {low}, got {v}"


def test_a_negative_window_names_the_setting():
    with pytest.raises(ValueError, match=r"^RollingMean: window_size must be at least 1, got -1$"):
        s.RollingMean(-1)
    with pytest.raises(ValueError, match=r"^Butter: order must be at least 1, got 0$"):
        s.Butter(0, 0.1)
