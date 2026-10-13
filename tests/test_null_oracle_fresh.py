"""Freshness gate for wasm/smoke/null_oracle.json.

devtools/wasm/gen_null_oracle.py records, for every operator setting, whether
the Python binding takes None for it; js/test/ops.test.ts checks that
JavaScript takes null for exactly the same settings. This test fails when the
committed fixture no longer matches the Python binding.
"""
import pathlib
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
GEN = REPO_ROOT / "devtools" / "wasm" / "gen_null_oracle.py"
FIXTURE = REPO_ROOT / "wasm" / "smoke" / "null_oracle.json"
MANIFEST = REPO_ROOT / "devtools" / "wasm" / "wasm_manifest.json"

pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="needs the source tree")


def test_null_oracle_is_fresh():
    r = subprocess.run([sys.executable, str(GEN), "--stdout"], cwd=REPO_ROOT, capture_output=True)
    assert r.returncode == 0, r.stderr.decode(errors="replace")
    assert r.stdout == FIXTURE.read_bytes(), (
        "wasm/smoke/null_oracle.json is stale: re-run "
        "python devtools/wasm/gen_null_oracle.py and commit the result."
    )


def test_null_oracle_matches_the_optional_settings():
    # Python takes None for exactly the std::optional settings, the rule the
    # JavaScript generator applies to null.
    import json
    manifest = {op["name"]: op for op in json.loads(MANIFEST.read_text())}
    for entry in json.loads(FIXTURE.read_text()):
        op = manifest[entry["name"]]
        for cpp_type, arg in zip(op["ctor"], op["ctor_args"]):
            assert entry["accepts_none"][arg["name"]] == cpp_type.startswith("std::optional<"), (entry["name"], arg["name"])
