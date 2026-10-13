"""Freshness gate for wasm/smoke/null_oracle.json.

devtools/wasm/gen_null_oracle.py records, for every operator setting, whether
the Python binding takes None for it; js/test/ops.test.ts checks that
JavaScript takes null for exactly the same settings. This test fails when the
committed fixture no longer matches the Python binding.
"""
import pathlib
import subprocess
import sys

from tests._oracle_fresh import assert_oracle_fresh

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
GEN = REPO_ROOT / "devtools" / "wasm" / "gen_null_oracle.py"
FIXTURE = REPO_ROOT / "wasm" / "smoke" / "null_oracle.json"
MANIFEST = REPO_ROOT / "devtools" / "wasm" / "wasm_manifest.json"

pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="needs the source tree")


def test_null_oracle_is_fresh():
    assert_oracle_fresh(GEN, FIXTURE, REPO_ROOT)


def test_null_oracle_matches_the_optional_settings():
    # Python takes None for exactly the std::optional settings, the rule the
    # JavaScript generator applies to null.
    import json
    manifest = {op["name"]: op for op in json.loads(MANIFEST.read_text())}
    for entry in json.loads(FIXTURE.read_text()):
        op = manifest[entry["name"]]
        for cpp_type, arg in zip(op["ctor"], op["ctor_args"]):
            assert entry["accepts_none"][arg["name"]] == cpp_type.startswith("std::optional<"), (entry["name"], arg["name"])


def test_null_oracle_bools_match_the_setting_types():
    # A bool is not a number setting (G25): Python refuses True for every
    # number, integer and vector setting, the rule JavaScript applies to true.
    import json
    manifest = {op["name"]: op for op in json.loads(MANIFEST.read_text())}
    for entry in json.loads(FIXTURE.read_text()):
        op = manifest[entry["name"]]
        for cpp_type, arg in zip(op["ctor"], op["ctor_args"]):
            assert entry["accepts_bool"][arg["name"]] == (cpp_type == "bool"), (entry["name"], arg["name"])
