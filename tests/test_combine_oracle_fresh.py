"""Freshness gate for wasm/smoke/oracle_combine_latest.json.

devtools/wasm/gen_combine_oracle.py records Python's combine_latest on a fixed
set of inputs; js/test/combine_latest_parity.test.ts checks that JavaScript's
combineLatest gives the same results. This test fails when the committed
fixture no longer matches the Python binding.
"""
import pathlib
import subprocess
import sys

from tests._oracle_fresh import assert_oracle_fresh

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
GEN = REPO_ROOT / "devtools" / "wasm" / "gen_combine_oracle.py"
FIXTURE = REPO_ROOT / "wasm" / "smoke" / "oracle_combine_latest.json"


def test_combine_oracle_is_fresh():
    assert_oracle_fresh(GEN, FIXTURE, REPO_ROOT)
