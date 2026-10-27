"""Freshness gate for wasm/smoke/oracle_iir_start.json.

devtools/wasm/gen_iir_start_oracle.py records Python's seven IIR filters under
both start policies; js/test/iir_start.test.ts checks that JavaScript gives the
same values. This test fails when the committed fixture no longer matches the
Python binding.
"""
import pathlib

from tests._oracle_fresh import assert_oracle_fresh

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
GEN = REPO_ROOT / "devtools" / "wasm" / "gen_iir_start_oracle.py"
FIXTURE = REPO_ROOT / "wasm" / "smoke" / "oracle_iir_start.json"


def test_iir_start_oracle_is_fresh():
    assert_oracle_fresh(GEN, FIXTURE, REPO_ROOT)
