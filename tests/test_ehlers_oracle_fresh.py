"""Freshness gate for wasm/smoke/oracle_ehlers.json.

devtools/wasm/gen_ehlers_oracle.py records Python's CycleSine, TrendMode,
InstantaneousTrendline, MAMA and HilbertPhasor on fixed inputs;
js/test/ehlers_parity.test.ts checks that JavaScript gives the same values.
This test fails when the committed fixture no longer matches the Python
binding.
"""
import pathlib
import subprocess
import sys

from tests._oracle_fresh import assert_oracle_fresh

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
GEN = REPO_ROOT / "devtools" / "wasm" / "gen_ehlers_oracle.py"
FIXTURE = REPO_ROOT / "wasm" / "smoke" / "oracle_ehlers.json"


def test_ehlers_oracle_is_fresh():
    assert_oracle_fresh(GEN, FIXTURE, REPO_ROOT)
