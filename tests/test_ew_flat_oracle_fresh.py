"""Freshness gate for wasm/smoke/oracle_ew_flat.json.

devtools/wasm/gen_ew_flat_oracle.py records the exponentially weighted ratios
of a series that stops varying as the Python binding computes them;
js/test/ew_flat.test.ts checks that JavaScript gives the same values. This
test fails when the committed fixture no longer matches the Python binding.
"""
import pathlib

import pytest

from tests._oracle_fresh import assert_oracle_fresh

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
GEN = REPO_ROOT / "devtools" / "wasm" / "gen_ew_flat_oracle.py"
FIXTURE = REPO_ROOT / "wasm" / "smoke" / "oracle_ew_flat.json"

pytestmark = pytest.mark.skipif(not GEN.exists(), reason="needs the source tree")


def test_ew_flat_oracle_is_fresh():
    assert_oracle_fresh(GEN, FIXTURE, REPO_ROOT)
