"""Freshness gate for wasm/smoke/oracle_research.json.

devtools/wasm/gen_research_oracle.py records the research-toolkit operators
and backtest helpers as the Python binding computes them;
js/test/research.test.ts checks that JavaScript gives the same values. This
test fails when the committed fixture no longer matches the Python binding.
"""
import pathlib

import pytest

from tests._oracle_fresh import assert_oracle_fresh

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
GEN = REPO_ROOT / "devtools" / "wasm" / "gen_research_oracle.py"
FIXTURE = REPO_ROOT / "wasm" / "smoke" / "oracle_research.json"

pytestmark = pytest.mark.skipif(not GEN.exists(), reason="needs the source tree")


def test_research_oracle_is_fresh():
    assert_oracle_fresh(GEN, FIXTURE, REPO_ROOT)
