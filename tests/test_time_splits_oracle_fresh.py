"""Freshness gate for wasm/smoke/oracle_time_splits.json.

devtools/wasm/gen_time_splits_oracle.py records Python's time_splits folds;
js/test/time_splits.test.ts checks that JavaScript gives the same folds. This
test fails when the committed fixture no longer matches the Python binding.
"""
import pathlib

from tests._oracle_fresh import assert_oracle_fresh

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
GEN = REPO_ROOT / "devtools" / "wasm" / "gen_time_splits_oracle.py"
FIXTURE = REPO_ROOT / "wasm" / "smoke" / "oracle_time_splits.json"


def test_time_splits_oracle_is_fresh():
    assert_oracle_fresh(GEN, FIXTURE, REPO_ROOT)
