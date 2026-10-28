"""Freshness gate for wasm/smoke/oracle_book_report.json.

devtools/wasm/gen_book_report_oracle.py records Python's backtest_report on
portfolio engine output; js/test/backtest_book.test.ts checks that JavaScript
gives the same values. This test fails when the committed fixture no longer
matches the Python binding.
"""
import pathlib

from tests._oracle_fresh import assert_oracle_fresh

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
GEN = REPO_ROOT / "devtools" / "wasm" / "gen_book_report_oracle.py"
FIXTURE = REPO_ROOT / "wasm" / "smoke" / "oracle_book_report.json"


def test_book_report_oracle_is_fresh():
    assert_oracle_fresh(GEN, FIXTURE, REPO_ROOT)
