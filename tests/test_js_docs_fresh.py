"""Freshness gate for the JavaScript package's built-in documentation.

devtools/wasm/gen_js_docs.py turns screamer/data (help.json, topics.json, and
the agent guide) into js/src/generated/docs_data.ts, and writes
js/test/docs_oracle.json, the Python search results that the JavaScript tests
compare against. Both are committed; this test fails when either is stale.
"""
import pathlib
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
GEN = REPO_ROOT / "devtools" / "wasm" / "gen_js_docs.py"

# The release wheel-test CI job runs against the installed wheel with the source
# tree's js/ directory still present but screamer/ removed; skip there.
pytestmark = pytest.mark.skipif(
    not (REPO_ROOT / "screamer" / "data" / "help.json").exists(),
    reason="needs the source tree's screamer/data")


def test_js_docs_are_fresh():
    r = subprocess.run([sys.executable, str(GEN), "--check"], cwd=REPO_ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0, (
        f"{r.stdout}{r.stderr}\nRe-run python {GEN.relative_to(REPO_ROOT)} and commit the result.")
