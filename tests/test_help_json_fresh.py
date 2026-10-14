"""Freshness gate for the built-in help registry, screamer/data/help.json.

devtools/build_help_registry.py builds help.json from the docs pages under
docs/functions_*/. The file is committed, and both bindings read it: Python for
help() and functor docstrings, and JavaScript through the generated factory
JSDoc. Nothing else checks that it was rebuilt after a docs page changed.

The generator's `--stdout` flag writes exactly what a plain run would write to
help.json, so this test can compare without touching the committed file.
"""
import pathlib
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
BUILD_HELP = REPO_ROOT / "devtools" / "build_help_registry.py"
HELP_JSON = REPO_ROOT / "screamer" / "data" / "help.json"

# The release wheel-test CI job runs the suite against the installed wheel after
# `rm -rf screamer`, which deletes help.json. This is a source-tree gate, so skip
# it there.
pytestmark = pytest.mark.skipif(
    not HELP_JSON.exists(),
    reason="needs the source tree's screamer/data/help.json (removed in the wheel-test env)",
)


def test_help_json_is_fresh():
    r = subprocess.run(
        [sys.executable, str(BUILD_HELP), "--stdout"],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    assert r.returncode == 0, (
        f"{BUILD_HELP.name} --stdout failed (exit {r.returncode}):\n"
        f"stderr:\n{r.stderr.decode(errors='replace')}"
    )
    assert r.stdout == HELP_JSON.read_bytes(), (
        f"{HELP_JSON.relative_to(REPO_ROOT)} is stale: rebuilding it from the docs "
        f"pages (python {BUILD_HELP.relative_to(REPO_ROOT)}) gives different "
        "content. Re-run the builder and commit the result."
    )
