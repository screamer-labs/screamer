"""Freshness gate for the "What you can build" page and the home page tiles.

devtools/build_examples.py builds docs/capabilities.md and
docs/examples_tiles.inc from docs/examples.yml. Its `--check` mode fails when a
recipe or notebook is not listed in examples.yml, a listed page, link, or
thumbnail does not exist, or a generated file differs from what the script
would write.
"""
import pathlib
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
BUILD_EXAMPLES = REPO_ROOT / "devtools" / "build_examples.py"
SPEC = REPO_ROOT / "docs" / "examples.yml"

# The release wheel-test CI job runs the suite against the installed wheel; skip
# there if the docs sources are absent.
pytestmark = pytest.mark.skipif(not SPEC.exists(), reason="needs the docs sources")


def test_examples_page_is_fresh():
    r = subprocess.run(
        [sys.executable, str(BUILD_EXAMPLES), "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, (
        f"{BUILD_EXAMPLES.relative_to(REPO_ROOT)} --check failed:\n{r.stdout}{r.stderr}\n"
        f"Update docs/examples.yml if needed, re-run python "
        f"{BUILD_EXAMPLES.relative_to(REPO_ROOT)}, and commit the result."
    )
