"""Freshness gate for wasm/smoke/oracle_combine_latest.json.

devtools/wasm/gen_combine_oracle.py records Python's combine_latest on a fixed
set of inputs; js/test/combine_latest_parity.test.ts checks that JavaScript's
combineLatest gives the same results. This test fails when the committed
fixture no longer matches the Python binding.
"""
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
GEN = REPO_ROOT / "devtools" / "wasm" / "gen_combine_oracle.py"
FIXTURE = REPO_ROOT / "wasm" / "smoke" / "oracle_combine_latest.json"


def test_combine_oracle_is_fresh():
    r = subprocess.run([sys.executable, str(GEN), "--stdout"], cwd=REPO_ROOT, capture_output=True)
    assert r.returncode == 0, r.stderr.decode(errors="replace")
    assert r.stdout == FIXTURE.read_bytes(), (
        "wasm/smoke/oracle_combine_latest.json is stale: re-run "
        "python devtools/wasm/gen_combine_oracle.py and commit the result."
    )
