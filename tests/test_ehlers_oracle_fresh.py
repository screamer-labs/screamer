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

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
GEN = REPO_ROOT / "devtools" / "wasm" / "gen_ehlers_oracle.py"
FIXTURE = REPO_ROOT / "wasm" / "smoke" / "oracle_ehlers.json"


def test_ehlers_oracle_is_fresh():
    r = subprocess.run([sys.executable, str(GEN), "--stdout"], cwd=REPO_ROOT, capture_output=True)
    assert r.returncode == 0, r.stderr.decode(errors="replace")
    assert r.stdout == FIXTURE.read_bytes(), (
        "wasm/smoke/oracle_ehlers.json is stale: re-run "
        "python devtools/wasm/gen_ehlers_oracle.py and commit the result."
    )
