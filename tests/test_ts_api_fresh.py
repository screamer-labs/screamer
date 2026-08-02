"""Codegen freshness gate for the ergonomic TypeScript factory layer.

devtools/wasm/gen_ts_api.py reads devtools/wasm/wasm_manifest.json plus the
installed screamer package's docstrings and emits js/src/generated/ops.ts
(226 typed factories) and js/src/generated/ops.d.ts. Both generated files
are committed (the JS build doesn't run the generator itself), so nothing
enforces that they were regenerated after a Python op's constructor
signature changed, or a new op was added to the manifest -- except this
test.

The generator supports `--stdout` (ops.ts) and `--stdout-dts` (ops.d.ts),
each writing exactly what a plain run would write to its committed output
path, but to stdout instead, so this test can diff without ever touching
the committed files.

This test only guards the generated *text*. It does not compile the
TypeScript or run tsc -- that's covered by `cd js && npm run build`
(js/package.json), exercised separately from the Python suite.
"""
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
GEN_TS_API = REPO_ROOT / "devtools" / "wasm" / "gen_ts_api.py"
OPS_TS = REPO_ROOT / "js" / "src" / "generated" / "ops.ts"
OPS_DTS = REPO_ROOT / "js" / "src" / "generated" / "ops.d.ts"


def _run_stdout(flag: str) -> bytes:
    r = subprocess.run(
        [sys.executable, str(GEN_TS_API), flag],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    assert r.returncode == 0, (
        f"{GEN_TS_API.name} {flag} failed (exit {r.returncode}):\n"
        f"stdout:\n{r.stdout.decode(errors='replace')}\n"
        f"stderr:\n{r.stderr.decode(errors='replace')}"
    )
    return r.stdout


def test_ops_ts_is_fresh():
    regenerated = _run_stdout("--stdout")
    committed = OPS_TS.read_bytes()
    assert regenerated == committed, (
        f"{OPS_TS.relative_to(REPO_ROOT)} is stale: regenerating it "
        f"(python3 {GEN_TS_API.relative_to(REPO_ROOT)}) produces different "
        "bytes than the committed file. Re-run the generator and commit "
        "the result."
    )


def test_ops_dts_is_fresh():
    regenerated = _run_stdout("--stdout-dts")
    committed = OPS_DTS.read_bytes()
    assert regenerated == committed, (
        f"{OPS_DTS.relative_to(REPO_ROOT)} is stale: regenerating it "
        f"(python3 {GEN_TS_API.relative_to(REPO_ROOT)}) produces different "
        "bytes than the committed file. Re-run the generator and commit "
        "the result."
    )
