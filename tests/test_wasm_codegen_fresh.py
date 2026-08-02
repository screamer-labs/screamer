"""Codegen freshness gate for the WASM/Embind binding layer.

The WASM manifest (devtools/wasm/wasm_manifest.json) and the generated
Embind bindings (wasm/generated/bindings_wasm.cpp) are derived, by a pure
text-parsing generator, from the nanobind bindings/*.cpp sources. Both
generated files are committed (the WASM build doesn't run the generators
itself), so nothing enforces that they were regenerated after a Python op
was added, removed, or had its constructor signature changed -- except
this test.

Each generator supports a `--stdout` flag that writes exactly what a plain
run would write to its committed output path, but to stdout instead, so
this test can diff without ever touching the committed files (see
gen_wasm_manifest.py / gen_wasm_bindings.py for the flag).

This test only guards the generated *text*. It does not invoke emcc or
build the .wasm binary -- that's a dedicated CI job in a later phase.
"""
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
GEN_MANIFEST = REPO_ROOT / "devtools" / "wasm" / "gen_wasm_manifest.py"
GEN_BINDINGS = REPO_ROOT / "devtools" / "wasm" / "gen_wasm_bindings.py"
MANIFEST_PATH = REPO_ROOT / "devtools" / "wasm" / "wasm_manifest.json"
BINDINGS_PATH = REPO_ROOT / "wasm" / "generated" / "bindings_wasm.cpp"


def _run_stdout(script: pathlib.Path) -> bytes:
    r = subprocess.run(
        [sys.executable, str(script), "--stdout"],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    assert r.returncode == 0, (
        f"{script.name} --stdout failed (exit {r.returncode}):\n"
        f"stdout:\n{r.stdout.decode(errors='replace')}\n"
        f"stderr:\n{r.stderr.decode(errors='replace')}"
    )
    return r.stdout


def test_wasm_manifest_is_fresh():
    regenerated = _run_stdout(GEN_MANIFEST)
    committed = MANIFEST_PATH.read_bytes()
    assert regenerated == committed, (
        f"{MANIFEST_PATH.relative_to(REPO_ROOT)} is stale: regenerating it from "
        f"the current bindings/*.cpp sources (python3 {GEN_MANIFEST.relative_to(REPO_ROOT)}) "
        "produces different bytes than the committed file. Re-run the generator "
        "and commit the result."
    )


def test_wasm_bindings_are_fresh():
    regenerated = _run_stdout(GEN_BINDINGS)
    committed = BINDINGS_PATH.read_bytes()
    assert regenerated == committed, (
        f"{BINDINGS_PATH.relative_to(REPO_ROOT)} is stale: regenerating it from "
        f"the current manifest (python3 {GEN_BINDINGS.relative_to(REPO_ROOT)}) "
        "produces different bytes than the committed file. Re-run the generator "
        "(after regenerating the manifest, if that's also stale) and commit the result."
    )
