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
import json
import pathlib
import re
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
GEN_MANIFEST = REPO_ROOT / "devtools" / "wasm" / "gen_wasm_manifest.py"
GEN_BINDINGS = REPO_ROOT / "devtools" / "wasm" / "gen_wasm_bindings.py"
MANIFEST_PATH = REPO_ROOT / "devtools" / "wasm" / "wasm_manifest.json"
BINDINGS_PATH = REPO_ROOT / "wasm" / "generated" / "bindings_wasm.cpp"
BINDINGS_DIR = REPO_ROOT / "bindings"

# bindings_streams.cpp and bindings_dag.cpp register the Stream/Pipeline
# combinator machinery, not point ops, and are out of scope for the WASM
# manifest.
NON_POINT_OP_SOURCES = {"bindings_streams.cpp", "bindings_dag.cpp"}

# Infrastructure classes that get an `nb::class_<...>(m, "NAME")`
# registration but aren't screamer ops, so they never appear in the
# manifest.
NON_OP_CLASS_NAMES = {
    "EvalOp",
    "ScreamerBase",
    "LazyEvalIterator",
    "LazyAsyncIterator",
    "AnextAwaitable",
}

# Matches `nb::class_<...anything, possibly with nested <>...>(m, "NAME")`.
# The `.*?` is non-greedy so it stops at the first `>` immediately followed
# by `(m, "..."`, i.e. the `>` that closes the nb::class_<...> template
# argument list itself, even when that list contains its own nested
# template brackets (e.g. `screamer::Transform<...>` base types).
CLASS_REGISTRATION_RE = re.compile(
    r'nb::class_<.*?>\s*\(\s*m\s*,\s*"([A-Za-z0-9_]+)"'
)


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


def test_wasm_manifest_matches_independent_binding_scan():
    """Cross-check the manifest against a scan that doesn't reuse the generator.

    gen_wasm_manifest.py classifies ops by base class and constructor shape.
    A future op registered with an unusual base or a factory-style
    constructor could be silently dropped by that classification -- and
    neither test_wasm_manifest_is_fresh (which compares the generator's
    output to itself) nor a consumer that iterates the manifest would
    notice, since a dropped op is just... absent, not wrong.

    This test enumerates every `nb::class_<...>(m, "NAME")` registration
    across bindings/*.cpp directly with a regex, independent of the
    generator's logic, and asserts that set is exactly the set of ops in
    the committed manifest. A name present in the bindings scan but
    missing from the manifest means the generator silently dropped it.
    """
    scanned_names = set()
    for source in sorted(BINDINGS_DIR.glob("*.cpp")):
        if source.name in NON_POINT_OP_SOURCES:
            continue
        text = source.read_text()
        for match in CLASS_REGISTRATION_RE.finditer(text):
            scanned_names.add(match.group(1))

    scanned_op_names = scanned_names - NON_OP_CLASS_NAMES

    manifest = json.loads(MANIFEST_PATH.read_text())
    manifest_names = {entry["name"] for entry in manifest}

    missing_from_manifest = scanned_op_names - manifest_names
    stale_in_manifest = manifest_names - scanned_op_names

    assert scanned_op_names == manifest_names, (
        "Independent scan of nb::class_<...>(m, \"NAME\") registrations in "
        f"bindings/*.cpp (excluding {sorted(NON_POINT_OP_SOURCES)}) does not "
        f"match {MANIFEST_PATH.relative_to(REPO_ROOT)}.\n"
        f"In bindings but not manifest (silently dropped by the generator): "
        f"{sorted(missing_from_manifest)}\n"
        f"In manifest but not bindings (stale entries): "
        f"{sorted(stale_in_manifest)}"
    )
