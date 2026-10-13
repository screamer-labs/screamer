"""Every devtools script that writes files prints its usage for --help and
writes nothing."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

SCRIPTS = [
    "devtools/build_function_index.py",
    "devtools/build_help_registry.py",
    "devtools/build_topic_pages.py",
    "devtools/generate_screamer__init__.py",
    "devtools/inject_nan_footnote.py",
    "devtools/stream_diagrams.py",
    "devtools/wasm/gen_combine_oracle.py",
    "devtools/wasm/gen_dag_oracle.py",
    "devtools/wasm/gen_null_oracle.py",
    "devtools/wasm/gen_oracle.py",
    "devtools/wasm/gen_shape_oracle.py",
    "devtools/wasm/gen_ts_api.py",
    "devtools/wasm/gen_wasm_bindings.py",
    "devtools/wasm/gen_wasm_manifest.py",
]

# Where the scripts write: every file under these is checked by mtime and size.
WATCHED = ["docs", "screamer", "wasm/smoke", "wasm/generated", "devtools", "js/src/generated"]
SKIP_PARTS = {"_build", "__pycache__", ".pytest_cache"}


def _snapshot():
    snap = {}
    for top in WATCHED:
        for dirpath, dirnames, filenames in os.walk(ROOT / top):
            dirnames[:] = [d for d in dirnames if d not in SKIP_PARTS]
            for name in filenames:
                p = Path(dirpath) / name
                st = p.stat()
                snap[p] = (st.st_mtime_ns, st.st_size)
    return snap


@pytest.mark.parametrize("script", SCRIPTS)
def test_help_prints_usage_and_writes_nothing(script):
    before = _snapshot()
    env = dict(os.environ, PYTHONPATH=str(ROOT), MPLBACKEND="Agg")
    r = subprocess.run([sys.executable, script, "--help"], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("usage:"), r.stdout[:200]
    after = _snapshot()
    changed = [str(p.relative_to(ROOT)) for p in after if before.get(p) != after[p]]
    changed += [str(p.relative_to(ROOT)) for p in before if p not in after]
    assert not changed, changed
