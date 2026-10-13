"""Shared freshness check for the committed JavaScript oracles.

Each oracle is the output of a generator script run on the Python binding. The
generator runs in a child process, so it gets the test's import path. The
numbers are compared to 1e-9 (relative, or absolute near zero), as the
JavaScript tests compare them: GCC builds do not fuse multiply-adds and clang
builds do (D4), so a value can differ in its last digits between a Linux and a
macOS build while the oracle is still current.
"""
import json
import math
import os
import subprocess
import sys


def _same(a, b, path="$"):
    if isinstance(a, float) or isinstance(b, float):
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or \
                isinstance(a, bool) or isinstance(b, bool):
            return f"{path}: {a!r} != {b!r}"
        if math.isnan(a) and math.isnan(b):
            return None
        return None if math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9) else f"{path}: {a!r} != {b!r}"
    if type(a) is not type(b):
        return f"{path}: {a!r} != {b!r}"
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return f"{path}: keys {sorted(a)} != {sorted(b)}"
        for k in a:
            d = _same(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = _same(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    return None if a == b else f"{path}: {a!r} != {b!r}"


def assert_oracle_fresh(gen, fixture, root):
    """Run `gen --stdout` and compare its JSON with the committed fixture."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(root)] + [p for p in sys.path if p])
    r = subprocess.run([sys.executable, str(gen), "--stdout"], cwd=root, env=env, capture_output=True)
    assert r.returncode == 0, r.stderr.decode(errors="replace")
    diff = _same(json.loads(r.stdout), json.loads(fixture.read_bytes()))
    rel = fixture.relative_to(root)
    assert diff is None, (f"{rel} is stale ({diff}): re-run python "
                          f"{gen.relative_to(root)} and commit the result.")
