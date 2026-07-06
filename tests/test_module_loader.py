"""Regression guard for the local-screamer module loader.

``ScreamerInstallInfo._load_module_from_file_path`` temporarily adjusts
``sys.path`` so the local ``screamer`` package resolves first. It used to
*replace* ``sys.path`` with only the project root, which removed the stdlib
path for the duration of ``exec_module``. Any module imported during init that
was not already cached in ``sys.modules`` -- e.g. ``import asyncio`` in
streams.py -- then failed with ``ModuleNotFoundError`` in a minimal environment
(this broke a release; see .github/workflows/test.yml). The loader must keep
the stdlib importable.
"""
import sys

from devtools import ScreamerInstallInfo


def test_loader_keeps_stdlib_importable(tmp_path):
    # A module that imports a rarely-pre-imported stdlib module at import time.
    mod_file = tmp_path / "needs_stdlib.py"
    mod_file.write_text("import colorsys\nvalue = colorsys.rgb_to_hls(0.0, 0.0, 0.0)\n")

    # Evict it so the import must resolve through sys.path during exec_module,
    # exactly as an uncached stdlib import would in a fresh CI environment.
    sys.modules.pop("colorsys", None)

    loaded = ScreamerInstallInfo()._load_module_from_file_path(
        "needs_stdlib", str(mod_file)
    )
    assert loaded.value == (0.0, 0.0, 0.0)


def test_loader_restores_sys_path_on_error(tmp_path):
    # A module that raises during import must not leave sys.path clobbered.
    mod_file = tmp_path / "boom.py"
    mod_file.write_text("raise RuntimeError('boom')\n")

    before = sys.path.copy()
    try:
        ScreamerInstallInfo()._load_module_from_file_path("boom", str(mod_file))
    except RuntimeError:
        pass
    assert sys.path == before
