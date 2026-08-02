"""Parity gate: the JS resample/agg code tables (js/src/codes.ts) must match the
Python source of truth (screamer/dag.py). Either side drifting fails this test.

The JS tables are read by running `node --import tsx src/codes.ts --dump`, which
prints the aggregated `CODES` object as JSON. The Python tables are imported
directly from screamer.dag.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

import screamer.dag as dag

_REPO = Path(__file__).resolve().parent.parent
_JS = _REPO / "js"
_CODES_TS = _JS / "src" / "codes.ts"


def _dump_js_codes():
    """Return the JS CODES object as a dict, or skip if node/tsx is unavailable."""
    if shutil.which("node") is None:
        pytest.skip("node not available; cannot dump js/src/codes.ts")
    if not _CODES_TS.exists():
        pytest.skip(f"missing {_CODES_TS}")
    try:
        out = subprocess.run(
            ["node", "--import", "tsx", str(_CODES_TS), "--dump"],
            cwd=str(_JS),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError:
        pytest.skip("node not runnable")
    if out.returncode != 0:
        pytest.skip(f"node dump failed (tsx not installed?):\n{out.stderr}")
    return json.loads(out.stdout)


# Fixed plans in dag.py store (agg, col) tuples; JSON round-trips them to lists.
def _plans_as_lists(plans):
    return {k: [list(pair) for pair in v] for k, v in plans.items()}


def test_js_codes_match_python():
    js = _dump_js_codes()

    assert js["RESAMPLE_AGG_CODE"] == dag._RESAMPLE_AGG_CODE
    assert js["RESAMPLE_MODE_CODE"] == dag._RESAMPLE_MODE_CODE
    assert js["RESAMPLE_FILL_CODE"] == dag._RESAMPLE_FILL_CODE
    assert js["PLAN_AGG_CODE"] == dag._PLAN_AGG
    assert js["BAR_AGG_FIXED_PLANS"] == _plans_as_lists(dag._BAR_AGG_FIXED_PLANS)

    # dag.py derives the label code inline (`1 if label == "right" else 0`); the
    # JS side spells it out as a table. Assert the derived mapping matches.
    assert js["RESAMPLE_LABEL_CODE"] == {"left": 0, "right": 1}
