"""A reference's default arguments must match the operator's documented ones.

The harness drives some operators with no explicit parameters, taking whichever
defaults each side declares. If a reference defaults to a different window than
the operator, the comparison is between two different calculations: it fails
confusingly, or worse, passes because both happen to be degenerate.

That cost real time here. `KAMA` was compared at window 30 against an operator
defaulting to 10, `CCI` at 20 against 14, and `TRIX` at span 15 against 14. All
three looked like operator bugs and were bugs in the reference.

`tests/test_documented_defaults.py` is the same check on the other side: the
documented default against the actual pybind11 signature. Together they pin
operator, page, and reference to one value.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from devtools import baselines, get_baselines

HELP_JSON = Path(__file__).resolve().parent.parent / "screamer" / "data" / "help.json"
HELP: dict[str, dict] = json.loads(HELP_JSON.read_text())

# The exponentially weighted mutex: exactly one of these must be supplied, so
# every one of them defaults to None on both sides. See test_documented_defaults.
EW_MUTEX = {"com", "span", "halflife", "alpha"}

# Stale references, from before the operators moved their `output`/`quantile`
# arguments. Pre-existing and unrelated to the change that added this test;
# listed so the gap is visible and can only shrink.
KNOWN_STALE: set[tuple[str, str]] = {
    ("RollingOU_numpy", "output"),
    ("RollingOU_numpy_v2", "output"),
    ("RollingQuantile_numpy", "quantile"),
    ("RollingQuantile_pandas", "quantile"),
    ("RollingSigmaClip_python", "output"),
}


def _equivalent(a, b) -> bool:
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return list(a) == list(b)
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)


CASES = [
    pytest.param(name, entry, id=name)
    for name, entry in sorted(HELP.items())
    if entry.get("kind", "functor") == "functor"
    and entry.get("parameters")
    and get_baselines(name)
]


@pytest.mark.parametrize("name,entry", CASES)
def test_baseline_defaults_match_the_documented_defaults(name: str, entry: dict):
    documented = {p["name"]: p.get("default") for p in entry["parameters"]}

    for baseline_name in get_baselines(name):
        cls = getattr(baselines, baseline_name)
        try:
            signature = inspect.signature(cls.__init__)
        except (TypeError, ValueError):
            continue

        for param_name, param in signature.parameters.items():
            if param_name == "self" or param.default is inspect.Parameter.empty:
                continue
            if param_name in EW_MUTEX:
                continue
            if (baseline_name, param_name) in KNOWN_STALE:
                continue
            expected = documented.get(param_name)
            if expected is None:
                continue
            assert _equivalent(param.default, expected), (
                f"{baseline_name}.{param_name} defaults to {param.default!r} but "
                f"{name} documents {expected!r}. Driven with no explicit "
                "parameters, the two sides would compute different things."
            )
