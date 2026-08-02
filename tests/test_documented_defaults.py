"""A documented default must be the default the constructor actually uses.

`ROC`, `ROCP` and `ROCR` documented `window_size: 10` (TA-Lib's timeperiod, and
what the sibling `Momentum` declares) while their bindings defaulted to 1. So
`ROC()` computed a 1-period rate of change, and the page said otherwise.

Nothing caught it. The docs build validates a page by instantiating the functor,
which succeeds whatever the default is, and every compliance suite instantiates
from the *documented* defaults in `help.json`, so they all exercised
`window_size=10` and never touched the value a user gets from `ROC()`. It
surfaced only when a baseline was written from the documented definition and
compared against the operator driven on its own defaults.

This reads the default out of the pybind11 signature and compares it to the
frontmatter.
"""
from __future__ import annotations

import ast
import json
import re
import importlib.resources

import pytest

import screamer

# Load help.json from the installed screamer package so this runs against the wheel.
HELP_JSON = importlib.resources.files("screamer").joinpath("data/help.json")
HELP: dict[str, dict] = json.loads(HELP_JSON.read_text())

# A parameter that is part of a mutually exclusive group defaults to None in
# the binding, because exactly one member of the group must be supplied and the
# constructor raises otherwise. The exponentially weighted family
# (com / span / halflife / alpha) and the Ehlers filters (period / cutoff,
# hp_period / hp_cutoff) both work this way. Their pages document a
# representative value, `span: 20` or `period: 60`, which is the value used in
# the examples rather than a constructor default.
#
# Rather than enumerate the groups, treat a binding default of None as the
# marker. That is exactly the "no default, one of the group required" case, and
# it does not exempt a real mismatch: ROC documented 10 against a binding
# default of 1, and neither is None.
def _is_mutex_member(binding_default: str) -> bool:
    return binding_default.strip() == "None"


def _binding_defaults(name: str) -> dict[str, str]:
    """Parameter -> default, parsed out of the pybind11 __init__ signature."""
    cls = getattr(screamer, name, None)
    doc = getattr(cls, "__init__", None) and cls.__init__.__doc__
    if not doc:
        return {}
    signature = doc.splitlines()[0]
    inner = signature[signature.index("(") + 1: signature.rindex(")")]

    defaults: dict[str, str] = {}
    depth = 0
    current = ""
    parts = []
    for ch in inner:                       # split on commas outside brackets
        if ch in "[({":
            depth += 1
        elif ch in "])}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += ch
    parts.append(current)

    for part in parts:
        if "=" not in part:
            continue
        head, _, value = part.partition("=")
        param = head.split(":")[0].strip()
        defaults[param] = value.strip()
    return defaults


def _equivalent(documented, binding: str) -> bool:
    try:
        parsed = ast.literal_eval(binding)
    except (ValueError, SyntaxError):
        return str(documented) == binding.strip("'\"")
    if isinstance(documented, (int, float)) and isinstance(parsed, (int, float)):
        return float(documented) == float(parsed)
    return documented == parsed


CASES = [
    pytest.param(name, entry, id=name)
    for name, entry in sorted(HELP.items())
    if entry.get("kind", "functor") == "functor" and entry.get("parameters")
]


@pytest.mark.parametrize("name,entry", CASES)
def test_documented_default_matches_the_binding(name: str, entry: dict):
    actual = _binding_defaults(name)
    if not actual:
        pytest.skip(f"{name}: no introspectable pybind11 signature")

    for param in entry["parameters"]:
        pname = param["name"]
        if "default" not in param or pname not in actual:
            continue
        if _is_mutex_member(actual[pname]):
            continue
        documented = param["default"]
        if documented is None:
            continue
        assert _equivalent(documented, actual[pname]), (
            f"{name}.{pname}: the page documents {documented!r} but the binding "
            f"defaults to {actual[pname]}. A user calling {name}() gets the "
            "binding's value, so the page is wrong or the binding is."
        )
