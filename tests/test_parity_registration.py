"""Every public functor must be reachable by the parity harness.

`tests/param_cases.py` drives the array-surface tests: stream vs batch, tensor,
view, matrix, io_size, and the baseline comparisons. It used to require a
hand-written entry for anything outside the `Rolling*` / `Ew*` / no-argument
groups, and nothing noticed when one was missing. 122 of 213 functors had no
entry at all, so they had no coverage of that surface and no way to find out.

`FillNa` was one of them. Its strided array path tested the *output* buffer for
NaN instead of the input, so on any strided input it filled random elements
with the fill value and let real NaNs through. The bug was reachable from the
public API and survived because no test in this harness ever ran it.

This file closes the loop: a functor must be driven by the harness, or named
in `PARITY_EXEMPT` with a reason. Adding an operator without either fails here.
"""
from __future__ import annotations

import json
import importlib.resources

from .param_cases import (
    PARITY_EXEMPT,
    SHAPES,
    expanded_test_cases,
    screamer_classes,
    test_definitions,
)

# Load help.json from the installed screamer package so this runs against the wheel.
HELP_JSON = importlib.resources.files("screamer").joinpath("data/help.json")


def _covered() -> set[str]:
    names: set[str] = set()
    for class_names, _ in expanded_test_cases(test_definitions):
        names.update(class_names)
    return names


def test_every_functor_is_driven_or_exempt():
    """No functor may be silently absent from the parity harness."""
    functors = {
        name
        for name, entry in json.loads(HELP_JSON.read_text()).items()
        if entry.get("kind", "functor") == "functor" and name in screamer_classes
    }
    unaccounted = sorted(functors - _covered() - set(PARITY_EXEMPT))
    assert not unaccounted, (
        f"functors with no parity coverage and no exemption: {unaccounted}. "
        "Add a tests/param_cases.py entry (a 1-in/1-out operator whose "
        "constructor is fully defaulted is adopted automatically), or add the "
        "name to PARITY_EXEMPT with the reason it cannot be driven."
    )


def test_exemptions_carry_a_reason():
    """An exemption without a reason is an unexplained coverage hole."""
    blank = sorted(name for name, reason in PARITY_EXEMPT.items() if not reason.strip())
    assert not blank, f"PARITY_EXEMPT entries with no reason: {blank}"


def test_exemptions_are_live():
    """An exemption naming a function that no longer exists is stale."""
    stale = sorted(set(PARITY_EXEMPT) - set(SHAPES))
    assert not stale, (
        f"PARITY_EXEMPT names functions that are not public functors: {stale}. "
        "Remove them."
    )


def test_exempt_functors_are_not_also_driven():
    """A name in both places means the exemption's reason is out of date."""
    both = sorted(set(PARITY_EXEMPT) & _covered())
    assert not both, (
        f"named in PARITY_EXEMPT and also driven by the harness: {both}. "
        "Drop the exemption."
    )
