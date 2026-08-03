"""Contract properties asserted against every operator in the registry.

These are the rules `CONTRIBUTING.md` states that every operator must satisfy,
checked by enumerating `screamer/data/help.json` rather than by naming
functions. A new operator is enrolled the day its docs page lands, which is
mandatory, so a contract cannot be missed by forgetting to write a test.

The properties here are the ones that apply to every operator regardless of
what it computes:

* :func:`test_reset_restores_initial_state` - `reset()` returns the operator
  to construction state, so a second pass over the same input reproduces the
  first exactly.
* :func:`test_causal` - output at index t depends only on input up to t.
  Changing the future must not change the past.
* :func:`test_regimes_agree` - eager (arrays), graph (`Pipeline`), and lazy
  (event iterator) produce identical output. `CONTRIBUTING.md` calls this the
  definition of done for an operator.

Per-operator numerical behaviour (does `ATR` match TA-Lib) belongs in that
operator's own test file. Anything stated as a rule for all operators belongs
here, driven by the registry.

Functions that cannot satisfy a property for a stated structural reason are
listed in the `KNOWN_*` set beside it and marked `xfail(strict=True)`, so a
fix fails the suite until the set is updated.
"""
from __future__ import annotations

import importlib.resources
import json

import numpy as np
import pytest

import screamer

# Load help.json from the installed screamer package (importlib.resources), not
# a source-relative path, so this suite runs against the built wheel too.
HELP_JSON = importlib.resources.files("screamer").joinpath("data/help.json")
HELP: dict[str, dict] = json.loads(HELP_JSON.read_text())

# Only compute functors carry these contracts. Stream operators and DAG nodes
# (kind != "functor") have their own semantics and their own test files.
# PortfolioReport consumes a dynamic-width (time, assets, 4) engine tensor;
# its reset and causal behavior are tested in tests/test_backtest.py rather than
# the fixed-width scalar/DAG harness below.
DYNAMIC_WIDTH_FUNCTORS = {"PortfolioReport"}
FUNCTORS = {
    name: entry
    for name, entry in HELP.items()
    if entry.get("kind", "functor") == "functor"
    and hasattr(screamer, name)
    and name not in DYNAMIC_WIDTH_FUNCTORS
}

# Operators whose output is not a pure function of the input, so running them
# twice legitimately differs. None today; kept so the reason is recorded if one
# ever appears.
KNOWN_NOT_RESETTABLE: set[str] = set()

KNOWN_NOT_CAUSAL: set[str] = set()

# Multi-output operators return a tuple or a 2-D array from the eager call but
# one node per output in a Pipeline, so driving them generically needs a
# per-operator adapter. Their regime equality is asserted in their own files.
KNOWN_NO_GENERIC_REGIME: set[str] = {
    "PortfolioReport",  # dynamic-width batch/stream reducer; bespoke tests cover it
}


def _instantiate(entry: dict):
    kwargs = {p["name"]: p["default"] for p in (entry.get("parameters") or [])}
    return getattr(screamer, entry["name"])(**kwargs)


def _inputs(n_inputs: int, n_samples: int = 200, seed: int = 7) -> list[np.ndarray]:
    """Bounded-positive input, matching the other compliance files: it keeps
    every operator's domain happy (Log and Sqrt want positive, Asin wants
    |x| <= 1)."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.1, 0.9, size=n_samples).astype(np.float64)
    return [base.copy() for _ in range(n_inputs)]


def _call(instance, arrays: list[np.ndarray]):
    return instance(arrays[0]) if len(arrays) == 1 else instance(*arrays)


def _as_arrays(out) -> list[np.ndarray]:
    """Flatten whatever an operator returns into a list of 1-D streams."""
    if isinstance(out, tuple):
        return [a for o in out for a in _as_arrays(o)]
    out = np.asarray(out)
    if out.ndim == 2:
        return [out[:, i] for i in range(out.shape[1])]
    return [out]


def _xfail_if_in(known: set[str], label: str):
    def factory(name: str):
        if name in known:
            return pytest.mark.xfail(
                strict=True,
                reason=f"{name}: {label}. Remove from the KNOWN_* set when fixed.",
            )
        return ()
    return factory


def _params(known: set[str], label: str):
    mark = _xfail_if_in(known, label)
    return [
        pytest.param(name, entry, id=name, marks=mark(name))
        for name, entry in sorted(FUNCTORS.items())
    ]


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name,entry", _params(KNOWN_NOT_RESETTABLE, "reset() does not restore initial state")
)
def test_reset_restores_initial_state(name: str, entry: dict):
    """A reset operator must behave exactly like a fresh one."""
    arrays = _inputs(int(entry.get("inputs", 1)))
    op = _instantiate(entry)

    first = _as_arrays(_call(op, arrays))
    op.reset()
    second = _as_arrays(_call(op, arrays))

    for a, b in zip(first, second):
        np.testing.assert_array_equal(
            a, b,
            err_msg=(
                f"{name}: output after reset() differs from the first pass, so "
                "state survived the reset."
            ),
        )


# ---------------------------------------------------------------------------
# Causality
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name,entry", _params(KNOWN_NOT_CAUSAL, "output depends on future input")
)
def test_causal(name: str, entry: dict):
    """Output up to index t must not change when input after t changes.

    This is the rule the library will not bend: no lookahead, no backfill. It
    is asserted here for every operator rather than trusted per author.
    """
    n_inputs = int(entry.get("inputs", 1))
    split = 120

    base = _inputs(n_inputs)
    tampered = [a.copy() for a in base]
    for a in tampered:
        # A different, still in-domain future.
        a[split:] = _inputs(n_inputs, seed=99)[0][split:]

    before = _as_arrays(_call(_instantiate(entry), base))
    after = _as_arrays(_call(_instantiate(entry), tampered))

    for a, b in zip(before, after):
        np.testing.assert_array_equal(
            a[:split], b[:split],
            err_msg=(
                f"{name}: changing input at index >= {split} changed output "
                f"before {split}. The operator reads the future."
            ),
        )


# ---------------------------------------------------------------------------
# Every regime
# ---------------------------------------------------------------------------

# The microstructure operators are Python compositions rather than C++
# functors. Their `__call__` takes one argument per input instead of a single
# combined frame, so the generic Pipeline driver below cannot wire them, and
# `tests/param_cases.py` excludes them from its harness for the same reason.
# Their regime equality is asserted in tests/test_microstructure.py. They are
# still covered by the reset() and causality properties above.
import screamer.microstructure as _micro

PYTHON_OPERATORS: set[str] = set(_micro.__all__)

_REGIME_PARAMS = [
    pytest.param(name, entry, id=name, marks=_xfail_if_in(
        KNOWN_NO_GENERIC_REGIME, "regimes disagree")(name))
    for name, entry in sorted(FUNCTORS.items())
    if int(entry.get("outputs", 1)) == 1 and name not in PYTHON_OPERATORS
]


@pytest.mark.parametrize("name,entry", _REGIME_PARAMS)
def test_regimes_agree(name: str, entry: dict):
    """Eager, graph, and lazy must produce identical output.

    `CONTRIBUTING.md`: "The operator is not done until a test proves it runs,
    with identical output, in all three regimes."
    """
    n_inputs = int(entry.get("inputs", 1))
    arrays = _inputs(n_inputs, n_samples=120)
    index = np.arange(len(arrays[0]))

    eager = _as_arrays(_call(_instantiate(entry), arrays))[0]

    nodes = [screamer.Input(f"in{i}") for i in range(n_inputs)]
    # A multi-input operator consumes one frame of n values, so the separate
    # input streams have to be aligned into that frame first. This is the
    # documented way to wire a multi-input operator into a Pipeline, not a
    # test-only workaround.
    source = nodes[0] if n_inputs == 1 else screamer.CombineLatest()(*nodes)
    pipeline = screamer.Pipeline(
        inputs=nodes, outputs=[_instantiate(entry)(source)]
    )

    graph_values, _ = pipeline(*[(a, index) for a in arrays])
    np.testing.assert_allclose(
        graph_values, eager, equal_nan=True, rtol=1e-12,
        err_msg=f"{name}: graph regime disagrees with eager",
    )

    lazy_rows = list(pipeline(*[
        ((float(v), int(i)) for i, v in enumerate(a)) for a in arrays
    ]))
    np.testing.assert_allclose(
        np.array([r[0] for r in lazy_rows]), eager, equal_nan=True, rtol=1e-12,
        err_msg=f"{name}: lazy regime disagrees with eager",
    )
