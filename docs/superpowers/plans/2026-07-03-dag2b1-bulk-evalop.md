# DAG-2b-1 — bulk EvalOp registration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Register every remaining functor under `EvalOp` in pybind (using each class's correct C++ base) so the DAG engine can recover an `EvalOp*` from *any* functor object generically, and every functor exposes `num_inputs`/`num_outputs`.

**Architecture:** DAG-2a made both `ScreamerBase` and `FunctorBase` derive from `EvalOp` at the C++ level and registered 3 representative functors. This increment finishes the job with a precise, pre-classified list: `FunctorBase`-derived functors are bound `py::class_<X, EvalOp>`; `ScreamerBase`-derived functors that were bound *bare* are bound `py::class_<X, ScreamerBase>` (which inherits `EvalOp` transitively and keeps `process_scalar`). No new abstractions — a targeted, mechanical completion.

**Tech Stack:** C++17, pybind11, pytest.

## Global Constraints

- **Correct base per class** (pre-classified below): `FunctorBase` → `, EvalOp`; `ScreamerBase`-bound-bare → `, ScreamerBase`. Registering a `ScreamerBase` functor under `EvalOp` would drop its `process_scalar` — do NOT do that.
- **Already done in DAG-2a (do not touch):** `RollingCorr`, `BollingerBands`, `Cart2Polar` are already `, EvalOp`.
- **Skip non-functors:** `EvalOp`, `ScreamerBase`, `LazyIterator`, `LazyAsyncIterator`, `AnextAwaitable`, and the streams pull-objects are not functors.
- **Hot path untouched; math never modified.** This is binding-only + tests.
- Never hand-edit `screamer/__init__.py` or version files.
- Build: `make build` **then `make install-dev`**. Tests: `poetry run pytest tests/test_evalop_registry.py -v` then the full suite.

---

## File Structure

- `bindings/*.cpp` (modify) — add the correct base to each bare functor `py::class_` (list below); add `#include "screamer/common/eval_op.h"` to each file touched.
- `tests/test_evalop_registry.py` (create) — assert every exported functor exposes `num_inputs`/`num_outputs` with the right values.

---

### Task 1: register every functor under its correct base

**Files:**
- Modify: `bindings/bindings_math.cpp`, `bindings/bindings_rolling.cpp`, `bindings/bindings_ew.cpp`, `bindings/bindings_fin.cpp`, `bindings/bindings_preprocessing.cpp`, `bindings/bindings_misc.cpp`, `bindings/bindings_myfunctors.cpp` (only the files that bind the classes listed below)
- Test: `tests/test_evalop_registry.py`

**Interfaces:**
- Produces: every functor bound under `EvalOp` (directly, or via `ScreamerBase`), so `py::cast<EvalOp*>(functor_obj)` succeeds for all and `.num_inputs`/`.num_outputs` are available library-wide.

- [ ] **Step 1: Write the failing test**

Create `tests/test_evalop_registry.py`:

```python
"""Every functor must be registered under EvalOp (exposes num_inputs/num_outputs)."""
import pytest
import screamer

# (constructor-args, expected num_inputs, expected num_outputs) per functor.
# Covers the arity spread and both base categories.
CASES = {
    # FunctorBase (now under EvalOp)
    "ATR": ((14,), 3, 1),
    "AD": ((), 4, 1),
    "ADOSC": ((3, 10), 4, 1),
    "BOP": ((), 4, 1),
    "MFI": ((14,), 4, 1),
    "Stoch": ((14,), 3, 2),
    "MACD": ((), 1, 3),
    "KeltnerChannels": ((20,), 3, 3),
    "DonchianChannels": ((20,), 2, 3),
    "RollingMinMax": ((10,), 1, 2),
    "RollingLinearRegression": ((10,), 2, 4),
    "Hypot": ((), 2, 1),
    "RollingSpread": ((20,), 2, 1),
    "WilliamsR": ((14,), 3, 1),
    "StochRSI": ((14,), 1, 2),
    # ScreamerBase-bound-bare (now under ScreamerBase -> inherits EvalOp)
    "RollingPoly1": ((10,), 1, 1),
    "RollingPoly2": ((10,), 1, 1),
    "RollingSigmaClip": ((10,), 1, 1),
    "RollingOU": ((10,), 1, 1),
}


@pytest.mark.parametrize("name,args,n_in,n_out", [(k, *v) for k, v in CASES.items()])
def test_functor_arity_registered(name, args, n_in, n_out):
    cls = getattr(screamer, name)
    obj = cls(*args)
    assert obj.num_inputs == n_in, f"{name}.num_inputs"
    assert obj.num_outputs == n_out, f"{name}.num_outputs"


def test_screamerbase_bare_keeps_process_scalar():
    # RollingPoly2 was bound bare; after `, ScreamerBase` it must still expose
    # process_scalar (would be lost if wrongly bound under EvalOp).
    f = screamer.RollingPoly2(10)
    assert hasattr(f, "process_scalar")
    assert isinstance(f.process_scalar(1.0), float)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_evalop_registry.py -v`
Expected: FAIL — e.g. `ATR` has no `num_inputs` (not yet registered under `EvalOp`).

- [ ] **Step 3: Register the FunctorBase functors under `EvalOp`**

For each class in this list, change its binding from `py::class_<screamer::X>(m, "X")` to `py::class_<screamer::X, screamer::EvalOp>(m, "X")`. Add `#include "screamer/common/eval_op.h"` to any binding file you touch that lacks it.

```
AD, ADOSC, ADX, ATR, Atan2, BOP, CCI, DonchianChannels,
EwBeta, EwCorr, EwCov, EwGarmanKlassVar, EwGarmanKlassVol,
EwParkinsonVar, EwParkinsonVol, EwRogersSatchellVar, EwRogersSatchellVol,
Hypot, KeltnerChannels, Linear2, MACD, MFI, NATR, OBV, Polar2Cart,
RollingAlpha, RollingBeta, RollingCov, RollingGarmanKlassVar, RollingGarmanKlassVol,
RollingInfoRatio, RollingLinearRegression, RollingMinMax,
RollingParkinsonVar, RollingParkinsonVol, RollingResidualStd,
RollingRogersSatchellVar, RollingRogersSatchellVol, RollingSpread, RollingVWAP,
RollingYangZhangVar, RollingYangZhangVol, Stoch, StochRSI, TrueRange,
UltimateOscillator, WilliamsR
```

(Do NOT touch `RollingCorr`, `BollingerBands`, `Cart2Polar` — already `, EvalOp` from DAG-2a. `MyFunctor11`/`MyFunctor22` in `bindings_myfunctors.cpp` are `FunctorBase` demos — register them under `EvalOp` too for consistency.)

- [ ] **Step 4: Register the bare ScreamerBase functors under `ScreamerBase`**

For each of these (all `: public ScreamerBase` but bound bare), change `py::class_<screamer::X>(m, "X")` to `py::class_<screamer::X, screamer::ScreamerBase>(m, "X")`:

```
Clip, RollingOU, RollingPoly1, RollingPoly2, RollingSigmaClip
```

These need no `eval_op.h` include (they get `EvalOp` transitively via `ScreamerBase`).

- [ ] **Step 5: Build and run the registry test**

Run: `make build && make install-dev && poetry run pytest tests/test_evalop_registry.py -v`
Expected: all parametrized cases PASS; `test_screamerbase_bare_keeps_process_scalar` PASS.

- [ ] **Step 6: Full-suite guard**

Run: `poetry run pytest -q`
Expected: green (~3083). Registering under a base must not change any functor's behavior; the full suite is the safety net for a mis-registration.

- [ ] **Step 7: Commit**

```bash
git add bindings/ tests/test_evalop_registry.py
git commit -m "feat(core): register all functors under EvalOp (engine can hold any op)"
```

---

## Self-Review

**1. Spec coverage:** the DAG-2 spec's "bulk-register all FunctorBase functors under EvalOp" (deferred from DAG-2a) → Task 1, with the pre-classified `FunctorBase`→`EvalOp` / `ScreamerBase`-bare→`ScreamerBase` split. ✓ After this, `py::cast<EvalOp*>` succeeds for every functor — the DAG-2b engine's precondition. ✓

**2. Placeholder scan:** none — the two class lists are explicit; the per-class change is shown exactly. Constructor args in the test may need a one-off correction if a class's signature differs; the implementer verifies against the class binding (the arities themselves — `num_inputs`/`num_outputs` — are fixed by the C++ `FunctorBase<_,N,M>` and are correct as listed).

**3. Type consistency:** `py::class_<X, EvalOp>` / `py::class_<X, ScreamerBase>` and the `num_inputs`/`num_outputs` property names match DAG-2a. No new names introduced.

**Note (design/clutter):** this increment adds a base argument to ~50 existing bindings — pre-existing per-class binding boilerplate, not new clutter. A future consolidation (a `bind_functor<T>(m, name)` helper encapsulating the `py::class_` + base + `init` + `__call__` + `reset` pattern) would remove that boilerplate library-wide; it is out of scope here (it would touch *all* functor bindings, not just these) and noted for a later cleanup.

---

## Follow-on (DAG-2b, remaining increments)

- **DAG-2b-2** — wide-edge value-span push interface + generalized `FunctorNode` (drives an `EvalOp` at any width) + hand-wired batch execution, byte-identity to eager.
- **DAG-2b-3** — `CombineLatestNode` (push fan-in) + fan-out/broadcast.
- **DAG-2b-4** — C++ graph representation + builder + compiler.
- **DAG-2b-5** — batch-replay + streaming drivers + `align_outputs` + thin `dag.py` cutover + DAG-1 `_run` → test oracle.
