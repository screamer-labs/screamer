# Adaptive Overlays (InstantaneousTrendline, MAMA) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two adaptive moving averages that reuse the cycle engine: `InstantaneousTrendline` and `MAMA` (with its companion `FAMA`).

**Architecture:** Both operators reuse `screamer::detail::HilbertCycle` (from the cycle-core plan) for the measured dominant-cycle period and the instantaneous phase, then apply an adaptive recursion whose smoothing tracks that measurement. `InstantaneousTrendline` uses the period; `MAMA` uses the phase rate of change. Both are thin `FunctorBase` nodes holding a `HilbertCycle`.

**Tech Stack:** C++17, pybind11, pytest, numpy, TA-Lib (`talib`, a loose reference for MAMA only).

**Prerequisite:** the cycle-core plan (`HilbertCycle` engine) must be merged first. This plan includes `include/screamer/detail/hilbert_cycle.h` and reads `engine_.period()` and `engine_.phase()`.

## Correctness strategy

Like the cycle core, these have no bit-exact external oracle (TA-Lib's `HT_TRENDLINE`/`MAMA` use the frozen TA-Lib engine, not our modern `HilbertCycle`). Gates, in order of strength:

1. **batch == stream** (`assert_batch_equals_scalar`) - hard gate, always required.
2. **Behavioral**: on a trending ramp, `InstantaneousTrendline` tracks the ramp with bounded lag; `MAMA` responds faster than `FAMA` (MAMA leads, FAMA follows), and both track a step change.
3. **MAMA loose reference**: compare `MAMA` shape against `talib.MAMA` (expect close-but-not-identical because the analytic-signal front end differs); assert high correlation on the overlap, not equality.

Flag both operators for the user's domain review (the `MAMA` DeltaPhase sign convention and the `InstantaneousTrendline` alpha formula are the review-worthy points).

## Global Constraints

- All logic in the C++ core; Python layer is the binding only. Reuse `HilbertCycle`; do not reimplement the analytic-signal front end.
- All three regimes identical, causal, batch == stream. `nan_policy: ignore`. Warm-up emits `NaN` (until `HilbertCycle` is ready).
- `MAMA` is `FunctorBase<MAMA, 1, 2>` returning `(mama, fama)`; `InstantaneousTrendline` is `FunctorBase<_, 1, 1>`. Bind against `EvalOp` with `handle_input` in `bindings/bindings_signal.cpp`. Confirm the 1-output return convention against an existing `FunctorBase<_, _, 1>` node.
- Both ship in the moving-average docs neighborhood with topic `smoothing` (per the design decision: `MAMA` groups with the moving averages). Docs pages in `docs/functions_signal/`.
- Constructor defaults so `build_help_registry` can instantiate with all defaults: `MAMA(double fast_limit = 0.5, double slow_limit = 0.05)`; `InstantaneousTrendline()` parameterless.
- After any C++ change run `make build`; `make tidy` must be clean. Docs prose per CONTRIBUTING.md (unit-agnostic, mechanism, no em dashes, no editorializing).

---

## Task 1: MAMA (MESA Adaptive Moving Average)

**Files:** Create `include/screamer/mama.h`; modify `bindings/bindings_signal.cpp`; create `devtools/baselines/MAMA.py`; create `docs/functions_signal/MAMA.md`; test `tests/test_overlays.py`.

**Interfaces:** Consumes `detail::HilbertCycle` (`phase()`). Produces `screamer::MAMA`, `FunctorBase<MAMA, 1, 2>`, `MAMA(fast_limit=0.5, slow_limit=0.05)`, returning `(mama, fama)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_overlays.py`:

```python
import numpy as np
import pytest

from screamer import MAMA
from tests.regime_helpers import assert_batch_equals_scalar


def _series(n=600, seed=0):
    rng = np.random.default_rng(seed)
    return 100 + np.cumsum(rng.standard_normal(n))


class TestMAMA:
    def test_mama_tracks_and_leads_fama(self):
        x = _series(600)
        mama, fama = MAMA()(x)
        mama = np.asarray(mama); fama = np.asarray(fama)
        m = np.isfinite(mama) & np.isfinite(fama)
        assert m.sum() > 100
        # MAMA and FAMA both track the series (bounded error), MAMA closer.
        err_mama = np.mean(np.abs(mama[m] - x[m]))
        err_fama = np.mean(np.abs(fama[m] - x[m]))
        assert err_mama < err_fama  # FAMA follows MAMA with more lag.

    def test_loose_talib_reference(self):
        import talib
        x = _series(800, seed=2)
        mama, _ = MAMA()(x)
        ref, _ = talib.MAMA(x, fastlimit=0.5, slowlimit=0.05)
        mama = np.asarray(mama)
        m = np.isfinite(mama) & np.isfinite(ref)
        assert m.sum() > 100
        # Different analytic-signal front end -> not identical, but correlated.
        c = np.corrcoef(mama[m], ref[m])[0, 1]
        assert c > 0.9, f"correlation {c:.3f}"

    def test_batch_equals_stream(self):
        x = _series(400, seed=3)
        assert_batch_equals_scalar(lambda: MAMA(), x)
```

- [ ] **Step 2: Run to verify it fails** — `poetry run pytest tests/test_overlays.py -q` → FAIL (ImportError).

- [ ] **Step 3: Write the node**

Create `include/screamer/mama.h`:

```cpp
#ifndef SCREAMER_MAMA_H
#define SCREAMER_MAMA_H

// MAMA: MESA Adaptive Moving Average (John Ehlers). The smoothing factor adapts
// to the rate of change of the instantaneous phase: when the phase moves fast
// (a new cycle), it tracks quickly; when the phase stalls, it smooths heavily.
// FAMA (following adaptive MA) is a second, slower pass. Returns (mama, fama).
// Reuses detail::HilbertCycle for the phase.

#include <algorithm>
#include <cmath>
#include <limits>
#include "screamer/common/functor_base.h"
#include "screamer/detail/hilbert_cycle.h"

namespace screamer {

class MAMA : public FunctorBase<MAMA, 1, 2> {
public:
    explicit MAMA(double fast_limit = 0.5, double slow_limit = 0.05)
        : fast_(fast_limit), slow_(slow_limit) {}

    void reset() override {
        engine_.reset();
        prev_phase_ = std::numeric_limits<double>::quiet_NaN();
        mama_ = fama_ = std::numeric_limits<double>::quiet_NaN();
    }

    ResultTuple call(const InputArray& inputs) override {
        const double price = inputs[0];
        engine_.update(price);
        const double phase = engine_.phase();
        const double nan = std::numeric_limits<double>::quiet_NaN();
        if (std::isnan(phase)) {
            return std::make_tuple(nan, nan);
        }
        // DeltaPhase: how far the phase advanced since the last sample. Ehlers
        // measures it as prev_phase - phase and floors it at 1 degree.
        double dphase = std::isnan(prev_phase_) ? 1.0 : (prev_phase_ - phase);
        prev_phase_ = phase;
        if (dphase < 1.0) dphase = 1.0;
        double alpha = fast_ / dphase;
        if (alpha < slow_) alpha = slow_;
        if (alpha > fast_) alpha = fast_;
        if (std::isnan(mama_)) {
            mama_ = price;
            fama_ = price;
        } else {
            mama_ = alpha * price + (1.0 - alpha) * mama_;
            fama_ = 0.5 * alpha * mama_ + (1.0 - 0.5 * alpha) * fama_;
        }
        return std::make_tuple(mama_, fama_);
    }

private:
    detail::HilbertCycle engine_;
    const double fast_;
    const double slow_;
    double prev_phase_ = std::numeric_limits<double>::quiet_NaN();
    double mama_ = std::numeric_limits<double>::quiet_NaN();
    double fama_ = std::numeric_limits<double>::quiet_NaN();
};

}  // namespace screamer

#endif  // SCREAMER_MAMA_H
```

Note: the `dphase = prev_phase - phase` sign follows Ehlers' convention (phase decreases as the analytic signal rotates). If `HilbertCycle`'s phase increases over time, this yields a near-constant clamp and `MAMA` degenerates to a slow EMA - the `test_mama_tracks_and_leads_fama` and the loose talib correlation will catch it. If they fail, flip to `dphase = phase - prev_phase`, unwrap into `(0, 360]`, and re-test. Record which convention passed in your report.

- [ ] **Step 4: Binding** — add include and:

```cpp
    py::class_<screamer::MAMA, screamer::EvalOp>(m, "MAMA")
        .def(py::init<double, double>(),
             py::arg("fast_limit") = 0.5, py::arg("slow_limit") = 0.05)
        .def("__call__", &screamer::MAMA::handle_input)
        .def("reset", &screamer::MAMA::reset, "Reset to the initial state.");
```

- [ ] **Step 5: Build and test** — `make build` then `poetry run pytest tests/test_overlays.py::TestMAMA -q` → PASS (apply the DeltaPhase note if needed).

- [ ] **Step 6: Baseline, docs, coverage, tidy, commit**

Create `devtools/baselines/MAMA.py`:

```python
import talib


class MAMA_talib:
    def __init__(self, fast_limit=0.5, slow_limit=0.05):
        self.f, self.s = fast_limit, slow_limit

    def __call__(self, array):
        mama, _fama = talib.MAMA(array, fastlimit=self.f, slowlimit=self.s)
        return mama
```

Note in your report: this baseline is a loose reference (different analytic-signal front end), so `tests/test_baselines.py` may show a mismatch; the authoritative gates are the behavioral and correlation tests in `TestMAMA`. If the baseline harness asserts exact equality, record it for the controller to decide on exclusion.

Create `docs/functions_signal/MAMA.md` (`outputs: 2`, topic `smoothing`, tags `ehlers`/`adaptive`, params `fast_limit` default 0.5 and `slow_limit` default 0.05, `nan_policy: ignore`; describe the phase-rate adaptation and the MAMA/FAMA pair). Regenerate help registry + topic pages, run `test_doc_coverage.py`, `make tidy`. Commit:

```bash
git add include/screamer/mama.h bindings/bindings_signal.cpp \
        devtools/baselines/MAMA.py docs/functions_signal/MAMA.md tests/test_overlays.py
git commit -m "feat(cycle): add MAMA adaptive moving average"
```

---

## Task 2: InstantaneousTrendline

**Files:** Create `include/screamer/instantaneous_trendline.h`; modify `bindings/bindings_signal.cpp`; create `docs/functions_signal/InstantaneousTrendline.md`; test in `tests/test_overlays.py`.

**Interfaces:** Consumes `detail::HilbertCycle` (`period()`). Produces `screamer::InstantaneousTrendline`, `FunctorBase<_, 1, 1>`, parameterless.

- [ ] **Step 1: Write the failing test** — append to `tests/test_overlays.py`:

```python
from screamer import InstantaneousTrendline


class TestInstantaneousTrendline:
    def test_tracks_ramp_with_bounded_lag(self):
        n = 800
        x = np.linspace(0.0, 100.0, n)  # clean upward ramp
        it = np.asarray(InstantaneousTrendline()(x))
        tail = slice(-200, None)
        m = np.isfinite(it[tail])
        # On a steady ramp the trendline tracks price with small bounded error.
        assert m.sum() > 50
        assert np.max(np.abs(it[tail][m] - x[tail][m])) < 5.0

    def test_batch_equals_stream(self):
        x = _series(400, seed=5)
        assert_batch_equals_scalar(lambda: InstantaneousTrendline(), x)
```

- [ ] **Step 2: Run to verify it fails** — FAIL (ImportError).

- [ ] **Step 3: Write the node**

Create `include/screamer/instantaneous_trendline.h`:

```cpp
#ifndef SCREAMER_INSTANTANEOUS_TRENDLINE_H
#define SCREAMER_INSTANTANEOUS_TRENDLINE_H

// InstantaneousTrendline: Ehlers' adaptive trendline. A 2-pole recursion whose
// smoothing factor alpha is set from the measured dominant cycle period, so the
// trendline follows the trend and removes the dominant cycle. Reuses
// detail::HilbertCycle for the period.
//
//   alpha = 2 / (period + 1)
//   it[t] = (alpha - alpha^2/4) price[t] + 0.5 alpha^2 price[t-1]
//           - (alpha - 0.75 alpha^2) price[t-2]
//           + 2 (1-alpha) it[t-1] - (1-alpha)^2 it[t-2]

#include <cmath>
#include <limits>
#include "screamer/common/functor_base.h"
#include "screamer/detail/hilbert_cycle.h"

namespace screamer {

class InstantaneousTrendline : public FunctorBase<InstantaneousTrendline, 1, 1> {
public:
    InstantaneousTrendline() = default;

    void reset() override {
        engine_.reset();
        p1_ = p2_ = std::numeric_limits<double>::quiet_NaN();
        it1_ = it2_ = std::numeric_limits<double>::quiet_NaN();
    }

    ResultTuple call(const InputArray& inputs) override {
        const double price = inputs[0];
        engine_.update(price);
        const double period = engine_.period();
        const double nan = std::numeric_limits<double>::quiet_NaN();
        if (std::isnan(period) || period <= 0.0) {
            return std::make_tuple(nan);
        }
        const double a = 2.0 / (period + 1.0);
        double it;
        if (std::isnan(it1_) || std::isnan(p1_) || std::isnan(p2_)) {
            it = price;  // seed the recursion once the period is available.
        } else {
            it = (a - a * a / 4.0) * price + 0.5 * a * a * p1_
                 - (a - 0.75 * a * a) * p2_
                 + 2.0 * (1.0 - a) * it1_ - (1.0 - a) * (1.0 - a) * it2_;
        }
        p2_ = p1_;
        p1_ = price;
        it2_ = it1_;
        it1_ = it;
        return std::make_tuple(it);
    }

private:
    detail::HilbertCycle engine_;
    double p1_ = std::numeric_limits<double>::quiet_NaN();
    double p2_ = std::numeric_limits<double>::quiet_NaN();
    double it1_ = std::numeric_limits<double>::quiet_NaN();
    double it2_ = std::numeric_limits<double>::quiet_NaN();
};

}  // namespace screamer

#endif  // SCREAMER_INSTANTANEOUS_TRENDLINE_H
```

Note: the two-bar price/trendline seeds (`p1_`, `p2_`, `it1_`, `it2_`) start once the period is available. The ramp test tolerates the seeding transient by checking only the last 200 samples. If the recursion is unstable (diverges) for very short periods, confirm the period is clamped to `[6, 50]` by `HilbertCycle` (it is) so `alpha <= 2/7`. Record any instability in your report.

- [ ] **Step 4: Binding** — add include and:

```cpp
    py::class_<screamer::InstantaneousTrendline, screamer::EvalOp>(m, "InstantaneousTrendline")
        .def(py::init<>())
        .def("__call__", &screamer::InstantaneousTrendline::handle_input)
        .def("reset", &screamer::InstantaneousTrendline::reset, "Reset to the initial state.");
```

- [ ] **Step 5: Build and test** — `make build` then `poetry run pytest tests/test_overlays.py::TestInstantaneousTrendline -q` → PASS.

- [ ] **Step 6: Docs, coverage, tidy, full suite, commit**

Create `docs/functions_signal/InstantaneousTrendline.md` (topic `smoothing`, tags `ehlers`/`adaptive`/`trend`, `parameters: []`, `nan_policy: ignore`; describe the adaptive 2-pole trendline that removes the dominant cycle). Regenerate help registry + topic pages, `test_doc_coverage.py`, `make tidy`, then the full suite:

```bash
poetry run pytest -q
```
Expected: green, zero skips. Commit:

```bash
git add include/screamer/instantaneous_trendline.h bindings/bindings_signal.cpp \
        docs/functions_signal/InstantaneousTrendline.md tests/test_overlays.py
git commit -m "feat(cycle): add InstantaneousTrendline adaptive trendline"
```

---

## Notes for the human (domain review)

- Both operators reuse the `HilbertCycle` engine, so their correctness rests on the engine's (pure-tone-validated) period and phase plus the adaptive recursion here.
- `MAMA`'s DeltaPhase sign convention (`prev_phase - phase` vs the reverse) is the review-worthy point; the implementer records which convention passed the behavioral and loose-talib-correlation gates.
- `InstantaneousTrendline`'s `alpha = 2/(period+1)` and the 2-pole recursion are Ehlers' adaptive form; there is no bit-exact oracle. Gated by batch==stream and the ramp-tracking behavioral test.
```
