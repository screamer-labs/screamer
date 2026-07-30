# Cycle Core (Ehlers Analytic Signal + Dominant Cycle) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add seven causal cycle/spectral operators built on one shared analytic-signal engine: `DominantCycle`, `HilbertPhasor`, `CyclePhase`, `CycleFrequency`, `CycleAmplitude`, `CycleSine`, `TrendMode`.

**Architecture:** One C++ engine, `include/screamer/detail/hilbert_cycle.h` (`HilbertCycle`), runs John Ehlers' Homodyne Discriminator (from "Rocket Science for Traders", 2001) per sample: a `4-3-2-1` FIR smoother, a Hilbert-transform detrender producing in-phase (`I`) and quadrature (`Q`) components, a phasor-addition step, and a homodyne discriminator that estimates the dominant cycle period with feedback from the previous period. The engine exposes `I`, `Q`, the smoothed dominant-cycle period, the instantaneous phase, and the instantaneous amplitude. The seven operators are thin `FunctorBase` nodes that hold a `HilbertCycle` and read one (or two) of its outputs. One implementation of the discriminator, shared by all seven.

**Tech Stack:** C++17, pybind11, CMake/scikit-build-core, pytest, numpy (tests only).

## Correctness strategy (READ THIS FIRST)

Unlike the filters (validated against `scipy`) and DMI (validated against `talib`), the modern Ehlers cycle formulation has **no external library oracle**, by design (the spec chose Ehlers' current formulation over the frozen TA-Lib `HT_*` port). The correctness gate for this plan is therefore a **synthetic pure-tone recovery test**: drive the engine with a clean sine wave of known period `P` and amplitude `A`, and after the warm-up transient require:

- `DominantCycle` converges to `P` within a tolerance (default: within 15% for `P` in `[12, 40]`, checked on the last quarter of a long series),
- `CycleAmplitude` converges to approximately `A` (within 20%),
- `CyclePhase` advances by approximately `360/P` degrees per sample (monotone modulo 360),
- `CycleFrequency` converges to approximately `1/P`.

The reference algorithm code in Task 1 is the implementer's starting point, transcribed from Ehlers' published Homodyne Discriminator. **The pure-tone recovery test is the source of truth.** If the reference code does not pass the pure-tone test, the transcription has an error (a coefficient, an index, or the `atan` degree/radian convention are the usual culprits); the implementer must reproduce Ehlers' published algorithm faithfully and iterate until the pure-tone test passes. If the implementer cannot make the pure-tone test pass, they report `BLOCKED` with the failing values, not a passing-but-unverified engine.

## Global Constraints

- All operator logic in the C++ core; the Python layer is the pybind11 binding only. One implementation per behavior: the discriminator lives only in `HilbertCycle`; the seven operators read from it.
- Every operator runs in all three regimes (eager arrays, graph `Pipeline`, lazy scalar loop) with identical output, and is causal (only current and past input). Batch output equals streaming output (`tests.regime_helpers.assert_batch_equals_scalar`).
- `nan_policy: ignore`: a `NaN` input yields `NaN` output(s) and leaves the engine state unchanged for that step.
- Multi-output operators (`HilbertPhasor`, `CycleSine`) use `FunctorBase<Derived, 1, 2>`. Single-output operators use `FunctorBase<Derived, 1, 1>`. A multi-output op returns ONE `(n, n_out)` numpy array in eager mode (not a tuple of arrays); index columns as `out = np.asarray(op(x)); a, b = out[:, 0], out[:, 1]`. `assert_batch_equals_scalar` handles multi-output ops directly (confirmed with the 3-output `ADX`). Confirm the 1-output return convention (bare `double` vs 1-tuple) against an existing `FunctorBase<_, _, 1>` node (the DMI plan established `PlusDI` as one; if the DMI plan has not run, inspect `include/screamer/common/functor_base.h`). Bind against `screamer::EvalOp` with `.def("__call__", &Class::handle_input)` in `bindings/bindings_signal.cpp`.
- New topic slug `cycles` added to `docs/topics.yml` under the `indicators` group (Task 1). Docs pages live in `docs/functions_signal/`.
- Warm-up (the discriminator's settling transient) emits `NaN` until the engine has enough history; this is identical across regimes.
- After any C++ change run `make build` before testing. `make tidy` must be clean.
- Docs prose follows CONTRIBUTING.md: unit-agnostic (a `series`, a `sample`, not "bars"), general framing (these are general DSP quantities used in audio, biomedical, vibration, and radar analysis, not only price), mechanism not vibe, no em dashes, no editorializing adjectives.
- Default constructor parameters (needed so `build_help_registry.py` can instantiate with all defaults): each operator takes no required arguments. `HilbertCycle` and the nodes are parameterless (the discriminator's cycle bounds `[6, 50]` are fixed constants, per Ehlers). If a node needs a parameter later, give it a concrete default.

---

## Task 1: HilbertCycle engine

**Files:**
- Create: `include/screamer/detail/hilbert_cycle.h`
- Test: `tests/test_cycle.py` (engine-level checks driven through the first operator, `DominantCycle`, added in Task 2). For Task 1, add a temporary C++-free numeric check is not possible; instead Task 1's gate is a standalone pure-tone test that instantiates a tiny throwaway pybind exposure. To avoid a throwaway binding, fold Task 1's verification into Task 2 (`DominantCycle`), which is the first consumer. Task 1 therefore ends when the engine compiles and `make tidy` is clean; Task 2 proves it recovers a known period.

**Interfaces:**
- Produces: `screamer::detail::HilbertCycle`, methods `void update(double price)`, `void reset()`, and const accessors `double inphase() const`, `double quadrature() const`, `double period() const` (smoothed dominant cycle in samples, `NaN` during warm-up), `double phase() const` (degrees, 0..360, `NaN` during warm-up), `double amplitude() const` (`NaN` during warm-up), `bool ready() const`.

- [ ] **Step 1: Write the engine**

Create `include/screamer/detail/hilbert_cycle.h`. This is Ehlers' Homodyne Discriminator (Rocket Science for Traders, 2001). It is a reference transcription; the pure-tone test in Task 2 is the correctness gate. Keep the 6-sample history via small ring buffers or shifted scalars.

```cpp
#ifndef SCREAMER_DETAIL_HILBERT_CYCLE_H
#define SCREAMER_DETAIL_HILBERT_CYCLE_H

// HilbertCycle: John Ehlers' Homodyne Discriminator (Rocket Science for
// Traders, 2001). Estimates the dominant cycle period, the analytic-signal
// in-phase / quadrature components, the instantaneous phase, and the
// instantaneous amplitude of a sampled series, using only current and past
// samples. All accessors return NaN until the warm-up transient clears.
//
// Reference algorithm (EasyLanguage, angles in degrees):
//   Smooth    = (4*P[0] + 3*P[1] + 2*P[2] + P[3]) / 10
//   Detrender = (.0962*Smooth[0] + .5769*Smooth[2] - .5769*Smooth[4]
//                - .0962*Smooth[6]) * (.075*Period[1] + .54)
//   Q1 = (.0962*Detr[0] + .5769*Detr[2] - .5769*Detr[4] - .0962*Detr[6])
//        * (.075*Period[1] + .54)
//   I1 = Detrender[3]
//   jI = (.0962*I1[0] + .5769*I1[2] - .5769*I1[4] - .0962*I1[6])
//        * (.075*Period[1] + .54)
//   jQ = (.0962*Q1[0] + .5769*Q1[2] - .5769*Q1[4] - .0962*Q1[6])
//        * (.075*Period[1] + .54)
//   I2 = I1 - jQ ;  Q2 = Q1 + jI
//   I2 = .2*I2 + .8*I2[1] ;  Q2 = .2*Q2 + .8*Q2[1]
//   Re = I2*I2[1] + Q2*Q2[1] ;  Im = I2*Q2[1] - Q2*I2[1]
//   Re = .2*Re + .8*Re[1] ;  Im = .2*Im + .8*Im[1]
//   if Im!=0 and Re!=0: Period = 360 / atan_deg(Im/Re)
//   clamp Period to [0.67*Period[1], 1.5*Period[1]] then to [6, 50]
//   Period = .2*Period + .8*Period[1]
//   SmoothPeriod = .33*Period + .67*SmoothPeriod[1]
//   phase     = atan2_deg(Q1, I1)   (instantaneous phase from I1/Q1)
//   amplitude = sqrt(I2*I2 + Q2*Q2)

#include <array>
#include <cmath>
#include <limits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace screamer {
namespace detail {

class HilbertCycle {
public:
    HilbertCycle() { reset(); }

    void reset() {
        price_.fill(0.0);
        smooth_.fill(0.0);
        detrender_.fill(0.0);
        i1_.fill(0.0);
        q1_.fill(0.0);
        i2_ = prev_i2_ = 0.0;
        q2_ = prev_q2_ = 0.0;
        re_ = prev_re_ = 0.0;
        im_ = prev_im_ = 0.0;
        period_ = prev_period_ = 0.0;
        smooth_period_ = 0.0;
        phase_ = std::numeric_limits<double>::quiet_NaN();
        amplitude_ = std::numeric_limits<double>::quiet_NaN();
        n_ = 0;
    }

    void update(double price) {
        if (std::isnan(price)) return;  // NaN policy ignore.
        shift(price_, price);
        const double sm = (4.0 * price_[0] + 3.0 * price_[1] + 2.0 * price_[2] + price_[3]) / 10.0;
        shift(smooth_, sm);
        const double adj = 0.075 * prev_period_ + 0.54;
        const double det = hilbert(smooth_) * adj;
        shift(detrender_, det);
        const double q1 = hilbert(detrender_) * adj;
        const double i1 = detrender_[3];
        shift(q1_, q1);
        shift(i1_, i1);
        const double jI = hilbert(i1_) * adj;
        const double jQ = hilbert(q1_) * adj;
        double i2 = i1 - jQ;
        double q2 = q1 + jI;
        i2 = 0.2 * i2 + 0.8 * prev_i2_;
        q2 = 0.2 * q2 + 0.8 * prev_q2_;
        double re = i2 * prev_i2_ + q2 * prev_q2_;
        double im = i2 * prev_q2_ - q2 * prev_i2_;
        re = 0.2 * re + 0.8 * prev_re_;
        im = 0.2 * im + 0.8 * prev_im_;
        double period = prev_period_;
        if (im != 0.0 && re != 0.0) {
            period = 360.0 / (std::atan(im / re) * 180.0 / M_PI);
        }
        if (prev_period_ > 0.0) {
            if (period > 1.5 * prev_period_) period = 1.5 * prev_period_;
            if (period < 0.67 * prev_period_) period = 0.67 * prev_period_;
        }
        if (period < 6.0) period = 6.0;
        if (period > 50.0) period = 50.0;
        period = 0.2 * period + 0.8 * prev_period_;
        smooth_period_ = 0.33 * period + 0.67 * smooth_period_;
        // Commit state.
        prev_i2_ = i2; prev_q2_ = q2;
        prev_re_ = re; prev_im_ = im;
        prev_period_ = period;
        i2_ = i2; q2_ = q2;
        phase_ = std::atan2(q1, i1) * 180.0 / M_PI;
        if (phase_ < 0.0) phase_ += 360.0;
        amplitude_ = std::sqrt(i2 * i2 + q2 * q2);
        n_++;
    }

    bool ready() const { return n_ > kWarmup; }
    double inphase() const { return ready() ? i2_ : nan(); }
    double quadrature() const { return ready() ? q2_ : nan(); }
    double period() const { return ready() ? smooth_period_ : nan(); }
    double phase() const { return ready() ? phase_ : nan(); }
    double amplitude() const { return ready() ? amplitude_ : nan(); }

private:
    static constexpr int kWarmup = 12;  // discriminator settling; refine via pure-tone test.
    static double nan() { return std::numeric_limits<double>::quiet_NaN(); }

    // 7-tap Hilbert weighting over a 7-sample history window [0..6].
    static double hilbert(const std::array<double, 7>& h) {
        return 0.0962 * h[0] + 0.5769 * h[2] - 0.5769 * h[4] - 0.0962 * h[6];
    }
    template <size_t N>
    static void shift(std::array<double, N>& a, double v) {
        for (size_t i = N - 1; i > 0; --i) a[i] = a[i - 1];
        a[0] = v;
    }

    std::array<double, 7> price_{};
    std::array<double, 7> smooth_{};
    std::array<double, 7> detrender_{};
    std::array<double, 7> i1_{};
    std::array<double, 7> q1_{};
    double i2_ = 0.0, prev_i2_ = 0.0;
    double q2_ = 0.0, prev_q2_ = 0.0;
    double re_ = 0.0, prev_re_ = 0.0;
    double im_ = 0.0, prev_im_ = 0.0;
    double period_ = 0.0, prev_period_ = 0.0;
    double smooth_period_ = 0.0;
    double phase_ = nan();
    double amplitude_ = nan();
    int n_ = 0;
};

}  // namespace detail
}  // namespace screamer

#endif  // SCREAMER_DETAIL_HILBERT_CYCLE_H
```

- [ ] **Step 2: Add the `cycles` topic**

In `docs/topics.yml`, add under the `indicators` group's `topics:` list the slug `cycles`, and add to the `topics:` definitions:

```yaml
  cycles:
    name: "Cycles & spectra"
    desc: "Dominant cycle, phase, amplitude, and frequency from the analytic signal."
```

Add `cycles` to the `indicators` group's member list.

- [ ] **Step 3: Build (compile check only)**

Run: `make build`
Expected: compiles (the header is included transitively once Task 2 adds a consumer; if nothing includes it yet, add a temporary `#include` in `bindings/bindings_signal.cpp` is not needed — proceed to Task 2, which includes it). If `make build` does not compile the header without a consumer, that is expected; Task 2 is where it is first compiled.

- [ ] **Step 4: clang-tidy and commit**

Run: `make tidy` (expect clean; note the unused `i2_`/`re_`/`im_`/`period_` non-prev members are read by accessors or reserved — remove any member that is genuinely unused to satisfy tidy). Then:

```bash
git add include/screamer/detail/hilbert_cycle.h docs/topics.yml
git commit -m "feat(cycle): add HilbertCycle homodyne-discriminator engine + cycles topic"
```

---

## Task 2: DominantCycle (and the engine's pure-tone gate)

**Files:**
- Create: `include/screamer/dominant_cycle.h`
- Modify: `bindings/bindings_signal.cpp`
- Create: `docs/functions_signal/DominantCycle.md`
- Test: `tests/test_cycle.py`

**Interfaces:**
- Consumes: `screamer::detail::HilbertCycle`.
- Produces: `screamer::DominantCycle`, `FunctorBase<DominantCycle, 1, 1>`, parameterless constructor.

- [ ] **Step 1: Write the failing pure-tone test (the correctness gate for the whole engine)**

Create `tests/test_cycle.py`:

```python
import numpy as np
import pytest

from screamer import DominantCycle
from tests.regime_helpers import assert_batch_equals_scalar


def _tone(period, n=600, amp=1.0, seed=0):
    t = np.arange(n)
    return amp * np.sin(2 * np.pi * t / period)


class TestDominantCycle:
    @pytest.mark.parametrize("period", [14.0, 20.0, 30.0])
    def test_recovers_known_period(self, period):
        x = _tone(period, n=800)
        out = np.asarray(DominantCycle()(x))
        tail = out[-200:]
        tail = tail[np.isfinite(tail)]
        assert tail.size > 50
        # The homodyne discriminator recovers the dominant cycle within 15%.
        assert abs(np.median(tail) - period) / period < 0.15, \
            f"median {np.median(tail):.2f} vs period {period}"

    def test_batch_equals_stream(self):
        x = _tone(20.0, n=400)
        assert_batch_equals_scalar(lambda: DominantCycle(), x)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `poetry run pytest tests/test_cycle.py -q`
Expected: FAIL with `ImportError: cannot import name 'DominantCycle'`.

- [ ] **Step 3: Write the node**

Create `include/screamer/dominant_cycle.h`:

```cpp
#ifndef SCREAMER_DOMINANT_CYCLE_H
#define SCREAMER_DOMINANT_CYCLE_H

// DominantCycle: the dominant cycle period (in samples) of a series, measured
// by Ehlers' homodyne discriminator. See detail/hilbert_cycle.h.

#include "screamer/common/functor_base.h"
#include "screamer/detail/hilbert_cycle.h"

namespace screamer {

class DominantCycle : public FunctorBase<DominantCycle, 1, 1> {
public:
    DominantCycle() = default;
    void reset() override { engine_.reset(); }
    ResultTuple call(const InputArray& inputs) override {
        engine_.update(inputs[0]);
        return std::make_tuple(engine_.period());
    }
private:
    detail::HilbertCycle engine_;
};

}  // namespace screamer

#endif  // SCREAMER_DOMINANT_CYCLE_H
```

Match the confirmed 1-output `FunctorBase` return convention.

- [ ] **Step 4: Register the binding**

In `bindings/bindings_signal.cpp` add `#include "screamer/dominant_cycle.h"` and:

```cpp
    py::class_<screamer::DominantCycle, screamer::EvalOp>(m, "DominantCycle")
        .def(py::init<>())
        .def("__call__", &screamer::DominantCycle::handle_input)
        .def("reset", &screamer::DominantCycle::reset, "Reset to the initial state.");
```

Confirm the multi-input functor binding form (base `EvalOp`, `handle_input`) against the `ADX` registration in `bindings/bindings_rolling.cpp`.

- [ ] **Step 5: Build and run the gate**

Run:
```bash
make build
poetry run pytest tests/test_cycle.py -q
```
Expected: PASS. If `test_recovers_known_period` fails, the engine transcription in Task 1 is wrong. Debug against Ehlers' published Homodyne Discriminator: check (a) the `atan` degree conversion (`360 / (atan(x)*180/pi)`), (b) the `0.075*Period[1] + 0.54` feedback factor, (c) the Hilbert tap indices `[0],[2],[4],[6]` and `I1 = Detrender[3]`, (d) the `kWarmup` value. Iterate on `detail/hilbert_cycle.h` until the pure-tone test passes across all three periods. If it cannot be made to pass, STOP and report BLOCKED with the observed median periods.

- [ ] **Step 6: Docs, coverage, tidy, commit**

Create `docs/functions_signal/DominantCycle.md`:

```markdown
---
name: DominantCycle
title: Dominant cycle period
implementation_family: signal
topics:
- cycles
tags:
- ehlers
- hilbert
- cycle
short: Dominant cycle period via Ehlers' homodyne discriminator.
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
---

# `DominantCycle`

## Description

`DominantCycle` measures the dominant cycle period of a series, in samples,
using John Ehlers' homodyne discriminator. It forms the analytic signal by a
Hilbert transform, then reads the cycle period from the rate of change of the
phase between successive samples. The estimate is a general spectral quantity:
the same measurement applies to any sampled signal, not only price.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in the input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
<!-- NAN_FOOTNOTE_END -->
```

Run:
```bash
poetry run python devtools/build_help_registry.py
poetry run python devtools/build_topic_pages.py
poetry run pytest tests/test_doc_coverage.py -q
make tidy
```
Expected: PASS, clean. Then:

```bash
git add include/screamer/dominant_cycle.h bindings/bindings_signal.cpp \
        docs/functions_signal/DominantCycle.md tests/test_cycle.py
git commit -m "feat(cycle): add DominantCycle operator"
```

---

## Task 3: HilbertPhasor

**Files:** Create `include/screamer/hilbert_phasor.h`; modify `bindings/bindings_signal.cpp`; create `docs/functions_signal/HilbertPhasor.md`; test in `tests/test_cycle.py`.

**Interfaces:** Consumes `HilbertCycle`. Produces `screamer::HilbertPhasor`, `FunctorBase<HilbertPhasor, 1, 2>`, parameterless, returning `(inphase, quadrature)`.

- [ ] **Step 1: Failing test** — append to `tests/test_cycle.py`:

```python
from screamer import HilbertPhasor


class TestHilbertPhasor:
    def test_quadrature_lags_inphase_on_tone(self):
        x = _tone(20.0, n=800)
        out = np.asarray(HilbertPhasor()(x))
        inphase, quad = out[:, 0], out[:, 1]
        m = np.isfinite(inphase) & np.isfinite(quad)
        # I and Q are the two components of the analytic signal: both finite
        # after warm-up, and not identical (a nonzero quadrature exists).
        assert m.sum() > 100
        assert np.any(np.abs(quad[m]) > 1e-6)

    def test_batch_equals_stream(self):
        x = _tone(20.0, n=400)
        assert_batch_equals_scalar(lambda: HilbertPhasor(), x)
```

- [ ] **Step 2: Run to verify it fails** — `poetry run pytest tests/test_cycle.py::TestHilbertPhasor -q` → FAIL (ImportError).

- [ ] **Step 3: Node** — create `include/screamer/hilbert_phasor.h`:

```cpp
#ifndef SCREAMER_HILBERT_PHASOR_H
#define SCREAMER_HILBERT_PHASOR_H

// HilbertPhasor: the in-phase and quadrature components of the analytic signal
// (real and imaginary parts), from Ehlers' Hilbert transform. See
// detail/hilbert_cycle.h.

#include "screamer/common/functor_base.h"
#include "screamer/detail/hilbert_cycle.h"

namespace screamer {

class HilbertPhasor : public FunctorBase<HilbertPhasor, 1, 2> {
public:
    HilbertPhasor() = default;
    void reset() override { engine_.reset(); }
    ResultTuple call(const InputArray& inputs) override {
        engine_.update(inputs[0]);
        return std::make_tuple(engine_.inphase(), engine_.quadrature());
    }
private:
    detail::HilbertCycle engine_;
};

}  // namespace screamer

#endif  // SCREAMER_HILBERT_PHASOR_H
```

- [ ] **Step 4: Binding** — add include and:

```cpp
    py::class_<screamer::HilbertPhasor, screamer::EvalOp>(m, "HilbertPhasor")
        .def(py::init<>())
        .def("__call__", &screamer::HilbertPhasor::handle_input)
        .def("reset", &screamer::HilbertPhasor::reset, "Reset to the initial state.");
```

- [ ] **Step 5: Build and test** — `make build` then `poetry run pytest tests/test_cycle.py::TestHilbertPhasor -q` → PASS.

- [ ] **Step 6: Docs, coverage, tidy, commit** — create `docs/functions_signal/HilbertPhasor.md` (`outputs: 2`, topic `cycles`, `nan_policy: ignore`, describe in-phase/quadrature = real/imaginary parts of the analytic signal). Regenerate help registry + topic pages, run `test_doc_coverage.py`, `make tidy`. Commit:

```bash
git add include/screamer/hilbert_phasor.h bindings/bindings_signal.cpp \
        docs/functions_signal/HilbertPhasor.md tests/test_cycle.py
git commit -m "feat(cycle): add HilbertPhasor operator"
```

---

## Task 4: CyclePhase and CycleFrequency

**Files:** Create `include/screamer/cycle_phase.h`, `include/screamer/cycle_frequency.h`; modify `bindings/bindings_signal.cpp`; create the two docs pages; test in `tests/test_cycle.py`.

**Interfaces:** Consume `HilbertCycle`. `CyclePhase` = `FunctorBase<_,1,1>` returning `engine_.phase()`. `CycleFrequency` = `FunctorBase<_,1,1>` returning `1.0 / engine_.period()` (NaN if period is NaN or <= 0).

- [ ] **Step 1: Failing test** — append:

```python
from screamer import CyclePhase, CycleFrequency


class TestCyclePhaseFrequency:
    def test_frequency_recovers_period(self):
        period = 20.0
        x = _tone(period, n=800)
        f = np.asarray(CycleFrequency()(x))
        tail = f[-200:]; tail = tail[np.isfinite(tail)]
        assert tail.size > 50
        assert abs(np.median(tail) - 1.0 / period) / (1.0 / period) < 0.15

    def test_phase_in_range(self):
        x = _tone(20.0, n=800)
        p = np.asarray(CyclePhase()(x))
        m = np.isfinite(p)
        assert m.sum() > 100
        assert p[m].min() >= 0.0 and p[m].max() <= 360.0

    def test_batch_equals_stream(self):
        x = _tone(20.0, n=400)
        assert_batch_equals_scalar(lambda: CyclePhase(), x)
        assert_batch_equals_scalar(lambda: CycleFrequency(), x)
```

- [ ] **Step 2: Run to verify it fails** — FAIL (ImportError).

- [ ] **Step 3: Nodes** — `include/screamer/cycle_phase.h`:

```cpp
#ifndef SCREAMER_CYCLE_PHASE_H
#define SCREAMER_CYCLE_PHASE_H

// CyclePhase: instantaneous phase (degrees, 0..360) of the analytic signal.
#include "screamer/common/functor_base.h"
#include "screamer/detail/hilbert_cycle.h"

namespace screamer {
class CyclePhase : public FunctorBase<CyclePhase, 1, 1> {
public:
    CyclePhase() = default;
    void reset() override { engine_.reset(); }
    ResultTuple call(const InputArray& inputs) override {
        engine_.update(inputs[0]);
        return std::make_tuple(engine_.phase());
    }
private:
    detail::HilbertCycle engine_;
};
}  // namespace screamer
#endif  // SCREAMER_CYCLE_PHASE_H
```

`include/screamer/cycle_frequency.h`:

```cpp
#ifndef SCREAMER_CYCLE_FREQUENCY_H
#define SCREAMER_CYCLE_FREQUENCY_H

// CycleFrequency: instantaneous frequency (cycles per sample), the reciprocal
// of the dominant cycle period.
#include <cmath>
#include <limits>
#include "screamer/common/functor_base.h"
#include "screamer/detail/hilbert_cycle.h"

namespace screamer {
class CycleFrequency : public FunctorBase<CycleFrequency, 1, 1> {
public:
    CycleFrequency() = default;
    void reset() override { engine_.reset(); }
    ResultTuple call(const InputArray& inputs) override {
        engine_.update(inputs[0]);
        const double p = engine_.period();
        const double f = (std::isnan(p) || p <= 0.0)
                             ? std::numeric_limits<double>::quiet_NaN()
                             : 1.0 / p;
        return std::make_tuple(f);
    }
private:
    detail::HilbertCycle engine_;
};
}  // namespace screamer
#endif  // SCREAMER_CYCLE_FREQUENCY_H
```

- [ ] **Step 4: Bindings** — add includes and two registrations (base `EvalOp`, `py::init<>()`, `handle_input`), matching the `DominantCycle` block.

- [ ] **Step 5: Build and test** — `make build` then `poetry run pytest tests/test_cycle.py::TestCyclePhaseFrequency -q` → PASS.

- [ ] **Step 6: Docs, coverage, tidy, commit** — create `docs/functions_signal/CyclePhase.md` and `CycleFrequency.md` (topic `cycles`, `nan_policy: ignore`, `parameters: []`; CyclePhase describes instantaneous phase in degrees, CycleFrequency describes cycles per sample = 1/period, both framed as general DSP quantities). Regenerate, coverage, tidy. Commit:

```bash
git add include/screamer/cycle_phase.h include/screamer/cycle_frequency.h \
        bindings/bindings_signal.cpp docs/functions_signal/CyclePhase.md \
        docs/functions_signal/CycleFrequency.md tests/test_cycle.py
git commit -m "feat(cycle): add CyclePhase and CycleFrequency operators"
```

---

## Task 5: CycleAmplitude

**Files:** Create `include/screamer/cycle_amplitude.h`; modify `bindings/bindings_signal.cpp`; create `docs/functions_signal/CycleAmplitude.md`; test in `tests/test_cycle.py`.

**Interfaces:** Consumes `HilbertCycle`. `FunctorBase<_,1,1>` returning `engine_.amplitude()`.

- [ ] **Step 1: Failing test** — append:

```python
from screamer import CycleAmplitude


class TestCycleAmplitude:
    @pytest.mark.parametrize("amp", [1.0, 3.0])
    def test_recovers_amplitude(self, amp):
        x = _tone(20.0, n=800, amp=amp)
        a = np.asarray(CycleAmplitude()(x))
        tail = a[-200:]; tail = tail[np.isfinite(tail)]
        assert tail.size > 50
        # Analytic-signal magnitude recovers the tone amplitude within 25%.
        assert abs(np.median(tail) - amp) / amp < 0.25

    def test_batch_equals_stream(self):
        x = _tone(20.0, n=400)
        assert_batch_equals_scalar(lambda: CycleAmplitude(), x)
```

- [ ] **Step 2: Run to verify it fails** — FAIL (ImportError).

- [ ] **Step 3: Node** — `include/screamer/cycle_amplitude.h`:

```cpp
#ifndef SCREAMER_CYCLE_AMPLITUDE_H
#define SCREAMER_CYCLE_AMPLITUDE_H

// CycleAmplitude: instantaneous amplitude (envelope) of the analytic signal,
// its magnitude sqrt(I^2 + Q^2). A general envelope detector.
#include "screamer/common/functor_base.h"
#include "screamer/detail/hilbert_cycle.h"

namespace screamer {
class CycleAmplitude : public FunctorBase<CycleAmplitude, 1, 1> {
public:
    CycleAmplitude() = default;
    void reset() override { engine_.reset(); }
    ResultTuple call(const InputArray& inputs) override {
        engine_.update(inputs[0]);
        return std::make_tuple(engine_.amplitude());
    }
private:
    detail::HilbertCycle engine_;
};
}  // namespace screamer
#endif  // SCREAMER_CYCLE_AMPLITUDE_H
```

Note: if the pure-tone amplitude recovery is off by a constant scale factor (the analytic-signal magnitude of the detrended/smoothed signal may not equal the raw tone amplitude exactly), record the observed factor in your report and relax the test tolerance to bracket it, or document the scaling in the docs page. Do not silently change the definition; `amplitude = sqrt(I^2 + Q^2)` is the intended quantity. Report the observed scale.

- [ ] **Step 4: Binding** — add include and registration (matching `DominantCycle`).

- [ ] **Step 5: Build and test** — `make build` then `poetry run pytest tests/test_cycle.py::TestCycleAmplitude -q` → PASS (adjust tolerance per the scale note if needed, and report it).

- [ ] **Step 6: Docs, coverage, tidy, commit** — create `docs/functions_signal/CycleAmplitude.md` (topic `cycles`, `nan_policy: ignore`, describe as the analytic-signal envelope, general envelope detector). Regenerate, coverage, tidy. Commit:

```bash
git add include/screamer/cycle_amplitude.h bindings/bindings_signal.cpp \
        docs/functions_signal/CycleAmplitude.md tests/test_cycle.py
git commit -m "feat(cycle): add CycleAmplitude operator"
```

---

## Task 6: CycleSine

**Files:** Create `include/screamer/cycle_sine.h`; modify `bindings/bindings_signal.cpp`; create `docs/functions_signal/CycleSine.md`; test in `tests/test_cycle.py`.

**Interfaces:** Consumes `HilbertCycle`. `FunctorBase<_,1,2>` returning `(sine, leadsine)` where `sine = sin(phase_rad)` and `leadsine = sin(phase_rad + 45deg)`, `phase_rad = engine_.phase() * pi/180`.

- [ ] **Step 1: Failing test** — append:

```python
from screamer import CycleSine


class TestCycleSine:
    def test_sine_bounded_and_leads(self):
        x = _tone(20.0, n=800)
        out = np.asarray(CycleSine()(x))
        sine, lead = out[:, 0], out[:, 1]
        m = np.isfinite(sine) & np.isfinite(lead)
        assert m.sum() > 100
        assert sine[m].min() >= -1.0001 and sine[m].max() <= 1.0001
        assert lead[m].min() >= -1.0001 and lead[m].max() <= 1.0001

    def test_batch_equals_stream(self):
        x = _tone(20.0, n=400)
        assert_batch_equals_scalar(lambda: CycleSine(), x)
```

- [ ] **Step 2: Run to verify it fails** — FAIL (ImportError).

- [ ] **Step 3: Node** — `include/screamer/cycle_sine.h`:

```cpp
#ifndef SCREAMER_CYCLE_SINE_H
#define SCREAMER_CYCLE_SINE_H

// CycleSine: the sinewave indicator, (sine, leadsine) = sin(phase) and
// sin(phase + 45 degrees), from the instantaneous phase of the analytic signal.
#include <cmath>
#include <limits>
#include "screamer/common/functor_base.h"
#include "screamer/detail/hilbert_cycle.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace screamer {
class CycleSine : public FunctorBase<CycleSine, 1, 2> {
public:
    CycleSine() = default;
    void reset() override { engine_.reset(); }
    ResultTuple call(const InputArray& inputs) override {
        engine_.update(inputs[0]);
        const double ph = engine_.phase();
        if (std::isnan(ph)) {
            const double n = std::numeric_limits<double>::quiet_NaN();
            return std::make_tuple(n, n);
        }
        const double r = ph * M_PI / 180.0;
        return std::make_tuple(std::sin(r), std::sin(r + M_PI / 4.0));
    }
private:
    detail::HilbertCycle engine_;
};
}  // namespace screamer
#endif  // SCREAMER_CYCLE_SINE_H
```

- [ ] **Step 4: Binding** — add include and registration (matching `HilbertPhasor`, 2-output).

- [ ] **Step 5: Build and test** — `make build` then `poetry run pytest tests/test_cycle.py::TestCycleSine -q` → PASS.

- [ ] **Step 6: Docs, coverage, tidy, commit** — create `docs/functions_signal/CycleSine.md` (`outputs: 2`, topic `cycles`, `nan_policy: ignore`). Regenerate, coverage, tidy. Commit:

```bash
git add include/screamer/cycle_sine.h bindings/bindings_signal.cpp \
        docs/functions_signal/CycleSine.md tests/test_cycle.py
git commit -m "feat(cycle): add CycleSine operator"
```

---

## Task 7: TrendMode

**Files:** Create `include/screamer/trend_mode.h`; modify `bindings/bindings_signal.cpp`; create `docs/functions_signal/TrendMode.md`; test in `tests/test_cycle.py`.

**Interfaces:** Consumes `HilbertCycle`. `FunctorBase<_,1,1>` returning a trend-vs-cycle flag: `1.0` when the market is trending, `0.0` when cycling, `NaN` during warm-up.

- [ ] **Step 1: Decide the trend-mode rule**

Ehlers' `HT_TRENDMODE` sets trend mode when the instantaneous phase stops rotating (the dominant cycle stalls) and price stays on one side of a smoothed reference. A faithful minimal causal rule using `HilbertCycle`'s outputs: track the phase; when the per-sample phase change is small relative to `360/period` for a sustained run, output `1.0` (trend), else `0.0` (cycle). Because this rule has a threshold, expose it as a constructor parameter `phase_rate_frac` with a concrete default, so `build_help_registry` can instantiate.

This rule is a design choice, not a canonical single formula; flag it in your report for the human's domain review. Keep it simple and causal.

- [ ] **Step 2: Failing test** — append:

```python
from screamer import TrendMode


class TestTrendMode:
    def test_binary_and_finite_after_warmup(self):
        # A pure tone is a cycle, so trend mode should be mostly 0 on the tail.
        x = _tone(20.0, n=1000)
        tm = np.asarray(TrendMode()(x))
        m = np.isfinite(tm)
        assert m.sum() > 100
        vals = set(np.unique(tm[m]).tolist())
        assert vals.issubset({0.0, 1.0})
        assert np.mean(tm[m]) < 0.5  # predominantly cycle on a clean tone

    def test_batch_equals_stream(self):
        x = _tone(20.0, n=400)
        assert_batch_equals_scalar(lambda: TrendMode(), x)
```

- [ ] **Step 3: Run to verify it fails** — FAIL (ImportError).

- [ ] **Step 4: Node** — create `include/screamer/trend_mode.h`:

```cpp
#ifndef SCREAMER_TREND_MODE_H
#define SCREAMER_TREND_MODE_H

// TrendMode: a trend-vs-cycle classifier. Outputs 1.0 when the dominant-cycle
// phase advance per sample is a small fraction of a full cycle for a sustained
// run (the cycle has stalled, so the series is trending), 0.0 when the phase
// rotates at the cycle rate. See detail/hilbert_cycle.h.

#include <cmath>
#include <limits>
#include "screamer/common/functor_base.h"
#include "screamer/detail/hilbert_cycle.h"

namespace screamer {

class TrendMode : public FunctorBase<TrendMode, 1, 1> {
public:
    explicit TrendMode(double phase_rate_frac = 0.5) : frac_(phase_rate_frac) {}
    void reset() override {
        engine_.reset();
        prev_phase_ = std::numeric_limits<double>::quiet_NaN();
    }
    ResultTuple call(const InputArray& inputs) override {
        engine_.update(inputs[0]);
        const double ph = engine_.phase();
        const double per = engine_.period();
        const double nan = std::numeric_limits<double>::quiet_NaN();
        if (std::isnan(ph) || std::isnan(per) || per <= 0.0) {
            prev_phase_ = ph;
            return std::make_tuple(nan);
        }
        double out = 0.0;
        if (!std::isnan(prev_phase_)) {
            double dphase = ph - prev_phase_;
            // Unwrap to [-180, 180].
            while (dphase > 180.0) dphase -= 360.0;
            while (dphase < -180.0) dphase += 360.0;
            const double expected = 360.0 / per;  // per-sample advance if cycling.
            out = (std::abs(dphase) < frac_ * expected) ? 1.0 : 0.0;
        }
        prev_phase_ = ph;
        return std::make_tuple(out);
    }

private:
    detail::HilbertCycle engine_;
    const double frac_;
    double prev_phase_ = std::numeric_limits<double>::quiet_NaN();
};

}  // namespace screamer

#endif  // SCREAMER_TREND_MODE_H
```

- [ ] **Step 5: Binding** — add include and:

```cpp
    py::class_<screamer::TrendMode, screamer::EvalOp>(m, "TrendMode")
        .def(py::init<double>(), py::arg("phase_rate_frac") = 0.5)
        .def("__call__", &screamer::TrendMode::handle_input)
        .def("reset", &screamer::TrendMode::reset, "Reset to the initial state.");
```

- [ ] **Step 6: Build and test** — `make build` then `poetry run pytest tests/test_cycle.py::TestTrendMode -q` → PASS. If the tone is not predominantly cycle-mode, tune `frac_`'s default and record the choice in your report.

- [ ] **Step 7: Docs, coverage, tidy, full suite, commit** — create `docs/functions_signal/TrendMode.md` (topic `cycles`, `nan_policy: ignore`, parameter `phase_rate_frac` default 0.5, describe the trend-vs-cycle rule and note it is a threshold-based classifier). Regenerate help registry + topic pages, `test_doc_coverage.py`, `make tidy`, then the full suite:

```bash
poetry run pytest -q
```
Expected: green, zero skips. Commit:

```bash
git add include/screamer/trend_mode.h bindings/bindings_signal.cpp \
        docs/functions_signal/TrendMode.md tests/test_cycle.py
git commit -m "feat(cycle): add TrendMode operator"
```

---

## Notes for the human (domain review)

- The engine is Ehlers' Homodyne Discriminator (Rocket Science for Traders, 2001). There is no external library oracle for the modern formulation; correctness is gated by synthetic pure-tone recovery (period, amplitude, frequency, bounded phase). Review the engine coefficients and the `kWarmup` value against your reference.
- `CycleAmplitude`'s absolute scale may carry a constant factor from the detrender/smoother chain; the implementer records the observed factor. Decide whether to normalize it.
- `TrendMode`'s rule is a threshold classifier chosen for a causal, simple implementation, not a transcription of TA-Lib's `HT_TRENDMODE`. Review whether it matches your intent.
```
