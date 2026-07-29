# Ehlers Filter Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three reusable causal Ehlers filters to the screamer C++ core: `SuperSmoother`, `Decycler`, and `RoofingFilter`.

**Architecture:** Each filter is a fixed-coefficient IIR. A new coefficient-helper header `include/screamer/signal/ehlers.h` computes Ehlers' `(b, a)` coefficients from a cutoff, exactly as `signal/butter.h` does for Butterworth. Each node is a thin `ScreamerBase` subclass that feeds the shared `IIRFilter` (`signal/signal.h`), the same shape as the existing `Butter`. `RoofingFilter` chains a 2-pole highpass into a `SuperSmoother`, composing two `IIRFilter` stages in one node the way `MACD` composes `EwMean`.

**Tech Stack:** C++17, pybind11, CMake/scikit-build-core, pytest, scipy (baselines/tests only).

## Global Constraints

- All operator logic lives in the C++ core; the Python layer is the pybind11 binding only. No compute in Python.
- Every operator runs in all three regimes (eager arrays, graph `Pipeline`, lazy scalar loop) with identical output, and is causal (depends only on current and past input). Batch output equals streaming output.
- `nan_policy: ignore` for all three filters. `IIRFilter` already implements it (a `NaN` input yields a `NaN` output and leaves the delay line untouched).
- Cutoff is polymorphic: each filter accepts exactly one of `period` (samples) or `cutoff` (normalized, in `(0, 1)`, `1` = Nyquist), resolved internally. The mapping is `cutoff = 2 / period`, i.e. `period = 2 / cutoff`. This mirrors `EwMean`'s `com|span|halflife|alpha` exactly-one pattern.
- After any C++ change run `make build` (or `make install-dev`) before testing, or Python imports a stale binding.
- `make tidy` must be clean (clang-tidy `cppcoreguidelines-pro-type-member-init`): value-initialize every class member. An operator is not done until `make tidy` passes.
- Every operator ships a docs page `docs/functions_signal/<Name>.md` with validated YAML frontmatter and topics from `docs/topics.yml`. The `filtering` and `smoothing` topics already exist; no `topics.yml` change is needed in this plan.
- Docs prose follows CONTRIBUTING.md: lead with what the thing is, state the mechanism, unit-agnostic (`series`, not `bars`), no em dashes (ASCII hyphens only), no editorializing adjectives.
- Reference coefficient formulas (`period` in samples, angles in radians):
  - **SuperSmoother (2-pole lowpass):** `a1 = exp(-sqrt(2)*pi/period)`, `b1 = 2*a1*cos(sqrt(2)*pi/period)`, `c2 = b1`, `c3 = -a1*a1`, `c1 = 1 - c2 - c3`. Recurrence `y[t] = c1*(x[t]+x[t-1])/2 + c2*y[t-1] + c3*y[t-2]`. Transfer function `b = {c1/2, c1/2}`, `a = {1, -c2, -c3}`.
  - **Decycler (1-pole highpass complement, a trend):** `alpha = (cos(2*pi/period) + sin(2*pi/period) - 1) / cos(2*pi/period)`. Recurrence `y[t] = (alpha/2)*(x[t]+x[t-1]) + (1-alpha)*y[t-1]`. Transfer function `b = {alpha/2, alpha/2}`, `a = {1, -(1-alpha)}`.
  - **2-pole highpass (roofing stage):** `k = 0.707`, `alpha = (cos(k*2*pi/period) + sin(k*2*pi/period) - 1) / cos(k*2*pi/period)`, `g = 1 - alpha/2`. Recurrence `y[t] = g*g*(x[t]-2*x[t-1]+x[t-2]) + 2*(1-alpha)*y[t-1] - (1-alpha)^2*y[t-2]`. Transfer function `b = {g*g, -2*g*g, g*g}`, `a = {1, -2*(1-alpha), (1-alpha)^2}`.

---

## Task 1: SuperSmoother

**Files:**
- Create: `include/screamer/signal/ehlers.h` (shared coefficient helpers)
- Create: `include/screamer/super_smoother.h`
- Modify: `bindings/bindings_signal.cpp` (add include + registration)
- Create: `devtools/baselines/SuperSmoother.py`
- Create: `docs/functions_signal/SuperSmoother.md`
- Test: `tests/test_ehlers_filters.py`

**Interfaces:**
- Produces: `include/screamer/signal/ehlers.h` exposes, in `namespace screamer`:
  - `double ehlers_resolve_period(std::optional<double> period, std::optional<double> cutoff)` returns the period, throwing `std::invalid_argument` unless exactly one argument is provided and it is in range (`period >= 2`, `0 < cutoff < 1`).
  - `void supersmoother_coeffs(double period, std::vector<double>& b, std::vector<double>& a)`.
  - `void ehlers_highpass2_coeffs(double period, std::vector<double>& b, std::vector<double>& a)` (used by Task 3).
- Produces: `screamer::SuperSmoother`, constructor `SuperSmoother(std::optional<double> period, std::optional<double> cutoff)`, base `ScreamerBase`. Task 3 constructs one internally.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ehlers_filters.py`:

```python
import numpy as np
import pytest
from scipy.signal import lfilter

from screamer import SuperSmoother
from regime_helpers import assert_batch_equals_scalar


def _x(n, seed=0):
    return np.random.default_rng(seed).standard_normal(n)


def _supersmoother_ba(period):
    a1 = np.exp(-np.sqrt(2) * np.pi / period)
    b1 = 2 * a1 * np.cos(np.sqrt(2) * np.pi / period)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3
    b = [c1 / 2, c1 / 2]
    a = [1.0, -c2, -c3]
    return b, a


class TestSuperSmoother:
    @pytest.mark.parametrize("period", [10.0, 20.0, 33.0])
    def test_matches_reference(self, period):
        x = _x(500, seed=int(period))
        ours = np.asarray(SuperSmoother(period=period)(x))
        b, a = _supersmoother_ba(period)
        ref = lfilter(b, a, x)
        np.testing.assert_allclose(ours, ref, atol=1e-12)

    def test_period_equals_cutoff(self):
        x = _x(200, seed=1)
        by_period = np.asarray(SuperSmoother(period=20.0)(x))
        by_cutoff = np.asarray(SuperSmoother(cutoff=2.0 / 20.0)(x))
        np.testing.assert_allclose(by_period, by_cutoff, atol=1e-12)

    def test_requires_exactly_one_arg(self):
        with pytest.raises(Exception):
            SuperSmoother()
        with pytest.raises(Exception):
            SuperSmoother(period=20.0, cutoff=0.1)

    def test_batch_equals_stream(self):
        x = _x(300, seed=7)
        assert_batch_equals_scalar(lambda: SuperSmoother(period=15.0), x)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `poetry run pytest tests/test_ehlers_filters.py -q`
Expected: FAIL with `ImportError: cannot import name 'SuperSmoother'`.

- [ ] **Step 3: Write the coefficient helper header**

Create `include/screamer/signal/ehlers.h`:

```cpp
#ifndef SCREAMER_SIGNAL_EHLERS
#define SCREAMER_SIGNAL_EHLERS

#include <cmath>
#include <optional>
#include <stdexcept>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace screamer {

// Resolve the exactly-one-of {period, cutoff} argument pair to a period in
// samples. cutoff is a fraction of Nyquist in (0, 1); period = 2 / cutoff.
inline double ehlers_resolve_period(std::optional<double> period,
                                    std::optional<double> cutoff) {
    const int provided = (period.has_value() ? 1 : 0) + (cutoff.has_value() ? 1 : 0);
    if (provided != 1) {
        throw std::invalid_argument("Exactly one of period or cutoff must be provided.");
    }
    double p;
    if (period.has_value()) {
        p = period.value();
    } else {
        const double c = cutoff.value();
        if (!(c > 0.0 && c < 1.0)) {
            throw std::invalid_argument("cutoff must be in (0, 1).");
        }
        p = 2.0 / c;
    }
    if (!(p >= 2.0)) {
        throw std::invalid_argument("period must be at least 2 samples.");
    }
    return p;
}

// Ehlers 2-pole SuperSmoother lowpass transfer function.
inline void supersmoother_coeffs(double period, std::vector<double>& b,
                                 std::vector<double>& a) {
    const double a1 = std::exp(-std::sqrt(2.0) * M_PI / period);
    const double b1 = 2.0 * a1 * std::cos(std::sqrt(2.0) * M_PI / period);
    const double c2 = b1;
    const double c3 = -a1 * a1;
    const double c1 = 1.0 - c2 - c3;
    b = {c1 / 2.0, c1 / 2.0};
    a = {1.0, -c2, -c3};
}

// Ehlers 2-pole highpass transfer function (the roofing-filter highpass stage).
inline void ehlers_highpass2_coeffs(double period, std::vector<double>& b,
                                    std::vector<double>& a) {
    const double k = 0.707;
    const double w = k * 2.0 * M_PI / period;
    const double alpha = (std::cos(w) + std::sin(w) - 1.0) / std::cos(w);
    const double g = 1.0 - alpha / 2.0;
    const double om = 1.0 - alpha;
    b = {g * g, -2.0 * g * g, g * g};
    a = {1.0, -2.0 * om, om * om};
}

}  // namespace screamer

#endif  // SCREAMER_SIGNAL_EHLERS
```

- [ ] **Step 4: Write the SuperSmoother node**

Create `include/screamer/super_smoother.h`:

```cpp
#ifndef SCREAMER_SUPER_SMOOTHER_H
#define SCREAMER_SUPER_SMOOTHER_H

// SuperSmoother: Ehlers' 2-pole low-lag lowpass filter. A critically
// damped IIR designed to pass cycles longer than `period` samples with
// less lag and less ringing than a Butterworth of the same rolloff.
//
//   a1 = exp(-sqrt(2) pi / period)
//   b1 = 2 a1 cos(sqrt(2) pi / period)
//   y[t] = c1 (x[t] + x[t-1]) / 2 + b1 y[t-1] - a1^2 y[t-2]
//
// where c1 = 1 - b1 + a1^2. Reference: J. Ehlers, "Cycle Analytics for
// Traders". Give exactly one of `period` (samples) or `cutoff`
// (fraction of Nyquist in (0, 1)); period = 2 / cutoff.

#include <optional>
#include <vector>
#include "screamer/common/base.h"
#include "screamer/signal/ehlers.h"
#include "screamer/signal/signal.h"

namespace screamer {

class SuperSmoother : public ScreamerBase {
public:
    explicit SuperSmoother(std::optional<double> period = std::nullopt,
                           std::optional<double> cutoff = std::nullopt) {
        const double p = ehlers_resolve_period(period, cutoff);
        std::vector<double> b, a;
        supersmoother_coeffs(p, b, a);
        iir_.init(b, a);
    }

    void reset() override { iir_.reset(); }

    double process_scalar(double x) override { return iir_.process_scalar(x); }

    void process_array_no_stride(double* y, const double* x, size_t size) override {
        iir_.process_array_no_stride(y, x, size);
    }

    void process_array_stride(double* y, size_t dyi, const double* x,
                              size_t dxi, size_t size) override {
        iir_.process_array_stride(y, dyi, x, dxi, size);
    }

private:
    IIRFilter iir_;
};

}  // namespace screamer

#endif  // SCREAMER_SUPER_SMOOTHER_H
```

- [ ] **Step 5: Register the binding**

In `bindings/bindings_signal.cpp`, add the include near the other filter includes:

```cpp
#include "screamer/super_smoother.h"
```

and add this registration inside `init_bindings_signal`, after the `ButterBandstop` block:

```cpp
    // SuperSmoother: Ehlers 2-pole low-lag lowpass. Exactly one of
    // period (samples) or cutoff (fraction of Nyquist).
    py::class_<screamer::SuperSmoother, screamer::ScreamerBase>(m, "SuperSmoother")
        .def(py::init<std::optional<double>, std::optional<double>>(),
             py::arg("period") = py::none(), py::arg("cutoff") = py::none())
        .def("__call__", &screamer::SuperSmoother::operator(), py::arg("value"))
        .def("reset", &screamer::SuperSmoother::reset, "Reset to the initial state.");
```

Add `#include <optional>` at the top of the file if not already present.

- [ ] **Step 6: Build**

Run: `make build`
Expected: compiles, copies the `.so` into `screamer/`, regenerates `screamer/__init__.py` (now containing `SuperSmoother`).

- [ ] **Step 7: Run the test to verify it passes**

Run: `poetry run pytest tests/test_ehlers_filters.py::TestSuperSmoother -q`
Expected: PASS (4 tests).

- [ ] **Step 8: Add the baseline**

Create `devtools/baselines/SuperSmoother.py`:

```python
import numpy as np
from scipy.signal import lfilter


class SuperSmoother_ehlers:
    def __init__(self, period=None, cutoff=None):
        if (period is None) == (cutoff is None):
            raise ValueError("Provide exactly one of period or cutoff.")
        p = period if period is not None else 2.0 / cutoff
        a1 = np.exp(-np.sqrt(2) * np.pi / p)
        b1 = 2 * a1 * np.cos(np.sqrt(2) * np.pi / p)
        c2, c3 = b1, -a1 * a1
        c1 = 1 - c2 - c3
        self.b = [c1 / 2, c1 / 2]
        self.a = [1.0, -c2, -c3]

    def __call__(self, array):
        return lfilter(self.b, self.a, np.asarray(array, float))
```

- [ ] **Step 9: Write the docs page**

Create `docs/functions_signal/SuperSmoother.md`:

```markdown
---
name: SuperSmoother
title: Ehlers SuperSmoother lowpass filter
implementation_family: signal
topics:
- smoothing
- filtering
tags:
- ehlers
- iir
- lowpass
short: Ehlers 2-pole low-lag lowpass filter.
inputs: 1
outputs: 1
parameters:
- name: period
  type: float
  default: null
  description: Cutoff as a cycle length in samples. Give this or cutoff, not both.
- name: cutoff
  type: float
  default: null
  description: Cutoff as a fraction of Nyquist in (0, 1). Give this or period, not both.
nan_policy: ignore
---

# `SuperSmoother`

## Description

`SuperSmoother` is a causal 2-pole IIR lowpass filter from John Ehlers. It passes
components of the input series slower than the cutoff and attenuates faster ones,
with less lag and less ringing than a Butterworth filter of comparable rolloff.

The cutoff is set by a cycle length in samples (`period`) or, equivalently, a
fraction of the Nyquist frequency (`cutoff`), related by `cutoff = 2 / period`.
The filter coefficients follow Ehlers' critically damped design.

### Parameters

**`period`** *(float)*: The cutoff expressed as a cycle length in samples.

**`cutoff`** *(float)*: The cutoff expressed as a fraction of the Nyquist frequency, in (0, 1). Provide exactly one of `period` or `cutoff`.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in the input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
<!-- NAN_FOOTNOTE_END -->
```

- [ ] **Step 10: Regenerate the help registry and verify docs coverage**

Run:
```bash
poetry run python devtools/build_help_registry.py
poetry run python devtools/build_topic_pages.py
poetry run pytest tests/test_doc_coverage.py -q
```
Expected: PASS (SuperSmoother has a page and valid topics).

- [ ] **Step 11: Run clang-tidy**

Run: `make tidy`
Expected: clean (no member-init or other errors).

- [ ] **Step 12: Commit**

```bash
git add include/screamer/signal/ehlers.h include/screamer/super_smoother.h \
        bindings/bindings_signal.cpp devtools/baselines/SuperSmoother.py \
        docs/functions_signal/SuperSmoother.md tests/test_ehlers_filters.py
git commit -m "feat(signal): add Ehlers SuperSmoother lowpass filter"
```

---

## Task 2: Decycler

**Files:**
- Modify: `include/screamer/signal/ehlers.h` (add `decycler_coeffs`)
- Create: `include/screamer/decycler.h`
- Modify: `bindings/bindings_signal.cpp`
- Create: `devtools/baselines/Decycler.py`
- Create: `docs/functions_signal/Decycler.md`
- Test: `tests/test_ehlers_filters.py` (add `TestDecycler`)

**Interfaces:**
- Consumes: `ehlers_resolve_period` from Task 1.
- Produces: `screamer::Decycler`, constructor `Decycler(std::optional<double> period, std::optional<double> cutoff)`, base `ScreamerBase`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ehlers_filters.py`:

```python
from screamer import Decycler


def _decycler_ba(period):
    w = 2 * np.pi / period
    alpha = (np.cos(w) + np.sin(w) - 1) / np.cos(w)
    b = [alpha / 2, alpha / 2]
    a = [1.0, -(1 - alpha)]
    return b, a


class TestDecycler:
    @pytest.mark.parametrize("period", [30.0, 60.0, 125.0])
    def test_matches_reference(self, period):
        x = _x(500, seed=int(period))
        ours = np.asarray(Decycler(period=period)(x))
        b, a = _decycler_ba(period)
        ref = lfilter(b, a, x)
        np.testing.assert_allclose(ours, ref, atol=1e-12)

    def test_batch_equals_stream(self):
        x = _x(300, seed=9)
        assert_batch_equals_scalar(lambda: Decycler(period=40.0), x)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `poetry run pytest tests/test_ehlers_filters.py::TestDecycler -q`
Expected: FAIL with `ImportError: cannot import name 'Decycler'`.

- [ ] **Step 3: Add the coefficient helper**

In `include/screamer/signal/ehlers.h`, add before the closing namespace brace:

```cpp
// Ehlers 1-pole highpass complement (a trend estimate).
inline void decycler_coeffs(double period, std::vector<double>& b,
                            std::vector<double>& a) {
    const double w = 2.0 * M_PI / period;
    const double alpha = (std::cos(w) + std::sin(w) - 1.0) / std::cos(w);
    b = {alpha / 2.0, alpha / 2.0};
    a = {1.0, -(1.0 - alpha)};
}
```

- [ ] **Step 4: Write the Decycler node**

Create `include/screamer/decycler.h`:

```cpp
#ifndef SCREAMER_DECYCLER_H
#define SCREAMER_DECYCLER_H

// Decycler: Ehlers' trend estimate, the input with cycles at or below
// `period` removed. Equivalent to input minus a 1-pole highpass.
//
//   alpha = (cos(2 pi / period) + sin(2 pi / period) - 1) / cos(2 pi / period)
//   y[t] = (alpha / 2) (x[t] + x[t-1]) + (1 - alpha) y[t-1]
//
// Give exactly one of `period` (samples) or `cutoff` (fraction of
// Nyquist in (0, 1)); period = 2 / cutoff.

#include <optional>
#include <vector>
#include "screamer/common/base.h"
#include "screamer/signal/ehlers.h"
#include "screamer/signal/signal.h"

namespace screamer {

class Decycler : public ScreamerBase {
public:
    explicit Decycler(std::optional<double> period = std::nullopt,
                      std::optional<double> cutoff = std::nullopt) {
        const double p = ehlers_resolve_period(period, cutoff);
        std::vector<double> b, a;
        decycler_coeffs(p, b, a);
        iir_.init(b, a);
    }

    void reset() override { iir_.reset(); }

    double process_scalar(double x) override { return iir_.process_scalar(x); }

    void process_array_no_stride(double* y, const double* x, size_t size) override {
        iir_.process_array_no_stride(y, x, size);
    }

    void process_array_stride(double* y, size_t dyi, const double* x,
                              size_t dxi, size_t size) override {
        iir_.process_array_stride(y, dyi, x, dxi, size);
    }

private:
    IIRFilter iir_;
};

}  // namespace screamer

#endif  // SCREAMER_DECYCLER_H
```

- [ ] **Step 5: Register the binding**

In `bindings/bindings_signal.cpp` add `#include "screamer/decycler.h"` and, after the `SuperSmoother` block:

```cpp
    // Decycler: Ehlers trend estimate (input minus a 1-pole highpass).
    py::class_<screamer::Decycler, screamer::ScreamerBase>(m, "Decycler")
        .def(py::init<std::optional<double>, std::optional<double>>(),
             py::arg("period") = py::none(), py::arg("cutoff") = py::none())
        .def("__call__", &screamer::Decycler::operator(), py::arg("value"))
        .def("reset", &screamer::Decycler::reset, "Reset to the initial state.");
```

- [ ] **Step 6: Build**

Run: `make build`
Expected: compiles; `Decycler` appears in `screamer/__init__.py`.

- [ ] **Step 7: Run the test to verify it passes**

Run: `poetry run pytest tests/test_ehlers_filters.py::TestDecycler -q`
Expected: PASS.

- [ ] **Step 8: Add the baseline**

Create `devtools/baselines/Decycler.py`:

```python
import numpy as np
from scipy.signal import lfilter


class Decycler_ehlers:
    def __init__(self, period=None, cutoff=None):
        if (period is None) == (cutoff is None):
            raise ValueError("Provide exactly one of period or cutoff.")
        p = period if period is not None else 2.0 / cutoff
        w = 2 * np.pi / p
        alpha = (np.cos(w) + np.sin(w) - 1) / np.cos(w)
        self.b = [alpha / 2, alpha / 2]
        self.a = [1.0, -(1 - alpha)]

    def __call__(self, array):
        return lfilter(self.b, self.a, np.asarray(array, float))
```

- [ ] **Step 9: Write the docs page**

Create `docs/functions_signal/Decycler.md`:

```markdown
---
name: Decycler
title: Ehlers Decycler trend filter
implementation_family: signal
topics:
- smoothing
- filtering
tags:
- ehlers
- iir
- trend
short: Ehlers trend estimate with short cycles removed.
inputs: 1
outputs: 1
parameters:
- name: period
  type: float
  default: null
  description: Cycles at or below this length in samples are removed. Give this or cutoff.
- name: cutoff
  type: float
  default: null
  description: Cutoff as a fraction of Nyquist in (0, 1). Give this or period, not both.
nan_policy: ignore
---

# `Decycler`

## Description

`Decycler` is a causal filter from John Ehlers that estimates the trend of a
series by removing cycles at or below the cutoff. It is the input minus its
1-pole highpass component, so slow movements pass through and fast cycles are
suppressed.

The cutoff is set by a cycle length in samples (`period`) or a fraction of the
Nyquist frequency (`cutoff`), related by `cutoff = 2 / period`.

### Parameters

**`period`** *(float)*: The shortest cycle length in samples that is removed.

**`cutoff`** *(float)*: The cutoff as a fraction of the Nyquist frequency, in (0, 1). Provide exactly one of `period` or `cutoff`.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in the input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
<!-- NAN_FOOTNOTE_END -->
```

- [ ] **Step 10: Regenerate help registry and verify docs coverage**

Run:
```bash
poetry run python devtools/build_help_registry.py
poetry run python devtools/build_topic_pages.py
poetry run pytest tests/test_doc_coverage.py -q
```
Expected: PASS.

- [ ] **Step 11: Run clang-tidy**

Run: `make tidy`
Expected: clean.

- [ ] **Step 12: Commit**

```bash
git add include/screamer/signal/ehlers.h include/screamer/decycler.h \
        bindings/bindings_signal.cpp devtools/baselines/Decycler.py \
        docs/functions_signal/Decycler.md tests/test_ehlers_filters.py
git commit -m "feat(signal): add Ehlers Decycler trend filter"
```

---

## Task 3: RoofingFilter

**Files:**
- Create: `include/screamer/roofing_filter.h`
- Modify: `bindings/bindings_signal.cpp`
- Create: `devtools/baselines/RoofingFilter.py`
- Create: `docs/functions_signal/RoofingFilter.md`
- Test: `tests/test_ehlers_filters.py` (add `TestRoofingFilter`)

**Interfaces:**
- Consumes: `ehlers_resolve_period`, `supersmoother_coeffs`, `ehlers_highpass2_coeffs` from Task 1.
- Produces: `screamer::RoofingFilter`, constructor `RoofingFilter(std::optional<double> hp_period, std::optional<double> lp_period, std::optional<double> hp_cutoff, std::optional<double> lp_cutoff)`. Exactly one of `hp_period`/`hp_cutoff` and exactly one of `lp_period`/`lp_cutoff`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ehlers_filters.py`:

```python
from screamer import RoofingFilter


def _highpass2_ba(period):
    w = 0.707 * 2 * np.pi / period
    alpha = (np.cos(w) + np.sin(w) - 1) / np.cos(w)
    g = 1 - alpha / 2
    om = 1 - alpha
    b = [g * g, -2 * g * g, g * g]
    a = [1.0, -2 * om, om * om]
    return b, a


class TestRoofingFilter:
    @pytest.mark.parametrize("hp,lp", [(48.0, 10.0), (80.0, 20.0)])
    def test_matches_reference(self, hp, lp):
        x = _x(600, seed=int(hp + lp))
        ours = np.asarray(RoofingFilter(hp_period=hp, lp_period=lp)(x))
        b_hp, a_hp = _highpass2_ba(hp)
        b_ss, a_ss = _supersmoother_ba(lp)
        ref = lfilter(b_ss, a_ss, lfilter(b_hp, a_hp, x))
        np.testing.assert_allclose(ours, ref, atol=1e-12)

    def test_batch_equals_stream(self):
        x = _x(400, seed=11)
        assert_batch_equals_scalar(
            lambda: RoofingFilter(hp_period=48.0, lp_period=10.0), x)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `poetry run pytest tests/test_ehlers_filters.py::TestRoofingFilter -q`
Expected: FAIL with `ImportError: cannot import name 'RoofingFilter'`.

- [ ] **Step 3: Write the RoofingFilter node**

Create `include/screamer/roofing_filter.h`:

```cpp
#ifndef SCREAMER_ROOFING_FILTER_H
#define SCREAMER_ROOFING_FILTER_H

// RoofingFilter: Ehlers' bandpass preprocessor. A 2-pole highpass at
// `hp_period` removes the trend, then a SuperSmoother at `lp_period`
// removes aliasing noise, leaving the tradeable cycle band. Chains two
// IIR stages in one node. Reference: J. Ehlers, "Predictive and
// Successful Indicators". Give exactly one of hp_period / hp_cutoff and
// exactly one of lp_period / lp_cutoff.

#include <optional>
#include <vector>
#include "screamer/common/base.h"
#include "screamer/signal/ehlers.h"
#include "screamer/signal/signal.h"

namespace screamer {

class RoofingFilter : public ScreamerBase {
public:
    RoofingFilter(std::optional<double> hp_period = std::nullopt,
                  std::optional<double> lp_period = std::nullopt,
                  std::optional<double> hp_cutoff = std::nullopt,
                  std::optional<double> lp_cutoff = std::nullopt) {
        const double hp = ehlers_resolve_period(hp_period, hp_cutoff);
        const double lp = ehlers_resolve_period(lp_period, lp_cutoff);
        std::vector<double> b, a;
        ehlers_highpass2_coeffs(hp, b, a);
        hp_.init(b, a);
        supersmoother_coeffs(lp, b, a);
        lp_.init(b, a);
    }

    void reset() override {
        hp_.reset();
        lp_.reset();
    }

    double process_scalar(double x) override {
        return lp_.process_scalar(hp_.process_scalar(x));
    }

private:
    IIRFilter hp_;
    IIRFilter lp_;
};

}  // namespace screamer

#endif  // SCREAMER_ROOFING_FILTER_H
```

Note: this node relies on the `ScreamerBase` default array methods, which call
`process_scalar` per element, so the two-stage chain composes correctly in every
regime. Do not override the array methods here.

- [ ] **Step 4: Register the binding**

In `bindings/bindings_signal.cpp` add `#include "screamer/roofing_filter.h"` and, after the `Decycler` block:

```cpp
    // RoofingFilter: Ehlers bandpass (2-pole highpass then SuperSmoother).
    py::class_<screamer::RoofingFilter, screamer::ScreamerBase>(m, "RoofingFilter")
        .def(py::init<std::optional<double>, std::optional<double>,
                      std::optional<double>, std::optional<double>>(),
             py::arg("hp_period") = py::none(), py::arg("lp_period") = py::none(),
             py::arg("hp_cutoff") = py::none(), py::arg("lp_cutoff") = py::none())
        .def("__call__", &screamer::RoofingFilter::operator(), py::arg("value"))
        .def("reset", &screamer::RoofingFilter::reset, "Reset to the initial state.");
```

- [ ] **Step 5: Build**

Run: `make build`
Expected: compiles; `RoofingFilter` appears in `screamer/__init__.py`.

- [ ] **Step 6: Run the test to verify it passes**

Run: `poetry run pytest tests/test_ehlers_filters.py::TestRoofingFilter -q`
Expected: PASS.

- [ ] **Step 7: Add the baseline**

Create `devtools/baselines/RoofingFilter.py`:

```python
import numpy as np
from scipy.signal import lfilter


class RoofingFilter_ehlers:
    def __init__(self, hp_period=None, lp_period=None, hp_cutoff=None, lp_cutoff=None):
        hp = hp_period if hp_period is not None else 2.0 / hp_cutoff
        lp = lp_period if lp_period is not None else 2.0 / lp_cutoff
        w = 0.707 * 2 * np.pi / hp
        alpha = (np.cos(w) + np.sin(w) - 1) / np.cos(w)
        g, om = 1 - alpha / 2, 1 - alpha
        self.b_hp = [g * g, -2 * g * g, g * g]
        self.a_hp = [1.0, -2 * om, om * om]
        a1 = np.exp(-np.sqrt(2) * np.pi / lp)
        b1 = 2 * a1 * np.cos(np.sqrt(2) * np.pi / lp)
        c2, c3 = b1, -a1 * a1
        c1 = 1 - c2 - c3
        self.b_lp = [c1 / 2, c1 / 2]
        self.a_lp = [1.0, -c2, -c3]

    def __call__(self, array):
        x = np.asarray(array, float)
        return lfilter(self.b_lp, self.a_lp, lfilter(self.b_hp, self.a_hp, x))
```

- [ ] **Step 8: Write the docs page**

Create `docs/functions_signal/RoofingFilter.md`:

```markdown
---
name: RoofingFilter
title: Ehlers roofing filter (bandpass)
implementation_family: signal
topics:
- filtering
tags:
- ehlers
- iir
- bandpass
short: Ehlers bandpass, a highpass then a SuperSmoother.
inputs: 1
outputs: 1
parameters:
- name: hp_period
  type: float
  default: null
  description: Highpass cutoff as a cycle length in samples. Give this or hp_cutoff.
- name: lp_period
  type: float
  default: null
  description: Lowpass cutoff as a cycle length in samples. Give this or lp_cutoff.
- name: hp_cutoff
  type: float
  default: null
  description: Highpass cutoff as a fraction of Nyquist in (0, 1). Give this or hp_period.
- name: lp_cutoff
  type: float
  default: null
  description: Lowpass cutoff as a fraction of Nyquist in (0, 1). Give this or lp_period.
nan_policy: ignore
---

# `RoofingFilter`

## Description

`RoofingFilter` is a causal bandpass from John Ehlers. It applies a 2-pole
highpass at `hp_period` to remove the trend, then a `SuperSmoother` at
`lp_period` to remove fast noise, leaving the band of cycles between the two
cutoffs. It is the preprocessing step Ehlers applies before measuring a cycle,
because a raw series carries a trend and high-frequency noise that distort the
measurement.

Each cutoff is a cycle length in samples (`hp_period`, `lp_period`) or a fraction
of the Nyquist frequency (`hp_cutoff`, `lp_cutoff`), related by `cutoff = 2 / period`.

### Parameters

**`hp_period`** / **`hp_cutoff`** *(float)*: The highpass cutoff. Provide exactly one.

**`lp_period`** / **`lp_cutoff`** *(float)*: The lowpass cutoff. Provide exactly one.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in the input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
<!-- NAN_FOOTNOTE_END -->
```

- [ ] **Step 9: Regenerate help registry and verify docs coverage**

Run:
```bash
poetry run python devtools/build_help_registry.py
poetry run python devtools/build_topic_pages.py
poetry run pytest tests/test_doc_coverage.py -q
```
Expected: PASS.

- [ ] **Step 10: Run clang-tidy**

Run: `make tidy`
Expected: clean.

- [ ] **Step 11: Run the full filter test file and the suite**

Run:
```bash
poetry run pytest tests/test_ehlers_filters.py -q
poetry run pytest -q
```
Expected: the filter file passes in full; the whole suite is green with zero skips.

- [ ] **Step 12: Commit**

```bash
git add include/screamer/roofing_filter.h bindings/bindings_signal.cpp \
        devtools/baselines/RoofingFilter.py docs/functions_signal/RoofingFilter.md \
        tests/test_ehlers_filters.py
git commit -m "feat(signal): add Ehlers roofing filter (bandpass)"
```

---

## Notes for the next plan (cycle core)

- The cycle core builds on `RoofingFilter` and the coefficient helpers in
  `signal/ehlers.h`. It adds a new `cycles` topic to `docs/topics.yml` under the
  `indicators` group. Multi-output nodes (`HilbertPhasor`, `CycleSine`) use
  `FunctorBase<T, NIn, NOut>` (see `include/screamer/adx.h`, `macd.h`), not
  `ScreamerBase`.
- Verify the `graph`/`Pipeline` regime once here manually before relying on it in
  the cycle plan: `from screamer import Pipeline, Input` then wrap a
  `SuperSmoother` and confirm the output matches the eager call. The three filters
  inherit `ScreamerBase`, which the Pipeline engine already supports, so no extra
  work is expected.
```
