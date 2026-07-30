# DMI Family Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the Wilder directional-movement operators (`PlusDI`, `MinusDI`, `PlusDM`, `MinusDM`, `DX`, `ADXR`) as first-class screamer operators, sharing one implementation with the existing `ADX`.

**Architecture:** The directional-movement math (TR, +DM, -DM, Wilder smoothing, +DI/-DI, DX, ADX) currently lives inline in `include/screamer/adx.h`. Task 1 extracts it into a reusable `include/screamer/detail/dmi_core.h` (a `DmiCore` class with one `update(high, low, close)` method returning all quantities), and refactors `ADX` to call it with byte-identical output. The new operators are thin `FunctorBase` nodes that hold a `DmiCore` and select one output. This satisfies the "one implementation per behavior" rule: the smoothing recurrence exists once.

**Tech Stack:** C++17, pybind11, CMake/scikit-build-core, pytest, TA-Lib (`talib`, baselines/tests only; version 0.6.8 is installed).

## Global Constraints

- All operator logic lives in the C++ core; the Python layer is the pybind11 binding only. No compute in Python. One implementation per behavior: the DMI smoothing lives only in `DmiCore`; `ADX` and the new operators call it.
- Every operator runs in all three regimes (eager arrays, graph `Pipeline`, lazy scalar loop) with identical output, and is causal. Batch output equals streaming output.
- `nan_policy: ignore`, matching `ADX`. A `NaN` in any of `high`/`low`/`close` yields `NaN` outputs and leaves running state unchanged.
- Multi-input operators use `FunctorBase<Derived, NInputs, NOutputs>` (see `include/screamer/adx.h`, `macd.h`). They bind against `screamer::EvalOp` (not `ScreamerBase`) with `.def("__call__", &Class::handle_input)`, and live in `bindings/bindings_rolling.cpp` and docs family `docs/functions_rolling/` (where `ADX` already lives).
- Input arity: `PlusDI`, `MinusDI`, `DX`, `ADXR` take `(high, low, close)` (3 inputs). `PlusDM`, `MinusDM` take `(high, low)` (2 inputs).
- Default `window_size = 14`, matching `ADX`. Validate `window_size >= 2`.
- After any C++ change run `make build` before testing (`poetry run pytest`). `make tidy` must be clean (value-initialize every member).
- Every operator ships a docs page `docs/functions_rolling/<Name>.md` with validated YAML frontmatter and topics from `docs/topics.yml`. Use topics `trend` for the DI/DX/ADXR operators and `momentum` where the operator is an oscillator input; `trend` alone is acceptable for all six. Both topics already exist. `window_size` has a concrete default (14), so the help-registry all-defaults instantiation is satisfied with no special handling.
- Docs prose follows CONTRIBUTING.md: lead with what the thing is, state the mechanism, unit-agnostic, no em dashes (ASCII hyphens only), no editorializing adjectives.
- Baselines in `devtools/baselines/<Name>.py` as class `<Name>_talib`, wrapping `talib`. Tests compare screamer against `talib` directly (like `test_signal.py` compares filters against scipy).
- **Verified TA-Lib facts (use these exact values):**
  - `ADXR[t] = (ADX[t] + ADX[t - (window_size - 1)]) / 2`.
  - `+DI`/`-DI`/`DX`/`ADX` first finite at sample index `window_size`; `ADX` (and thus `ADXR`) settles later per `adx.h`.
  - `talib.PLUS_DM(high, low, timeperiod=w)` and `talib.MINUS_DM(high, low, timeperiod=w)` take `(high, low)` only, no `close`. Their smoothed values equal `adx.h`'s sum-form smoothed `+DM`/`-DM` exactly from index `w` onward; `talib` additionally emits one value at index `w-1`. **Design decision:** screamer's DMI family uses a uniform first-valid index of `window_size` across all members (internally consistent, shares one core). The `PlusDM`/`MinusDM` parity tests therefore compare against `talib` on the aligned region (index `>= window_size`), where the values are identical.

---

## Task 1: Extract DmiCore and refactor ADX

**Files:**
- Create: `include/screamer/detail/dmi_core.h`
- Modify: `include/screamer/adx.h` (delegate to `DmiCore`)
- Test: `tests/test_dmi.py`

**Interfaces:**
- Produces: `screamer::detail::DmiCore`, constructor `DmiCore(int window_size)`, method `DmiResult update(double high, double low, double close)`, and `void reset()`. `DmiResult` is a struct `{double plus_di, minus_di, plus_dm, minus_dm, dx, adx;}` with `NaN` fields during warmup. Tasks 2-5 construct a `DmiCore` and read one field.

- [ ] **Step 1: Write the failing test (ADX output unchanged + matches talib)**

Create `tests/test_dmi.py`:

```python
import numpy as np
import pytest
import talib

from screamer import ADX
from tests.regime_helpers import assert_batch_equals_scalar


def _ohlc(n, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.standard_normal(n))
    high = close + np.abs(rng.standard_normal(n))
    low = close - np.abs(rng.standard_normal(n))
    return high, low, close


class TestADXUnchanged:
    @pytest.mark.parametrize("w", [14, 20])
    def test_adx_matches_talib_after_refactor(self, w):
        high, low, close = _ohlc(300, seed=w)
        pdi, mdi, adx = ADX(w)(high, low, close)
        ref_pdi = talib.PLUS_DI(high, low, close, timeperiod=w)
        ref_mdi = talib.MINUS_DI(high, low, close, timeperiod=w)
        ref_adx = talib.ADX(high, low, close, timeperiod=w)
        for ours, ref in [(pdi, ref_pdi), (mdi, ref_mdi), (adx, ref_adx)]:
            m = np.isfinite(np.asarray(ours)) & np.isfinite(ref)
            np.testing.assert_allclose(np.asarray(ours)[m], ref[m], atol=1e-8)

    def test_adx_batch_equals_stream(self):
        high, low, close = _ohlc(200, seed=1)
        assert_batch_equals_scalar(lambda: ADX(14), high, low, close)
```

- [ ] **Step 2: Run the test to verify it passes against current ADX**

Run: `poetry run pytest tests/test_dmi.py::TestADXUnchanged -q`
Expected: PASS (this pins current `ADX` behavior before the refactor; it is a characterization test, so it should pass now).

- [ ] **Step 3: Create `include/screamer/detail/dmi_core.h`**

Move the exact logic from `include/screamer/adx.h` into a reusable class. The body of `update` is `adx.h`'s current `call` body, returning the struct instead of a tuple:

```cpp
#ifndef SCREAMER_DETAIL_DMI_CORE_H
#define SCREAMER_DETAIL_DMI_CORE_H

// DmiCore: Wilder directional-movement engine (Wilder, 1978), bit-exact to
// TA-Lib. One update per (high, low, close) bar. All outputs NaN during
// warmup. Shared by ADX and the standalone DMI operators so the smoothing
// recurrence exists in exactly one place. See include/screamer/adx.h history
// for the timeline derivation.

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include "screamer/common/float_info.h"

namespace screamer {
namespace detail {

struct DmiResult {
    double plus_di = std::numeric_limits<double>::quiet_NaN();
    double minus_di = std::numeric_limits<double>::quiet_NaN();
    double plus_dm = std::numeric_limits<double>::quiet_NaN();
    double minus_dm = std::numeric_limits<double>::quiet_NaN();
    double dx = std::numeric_limits<double>::quiet_NaN();
    double adx = std::numeric_limits<double>::quiet_NaN();
};

class DmiCore {
public:
    explicit DmiCore(int window_size) : w_(window_size) {
        if (window_size < 2) {
            throw std::invalid_argument("Window size must be at least 2.");
        }
    }

    void reset() {
        sum_tr_ = sum_pdm_ = sum_mdm_ = 0.0;
        prev_tr_ = prev_pdm_ = prev_mdm_ = 0.0;
        sum_dx_ = 0.0;
        prev_adx_ = 0.0;
        prev_high_ = std::numeric_limits<double>::quiet_NaN();
        prev_low_ = std::numeric_limits<double>::quiet_NaN();
        prev_close_ = std::numeric_limits<double>::quiet_NaN();
        n_seen_ = 0;
    }

    DmiResult update(double high, double low, double close) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        DmiResult r;
        if (isnan2(high) || isnan2(low) || isnan2(close)) {
            return r;  // NaN policy "ignore": leave state, emit all NaN.
        }
        if (isnan2(prev_close_)) {
            prev_high_ = high;
            prev_low_ = low;
            prev_close_ = close;
            return r;
        }
        const double tr = std::max({high - low, std::abs(high - prev_close_),
                                    std::abs(low - prev_close_)});
        const double up = high - prev_high_;
        const double down = prev_low_ - low;
        const double pdm = (up > down && up > 0.0) ? up : 0.0;
        const double mdm = (down > up && down > 0.0) ? down : 0.0;
        prev_high_ = high;
        prev_low_ = low;
        prev_close_ = close;
        n_seen_++;
        if (n_seen_ < w_) {
            sum_tr_ += tr;
            sum_pdm_ += pdm;
            sum_mdm_ += mdm;
            return r;
        }
        const double wd = static_cast<double>(w_);
        if (n_seen_ == w_) {
            prev_tr_ = sum_tr_ * (wd - 1.0) / wd + tr;
            prev_pdm_ = sum_pdm_ * (wd - 1.0) / wd + pdm;
            prev_mdm_ = sum_mdm_ * (wd - 1.0) / wd + mdm;
        } else {
            prev_tr_ = prev_tr_ * (wd - 1.0) / wd + tr;
            prev_pdm_ = prev_pdm_ * (wd - 1.0) / wd + pdm;
            prev_mdm_ = prev_mdm_ * (wd - 1.0) / wd + mdm;
        }
        r.plus_dm = prev_pdm_;
        r.minus_dm = prev_mdm_;
        if (prev_tr_ <= 0.0) {
            return r;  // DM values valid, DI/DX undefined this bar.
        }
        const double plus_di = 100.0 * prev_pdm_ / prev_tr_;
        const double minus_di = 100.0 * prev_mdm_ / prev_tr_;
        r.plus_di = plus_di;
        r.minus_di = minus_di;
        const double sum_di = plus_di + minus_di;
        const double dx = (sum_di > 0.0)
                              ? 100.0 * std::abs(plus_di - minus_di) / sum_di
                              : 0.0;
        r.dx = dx;
        if (n_seen_ < 2 * w_ - 1) {
            sum_dx_ += dx;
            return r;
        }
        if (n_seen_ == 2 * w_ - 1) {
            sum_dx_ += dx;
            prev_adx_ = sum_dx_ / wd;
        } else {
            prev_adx_ = ((wd - 1.0) * prev_adx_ + dx) / wd;
        }
        r.adx = prev_adx_;
        return r;
    }

    int window() const { return w_; }

private:
    const int w_;
    double sum_tr_ = 0.0;
    double sum_pdm_ = 0.0;
    double sum_mdm_ = 0.0;
    double prev_tr_ = 0.0;
    double prev_pdm_ = 0.0;
    double prev_mdm_ = 0.0;
    double sum_dx_ = 0.0;
    double prev_adx_ = 0.0;
    double prev_high_ = std::numeric_limits<double>::quiet_NaN();
    double prev_low_ = std::numeric_limits<double>::quiet_NaN();
    double prev_close_ = std::numeric_limits<double>::quiet_NaN();
    int n_seen_ = 0;
};

}  // namespace detail
}  // namespace screamer

#endif  // SCREAMER_DETAIL_DMI_CORE_H
```

- [ ] **Step 4: Refactor `include/screamer/adx.h` to delegate**

Replace the private members and the body of `call` so `ADX` holds a `detail::DmiCore` and forwards. Keep the class name, base `FunctorBase<ADX, 3, 3>`, and the returned tuple `(plus_di, minus_di, adx)` identical:

```cpp
#include "screamer/common/functor_base.h"
#include "screamer/detail/dmi_core.h"

namespace screamer {

class ADX : public FunctorBase<ADX, 3, 3> {
public:
    explicit ADX(int window_size = 14) : core_(window_size) {}

    void reset() override { core_.reset(); }

    ResultTuple call(const InputArray& inputs) override {
        const detail::DmiResult r = core_.update(inputs[0], inputs[1], inputs[2]);
        return std::make_tuple(r.plus_di, r.minus_di, r.adx);
    }

private:
    detail::DmiCore core_;
};

}  // namespace screamer
```

Keep the file's existing top-of-file doc comment. Remove the now-duplicated math.

- [ ] **Step 5: Build**

Run: `make build`
Expected: compiles; `screamer/__init__.py` unchanged (no new public class yet).

- [ ] **Step 6: Run the characterization test and the ADX suite**

Run:
```bash
poetry run pytest tests/test_dmi.py::TestADXUnchanged -q
poetry run pytest -q -k "adx or ADX"
```
Expected: PASS. `ADX` output is byte-identical to before the refactor.

- [ ] **Step 7: Run clang-tidy**

Run: `make tidy`
Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add include/screamer/detail/dmi_core.h include/screamer/adx.h tests/test_dmi.py
git commit -m "refactor(dmi): extract DmiCore from ADX (no behavior change)"
```

---

## Task 2: PlusDI and MinusDI

**Files:**
- Create: `include/screamer/plus_di.h`, `include/screamer/minus_di.h`
- Modify: `bindings/bindings_rolling.cpp`
- Create: `devtools/baselines/PlusDI.py`, `devtools/baselines/MinusDI.py`
- Create: `docs/functions_rolling/PlusDI.md`, `docs/functions_rolling/MinusDI.md`
- Test: `tests/test_dmi.py` (add `TestPlusMinusDI`)

**Interfaces:**
- Consumes: `screamer::detail::DmiCore`, `DmiResult` from Task 1.
- Produces: `screamer::PlusDI`, `screamer::MinusDI`, each `FunctorBase<_, 3, 1>`, constructor `(int window_size = 14)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dmi.py`:

```python
from screamer import PlusDI, MinusDI


class TestPlusMinusDI:
    @pytest.mark.parametrize("w", [14, 20])
    def test_plus_di_matches_talib(self, w):
        high, low, close = _ohlc(300, seed=w + 1)
        ours = np.asarray(PlusDI(w)(high, low, close))
        ref = talib.PLUS_DI(high, low, close, timeperiod=w)
        m = np.isfinite(ours) & np.isfinite(ref)
        np.testing.assert_allclose(ours[m], ref[m], atol=1e-8)

    @pytest.mark.parametrize("w", [14, 20])
    def test_minus_di_matches_talib(self, w):
        high, low, close = _ohlc(300, seed=w + 2)
        ours = np.asarray(MinusDI(w)(high, low, close))
        ref = talib.MINUS_DI(high, low, close, timeperiod=w)
        m = np.isfinite(ours) & np.isfinite(ref)
        np.testing.assert_allclose(ours[m], ref[m], atol=1e-8)

    def test_di_batch_equals_stream(self):
        high, low, close = _ohlc(200, seed=3)
        assert_batch_equals_scalar(lambda: PlusDI(14), high, low, close)
        assert_batch_equals_scalar(lambda: MinusDI(14), high, low, close)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `poetry run pytest tests/test_dmi.py::TestPlusMinusDI -q`
Expected: FAIL with `ImportError: cannot import name 'PlusDI'`.

- [ ] **Step 3: Write the nodes**

Create `include/screamer/plus_di.h`:

```cpp
#ifndef SCREAMER_PLUS_DI_H
#define SCREAMER_PLUS_DI_H

// PlusDI: Wilder's +DI (positive directional indicator), 100 times the
// smoothed positive directional movement divided by the smoothed true range.
// Measures the strength of upward movement. Shares DmiCore with ADX.

#include "screamer/common/functor_base.h"
#include "screamer/detail/dmi_core.h"

namespace screamer {

class PlusDI : public FunctorBase<PlusDI, 3, 1> {
public:
    explicit PlusDI(int window_size = 14) : core_(window_size) {}
    void reset() override { core_.reset(); }
    ResultTuple call(const InputArray& inputs) override {
        return std::make_tuple(core_.update(inputs[0], inputs[1], inputs[2]).plus_di);
    }
private:
    detail::DmiCore core_;
};

}  // namespace screamer

#endif  // SCREAMER_PLUS_DI_H
```

Create `include/screamer/minus_di.h` (identical shape, reading `.minus_di`):

```cpp
#ifndef SCREAMER_MINUS_DI_H
#define SCREAMER_MINUS_DI_H

// MinusDI: Wilder's -DI (negative directional indicator), 100 times the
// smoothed negative directional movement divided by the smoothed true range.
// Measures the strength of downward movement. Shares DmiCore with ADX.

#include "screamer/common/functor_base.h"
#include "screamer/detail/dmi_core.h"

namespace screamer {

class MinusDI : public FunctorBase<MinusDI, 3, 1> {
public:
    explicit MinusDI(int window_size = 14) : core_(window_size) {}
    void reset() override { core_.reset(); }
    ResultTuple call(const InputArray& inputs) override {
        return std::make_tuple(core_.update(inputs[0], inputs[1], inputs[2]).minus_di);
    }
private:
    detail::DmiCore core_;
};

}  // namespace screamer

#endif  // SCREAMER_MINUS_DI_H
```

Note: confirm `FunctorBase`'s single-output `ResultTuple` and `std::make_tuple(x)` usage against an existing 1-output functor. If `FunctorBase<_, _, 1>` expects a bare `double` return rather than a 1-tuple, return the `double` directly. Check `include/screamer/` for an existing `FunctorBase<_, N, 1>` example before writing; match its return convention exactly. If none exists, inspect `functor_base.h` to determine the 1-output return type and use it. Report the convention you found in your report.

- [ ] **Step 4: Register the bindings**

In `bindings/bindings_rolling.cpp`, near the `ADX` registration, add includes for `plus_di.h` and `minus_di.h`, and register (matching the `ADX` pattern: base `EvalOp`, `handle_input`):

```cpp
    py::class_<screamer::PlusDI, screamer::EvalOp>(m, "PlusDI")
        .def(py::init<int>(), py::arg("window_size") = 14)
        .def("__call__", &screamer::PlusDI::handle_input)
        .def("reset", &screamer::PlusDI::reset, "Reset to the initial state.");

    py::class_<screamer::MinusDI, screamer::EvalOp>(m, "MinusDI")
        .def(py::init<int>(), py::arg("window_size") = 14)
        .def("__call__", &screamer::MinusDI::handle_input)
        .def("reset", &screamer::MinusDI::reset, "Reset to the initial state.");
```

Copy the exact registration form (base class, `handle_input` vs `operator()`) from the adjacent `ADX` block; if it differs from the above, follow `ADX`.

- [ ] **Step 5: Build and test**

Run:
```bash
make build
poetry run pytest tests/test_dmi.py::TestPlusMinusDI -q
```
Expected: PASS.

- [ ] **Step 6: Add baselines**

Create `devtools/baselines/PlusDI.py`:

```python
import talib


class PlusDI_talib:
    def __init__(self, window_size=14):
        self.w = window_size

    def __call__(self, high, low, close):
        return talib.PLUS_DI(high, low, close, timeperiod=self.w)
```

Create `devtools/baselines/MinusDI.py` (identical, calling `talib.MINUS_DI`).

- [ ] **Step 7: Write docs pages**

Create `docs/functions_rolling/PlusDI.md`:

```markdown
---
name: PlusDI
title: Plus directional indicator (+DI)
implementation_family: rolling
topics:
- trend
tags:
- wilder
- dmi
- directional-movement
short: Wilder positive directional indicator.
inputs: 3
outputs: 1
parameters:
- name: window_size
  type: int
  default: 14
  min: 2
  description: Wilder smoothing period.
nan_policy: ignore
---

# `PlusDI`

## Description

`PlusDI` is Wilder's positive directional indicator, `+DI`. It is 100 times the
Wilder-smoothed positive directional movement divided by the smoothed true range,
so it measures the share of recent range attributable to upward movement. It
takes the high, low, and close series and shares its smoothing engine with `ADX`.

### Parameters

**`window_size`** *(int)*: The Wilder smoothing period.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
<!-- NAN_FOOTNOTE_END -->
```

Create `docs/functions_rolling/MinusDI.md` (same shape; title "Minus directional indicator (-DI)"; short "Wilder negative directional indicator."; describe downward movement).

- [ ] **Step 8: Regenerate help registry and verify docs coverage**

Run:
```bash
poetry run python devtools/build_help_registry.py
poetry run python devtools/build_topic_pages.py
poetry run pytest tests/test_doc_coverage.py -q
```
Expected: PASS.

- [ ] **Step 9: clang-tidy and commit**

Run: `make tidy` (expect clean), then:

```bash
git add include/screamer/plus_di.h include/screamer/minus_di.h \
        bindings/bindings_rolling.cpp devtools/baselines/PlusDI.py \
        devtools/baselines/MinusDI.py docs/functions_rolling/PlusDI.md \
        docs/functions_rolling/MinusDI.md tests/test_dmi.py
git commit -m "feat(dmi): add PlusDI and MinusDI operators"
```

---

## Task 3: PlusDM and MinusDM

**Files:**
- Create: `include/screamer/plus_dm.h`, `include/screamer/minus_dm.h`
- Modify: `bindings/bindings_rolling.cpp`
- Create: `devtools/baselines/PlusDM.py`, `devtools/baselines/MinusDM.py`
- Create: `docs/functions_rolling/PlusDM.md`, `docs/functions_rolling/MinusDM.md`
- Test: `tests/test_dmi.py` (add `TestPlusMinusDM`)

**Interfaces:**
- Consumes: `screamer::detail::DmiCore` from Task 1.
- Produces: `screamer::PlusDM`, `screamer::MinusDM`, each `FunctorBase<_, 2, 1>`, constructor `(int window_size = 14)`. They take `(high, low)` and pass a `NaN` close into `DmiCore::update` is wrong; instead call `update(high, low, low)` is also wrong. Read the note in Step 3.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dmi.py`:

```python
from screamer import PlusDM, MinusDM


class TestPlusMinusDM:
    @pytest.mark.parametrize("w", [14, 20])
    def test_plus_dm_matches_talib_aligned(self, w):
        high, low, close = _ohlc(300, seed=w + 4)
        ours = np.asarray(PlusDM(w)(high, low))
        ref = talib.PLUS_DM(high, low, timeperiod=w)
        # screamer uses a uniform first-valid index of window_size across the
        # DMI family; talib emits PLUS_DM one bar earlier. Compare where both
        # are finite (index >= window_size), where the values are identical.
        m = np.isfinite(ours) & np.isfinite(ref)
        assert m.sum() > 0
        np.testing.assert_allclose(ours[m], ref[m], atol=1e-8)

    @pytest.mark.parametrize("w", [14, 20])
    def test_minus_dm_matches_talib_aligned(self, w):
        high, low, close = _ohlc(300, seed=w + 5)
        ours = np.asarray(MinusDM(w)(high, low))
        ref = talib.MINUS_DM(high, low, timeperiod=w)
        m = np.isfinite(ours) & np.isfinite(ref)
        assert m.sum() > 0
        np.testing.assert_allclose(ours[m], ref[m], atol=1e-8)

    def test_dm_batch_equals_stream(self):
        high, low, close = _ohlc(200, seed=6)
        assert_batch_equals_scalar(lambda: PlusDM(14), high, low)
        assert_batch_equals_scalar(lambda: MinusDM(14), high, low)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `poetry run pytest tests/test_dmi.py::TestPlusMinusDM -q`
Expected: FAIL with `ImportError: cannot import name 'PlusDM'`.

- [ ] **Step 3: Write the nodes**

`DmiCore::update` requires a `close` (for TR), but `+DM`/`-DM` depend only on `high` and `low`. The smoothed `+DM`/`-DM` fields of `DmiResult` do not use `close` or `prev_tr_`, so a `PlusDM` node feeds `close = high` (any finite value keeps the not-NaN gate open without affecting `plus_dm`/`minus_dm`). Confirm by reading `DmiCore::update`: `plus_dm`/`minus_dm` are set before the `prev_tr_` branch and never read `close` except through the initial `isnan2(close)` gate and the `prev_close_` first-bar gate. Feeding `close = high` is finite and does not affect `+DM`/`-DM`.

Create `include/screamer/plus_dm.h`:

```cpp
#ifndef SCREAMER_PLUS_DM_H
#define SCREAMER_PLUS_DM_H

// PlusDM: Wilder's smoothed positive directional movement, +DM. The
// Wilder-smoothed run of upward moves (up = high - prev_high, counted when it
// exceeds the down move). Takes high and low. Shares DmiCore with ADX; +DM
// does not depend on close.

#include "screamer/common/functor_base.h"
#include "screamer/detail/dmi_core.h"

namespace screamer {

class PlusDM : public FunctorBase<PlusDM, 2, 1> {
public:
    explicit PlusDM(int window_size = 14) : core_(window_size) {}
    void reset() override { core_.reset(); }
    ResultTuple call(const InputArray& inputs) override {
        const double high = inputs[0];
        const double low = inputs[1];
        return std::make_tuple(core_.update(high, low, high).plus_dm);
    }
private:
    detail::DmiCore core_;
};

}  // namespace screamer

#endif  // SCREAMER_PLUS_DM_H
```

Create `include/screamer/minus_dm.h` (identical shape, reading `.minus_dm`, doc says negative directional movement `down = prev_low - low`).

Match the 1-output `FunctorBase` return convention you confirmed in Task 2.

- [ ] **Step 4: Register the bindings**

In `bindings/bindings_rolling.cpp` add includes and, matching the `ADX`/`PlusDI` pattern (base `EvalOp`, 2-input `handle_input`):

```cpp
    py::class_<screamer::PlusDM, screamer::EvalOp>(m, "PlusDM")
        .def(py::init<int>(), py::arg("window_size") = 14)
        .def("__call__", &screamer::PlusDM::handle_input)
        .def("reset", &screamer::PlusDM::reset, "Reset to the initial state.");

    py::class_<screamer::MinusDM, screamer::EvalOp>(m, "MinusDM")
        .def(py::init<int>(), py::arg("window_size") = 14)
        .def("__call__", &screamer::MinusDM::handle_input)
        .def("reset", &screamer::MinusDM::reset, "Reset to the initial state.");
```

- [ ] **Step 5: Build and test**

Run:
```bash
make build
poetry run pytest tests/test_dmi.py::TestPlusMinusDM -q
```
Expected: PASS.

- [ ] **Step 6: Add baselines**

Create `devtools/baselines/PlusDM.py`:

```python
import talib


class PlusDM_talib:
    def __init__(self, window_size=14):
        self.w = window_size

    def __call__(self, high, low):
        return talib.PLUS_DM(high, low, timeperiod=self.w)
```

Create `devtools/baselines/MinusDM.py` (identical, calling `talib.MINUS_DM`). Note: if `tests/test_baselines.py` compares full arrays and fails on the one-bar warmup difference, that is the documented uniform-warmup decision; record it in your report so the controller can decide whether to exclude these two from the array-level baseline comparison. The authoritative parity gate is the aligned comparison in `TestPlusMinusDM`.

- [ ] **Step 7: Write docs pages**

Create `docs/functions_rolling/PlusDM.md` and `MinusDM.md`, `inputs: 2`, topic `trend`, `nan_policy: ignore`, `window_size` default 14. Describe `+DM`/`-DM` as the Wilder-smoothed directional movement, and state that the value depends on high and low only. Mention the uniform first-valid index of `window_size`.

- [ ] **Step 8: Regenerate help registry and verify docs coverage**

Run:
```bash
poetry run python devtools/build_help_registry.py
poetry run python devtools/build_topic_pages.py
poetry run pytest tests/test_doc_coverage.py -q
```
Expected: PASS.

- [ ] **Step 9: clang-tidy and commit**

Run: `make tidy` (expect clean), then:

```bash
git add include/screamer/plus_dm.h include/screamer/minus_dm.h \
        bindings/bindings_rolling.cpp devtools/baselines/PlusDM.py \
        devtools/baselines/MinusDM.py docs/functions_rolling/PlusDM.md \
        docs/functions_rolling/MinusDM.md tests/test_dmi.py
git commit -m "feat(dmi): add PlusDM and MinusDM operators"
```

---

## Task 4: DX

**Files:**
- Create: `include/screamer/dx.h`
- Modify: `bindings/bindings_rolling.cpp`
- Create: `devtools/baselines/DX.py`
- Create: `docs/functions_rolling/DX.md`
- Test: `tests/test_dmi.py` (add `TestDX`)

**Interfaces:**
- Consumes: `screamer::detail::DmiCore` from Task 1.
- Produces: `screamer::DX`, `FunctorBase<DX, 3, 1>`, constructor `(int window_size = 14)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dmi.py`:

```python
from screamer import DX


class TestDX:
    @pytest.mark.parametrize("w", [14, 20])
    def test_dx_matches_talib(self, w):
        high, low, close = _ohlc(300, seed=w + 7)
        ours = np.asarray(DX(w)(high, low, close))
        ref = talib.DX(high, low, close, timeperiod=w)
        m = np.isfinite(ours) & np.isfinite(ref)
        np.testing.assert_allclose(ours[m], ref[m], atol=1e-8)

    def test_dx_batch_equals_stream(self):
        high, low, close = _ohlc(200, seed=8)
        assert_batch_equals_scalar(lambda: DX(14), high, low, close)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `poetry run pytest tests/test_dmi.py::TestDX -q`
Expected: FAIL with `ImportError: cannot import name 'DX'`.

- [ ] **Step 3: Write the node**

Create `include/screamer/dx.h`:

```cpp
#ifndef SCREAMER_DX_H
#define SCREAMER_DX_H

// DX: Wilder's directional index, 100 times the absolute difference of +DI and
// -DI over their sum. The pre-average input to ADX. Shares DmiCore with ADX.

#include "screamer/common/functor_base.h"
#include "screamer/detail/dmi_core.h"

namespace screamer {

class DX : public FunctorBase<DX, 3, 1> {
public:
    explicit DX(int window_size = 14) : core_(window_size) {}
    void reset() override { core_.reset(); }
    ResultTuple call(const InputArray& inputs) override {
        return std::make_tuple(core_.update(inputs[0], inputs[1], inputs[2]).dx);
    }
private:
    detail::DmiCore core_;
};

}  // namespace screamer

#endif  // SCREAMER_DX_H
```

Match the confirmed 1-output return convention.

- [ ] **Step 4: Register the binding**

Add include and registration in `bindings/bindings_rolling.cpp`:

```cpp
    py::class_<screamer::DX, screamer::EvalOp>(m, "DX")
        .def(py::init<int>(), py::arg("window_size") = 14)
        .def("__call__", &screamer::DX::handle_input)
        .def("reset", &screamer::DX::reset, "Reset to the initial state.");
```

- [ ] **Step 5: Build and test**

Run:
```bash
make build
poetry run pytest tests/test_dmi.py::TestDX -q
```
Expected: PASS.

- [ ] **Step 6: Baseline, docs, coverage**

Create `devtools/baselines/DX.py` (`DX_talib`, `__call__(self, high, low, close)` -> `talib.DX(...)`). Create `docs/functions_rolling/DX.md` (`inputs: 3`, topic `trend`, `nan_policy: ignore`, default 14). Then:
```bash
poetry run python devtools/build_help_registry.py
poetry run python devtools/build_topic_pages.py
poetry run pytest tests/test_doc_coverage.py -q
```
Expected: PASS.

- [ ] **Step 7: clang-tidy and commit**

Run: `make tidy` (expect clean), then:

```bash
git add include/screamer/dx.h bindings/bindings_rolling.cpp \
        devtools/baselines/DX.py docs/functions_rolling/DX.md tests/test_dmi.py
git commit -m "feat(dmi): add DX operator"
```

---

## Task 5: ADXR

**Files:**
- Create: `include/screamer/adxr.h`
- Modify: `bindings/bindings_rolling.cpp`
- Create: `devtools/baselines/ADXR.py`
- Create: `docs/functions_rolling/ADXR.md`
- Test: `tests/test_dmi.py` (add `TestADXR`)

**Interfaces:**
- Consumes: `screamer::detail::DmiCore` from Task 1.
- Produces: `screamer::ADXR`, `FunctorBase<ADXR, 3, 1>`, constructor `(int window_size = 14)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dmi.py`:

```python
from screamer import ADXR


class TestADXR:
    @pytest.mark.parametrize("w", [14, 20])
    def test_adxr_matches_talib(self, w):
        high, low, close = _ohlc(400, seed=w + 9)
        ours = np.asarray(ADXR(w)(high, low, close))
        ref = talib.ADXR(high, low, close, timeperiod=w)
        m = np.isfinite(ours) & np.isfinite(ref)
        assert m.sum() > 0
        np.testing.assert_allclose(ours[m], ref[m], atol=1e-8)

    def test_adxr_batch_equals_stream(self):
        high, low, close = _ohlc(300, seed=10)
        assert_batch_equals_scalar(lambda: ADXR(14), high, low, close)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `poetry run pytest tests/test_dmi.py::TestADXR -q`
Expected: FAIL with `ImportError: cannot import name 'ADXR'`.

- [ ] **Step 3: Write the node**

`ADXR[t] = (ADX[t] + ADX[t - (window_size - 1)]) / 2`. The node holds a `DmiCore` and a ring buffer of the last `window_size` finite `adx` values. It emits once it has an `adx` now and an `adx` from `window_size - 1` steps earlier. Because `DmiCore` returns `NaN` adx during its warmup, buffer only finite adx values and index back by `window_size - 1` finite steps? No: talib indexes by raw bars. Buffer adx (including NaN) by bar and index back `window_size - 1` bars.

Create `include/screamer/adxr.h`:

```cpp
#ifndef SCREAMER_ADXR_H
#define SCREAMER_ADXR_H

// ADXR: Wilder's average directional index rating, the mean of ADX now and ADX
// (window_size - 1) bars ago. Smooths ADX to gauge trend-strength momentum.
// Shares DmiCore with ADX.

#include <limits>
#include <vector>
#include "screamer/common/functor_base.h"
#include "screamer/detail/dmi_core.h"

namespace screamer {

class ADXR : public FunctorBase<ADXR, 3, 1> {
public:
    explicit ADXR(int window_size = 14)
        : core_(window_size), lag_(window_size - 1),
          buf_(static_cast<size_t>(window_size), std::numeric_limits<double>::quiet_NaN()) {}

    void reset() override {
        core_.reset();
        std::fill(buf_.begin(), buf_.end(), std::numeric_limits<double>::quiet_NaN());
        pos_ = 0;
    }

    ResultTuple call(const InputArray& inputs) override {
        const double adx = core_.update(inputs[0], inputs[1], inputs[2]).adx;
        // adx_lag is the adx value from `lag_` bars ago (the slot we are about
        // to overwrite holds the value `buf_.size()` bars ago; index the one
        // `lag_` bars back explicitly).
        const size_t back = static_cast<size_t>(lag_);
        const size_t idx = (pos_ + buf_.size() - back) % buf_.size();
        const double adx_lag = buf_[idx];
        buf_[pos_] = adx;
        pos_ = (pos_ + 1) % buf_.size();
        const double nan = std::numeric_limits<double>::quiet_NaN();
        if (std::isnan(adx) || std::isnan(adx_lag)) {
            return std::make_tuple(nan);
        }
        return std::make_tuple((adx + adx_lag) / 2.0);
    }

private:
    detail::DmiCore core_;
    const int lag_;
    std::vector<double> buf_;
    size_t pos_ = 0;
};

}  // namespace screamer

#endif  // SCREAMER_ADXR_H
```

Note on the ring buffer: `buf_` has `window_size` slots, so the value `lag_ = window_size - 1` bars ago is retrievable before being overwritten. Verify the index arithmetic against the `TestADXR` talib parity test; if off by one, adjust `back`/`buf_` size and re-run. Match the confirmed 1-output return convention.

- [ ] **Step 4: Register the binding**

Add include and registration in `bindings/bindings_rolling.cpp`:

```cpp
    py::class_<screamer::ADXR, screamer::EvalOp>(m, "ADXR")
        .def(py::init<int>(), py::arg("window_size") = 14)
        .def("__call__", &screamer::ADXR::handle_input)
        .def("reset", &screamer::ADXR::reset, "Reset to the initial state.");
```

- [ ] **Step 5: Build and test**

Run:
```bash
make build
poetry run pytest tests/test_dmi.py::TestADXR -q
```
Expected: PASS. If the parity test fails on alignment, the ring-buffer offset is the thing to adjust (the value math is `(adx + adx_lag)/2`); the `talib.ADXR` array is the oracle.

- [ ] **Step 6: Baseline, docs, coverage**

Create `devtools/baselines/ADXR.py` (`ADXR_talib`, `__call__(self, high, low, close)` -> `talib.ADXR(...)`). Create `docs/functions_rolling/ADXR.md` (`inputs: 3`, topic `trend`, `nan_policy: ignore`, default 14; state the `(ADX[t] + ADX[t-(window-1)])/2` definition). Then:
```bash
poetry run python devtools/build_help_registry.py
poetry run python devtools/build_topic_pages.py
poetry run pytest tests/test_doc_coverage.py -q
```
Expected: PASS.

- [ ] **Step 7: clang-tidy, full suite, commit**

Run:
```bash
make tidy
poetry run pytest -q
```
Expected: tidy clean; full suite green with zero skips.

```bash
git add include/screamer/adxr.h bindings/bindings_rolling.cpp \
        devtools/baselines/ADXR.py docs/functions_rolling/ADXR.md tests/test_dmi.py
git commit -m "feat(dmi): add ADXR operator"
```

---

## Notes

- The one design decision to surface to the user: screamer's `PlusDM`/`MinusDM` use a uniform first-valid index of `window_size`, one bar later than `talib.PLUS_DM`/`MINUS_DM`. The values are identical from `window_size` onward. This keeps the whole DMI family consistent and sharing one core. If strict `talib` index-parity is preferred, the DM path would need to emit one bar earlier.
- `FunctorBase` 1-output return convention (bare `double` vs 1-tuple) is confirmed in Task 2 and reused in Tasks 3-5. If it turns out to be a bare `double`, drop the `std::make_tuple(...)` wrapper in every node here.
