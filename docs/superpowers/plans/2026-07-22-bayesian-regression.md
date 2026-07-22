# BayesianRegression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `BayesianRegression`, a C++ online Bayesian univariate regression op that emits a causal one-step-ahead predictive mean and standard deviation plus the current slope and intercept, updating a Normal-Inverse-Gamma posterior recursively with exponential forgetting.

**Architecture:** A header-only `FunctorBase<BayesianRegression, 2, 4>` op modeled on `EwCov` (the `com/span/halflife/alpha` forgetting parameterization and 2-input NaN handling). It keeps a 2x2 precision matrix, a 2-vector, and two inverse-gamma scalars as state; each step it relaxes the posterior toward the prior (stabilized forgetting), forms the Student-t predictive from data before `t`, then folds in `(x_t, y_t)`. All 2x2 algebra is analytic, O(1) per step.

**Tech Stack:** C++17 header-only op (`include/screamer/`), pybind11 (`bindings/bindings_fin.cpp`), docs + help regeneration, pytest.

## Global Constraints

- Name `BayesianRegression`, called `(y, x)` (matching `RollingLinearRegression`'s argument order: `inputs[0]=y`, `inputs[1]=x`). `FunctorBase<BayesianRegression, 2, 4>`.
- Outputs, 4 columns in order: `[pred_mean, pred_std, slope, intercept]`. `pred_mean`/`pred_std` are the one-step-ahead predictive for `y_t` using only data before `t`; `slope`/`intercept` are the posterior means after folding in step `t`.
- Forgetting parameterized exactly like the `Ew*` family: exactly one of `com`, `span`, `halflife`, `alpha`, mapped to `alpha` the same way `EwCov` does, then the forgetting factor is `lambda = 1 - alpha`. Reject `alpha` outside `(0, 1)`.
- Prior parameters: `prior_precision` (default 1.0, coefficient shrinkage; must be > 0) and `prior_sigma` (default 1.0, prior noise scale; must be > 0). The inverse-gamma shape is fixed at `a0 = 2.0` (weak and proper, so the predictive variance is defined from step 1); `b0 = prior_sigma^2 * (a0 - 1)`. Prior coefficient mean is zero.
- Causal (predict-then-update), O(1) per step, batch == stream. `nan_policy: ignore`: a NaN `x` or `y` emits an all-NaN row and leaves state untouched. `reset()` restores the prior.
- After C++ changes: `make install-dev`, then `poetry run python devtools/build_help_registry.py`, `poetry run python devtools/build_topic_pages.py`, `make regen-init`. Docs built with `make docs`.
- No em-dashes in prose/comments/docstrings (ASCII hyphens). Commit as `simu.ai <claude@sitmo.com>` with:

      Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
      Claude-Session: https://claude.ai/code/session_018q4wFbrQaLrzUFc1H5NpJx

  Do not edit version files. Do not push.

## The algorithm (verified in numpy; the header below is the exact transcription)

State: precision `L = [[L11, L12], [L12, L22]]`, vector `eta = [eta1, eta2]`, scalars `s` and `a`. Prior: `L0 = prior_precision * I`, `eta0 = 0`, `s0 = 2*b0`, `a0 = 2`. Per non-NaN step, with `lambda = 1 - alpha`:
1. Stabilized forgetting (relax toward the prior): `L = lambda*L + (1-lambda)*L0`, same for `eta`, `s`, `a`.
2. Predictive from `phi = [1, x]` using the current (pre-update) state: `det = L11*L22 - L12*L12`; `m = Linv*eta`; `pred_mean = m1 + m2*x`; `b = 0.5*(s - eta^T Linv eta)`; `pred_var = (b/a)*(1 + phi^T Linv phi) * a/(a-1)`; `pred_std = sqrt(pred_var)`.
3. Update: `L11 += 1; L12 += x; L22 += x*x; eta1 += y; eta2 += x*y; s += y*y; a += 0.5`.
4. Reported model: recompute `m` from the updated state; `slope = m2`, `intercept = m1`.

Reference numbers from the numpy validation (used in the tests): with `alpha=1e-4` on `y = 1.5 - 0.8x + 0.5*noise`, the end slope is `-0.798` and intercept `1.507`; calibration is `0.680` within one `pred_std` and `0.953` within two; step-1 outputs are finite (`pred_std ~ 1.42`); with `alpha=0.02` a mid-series slope change from `-0.8` to `+2.0` is tracked (end slope `~2.03`).

---

## File Structure

- `include/screamer/bayesian_regression.h` (create) - the op.
- `bindings/bindings_fin.cpp` (modify) - register near `RollingLinearRegression`.
- `tests/test_bayesian_regression.py` (create) - all tests.
- `tests/param_cases.py` (modify) - exclude from the no-arg sweep (requires a forgetting arg, like `EwCov`).
- `docs/functions_fin/BayesianRegression.md` (create) - docs page.
- `CHANGELOG.md` (modify).

---

## Task 1: the `BayesianRegression` C++ op, binding, and core tests

**Files:**
- Create: `include/screamer/bayesian_regression.h`
- Modify: `bindings/bindings_fin.cpp`
- Test: `tests/test_bayesian_regression.py`

**Interfaces:**
- Produces: `screamer.BayesianRegression(com=None, span=None, halflife=None, alpha=None, prior_precision=1.0, prior_sigma=1.0)`, called `(y, x)`, returns an `(n, 4)` array `[pred_mean, pred_std, slope, intercept]`.

- [ ] **Step 1: Write the failing core tests**

```python
import numpy as np
import pytest
from screamer import BayesianRegression


def test_requires_exactly_one_forgetting_arg():
    with pytest.raises((ValueError, Exception)):
        BayesianRegression()                       # none
    with pytest.raises((ValueError, Exception)):
        BayesianRegression(span=20, alpha=0.1)     # two


def test_recovers_true_slope_and_intercept():
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.standard_normal(n)
    y = 1.5 - 0.8 * x + rng.standard_normal(n) * 0.5
    out = BayesianRegression(alpha=1e-4)(y, x)     # long memory
    pred_mean, pred_std, slope, intercept = out.T
    assert abs(slope[-1] - (-0.8)) < 0.05
    assert abs(intercept[-1] - 1.5) < 0.05


def test_defined_from_first_step_no_nan_warmup():
    out = BayesianRegression(alpha=0.1)(np.array([1.0, 2.0, 3.0]),
                                        np.array([0.5, -0.5, 1.0]))
    assert np.all(np.isfinite(out))                # no NaN warmup, unlike RollingLinearRegression
    assert (out[:, 1] > 0).all()                   # pred_std positive


def test_nan_input_emits_nan_row_state_untouched():
    x = np.array([0.5, np.nan, -0.5, 1.0])
    y = np.array([1.0, 5.0, 2.0, 0.5])
    out = BayesianRegression(alpha=0.1)(y, x)
    assert np.isnan(out[1]).all()                  # the NaN row is all NaN
    # the step after the NaN equals the same series with the NaN pair removed
    x2 = np.array([0.5, -0.5, 1.0]); y2 = np.array([1.0, 2.0, 0.5])
    out2 = BayesianRegression(alpha=0.1)(y2, x2)
    np.testing.assert_allclose(out[2], out2[1], rtol=1e-12, atol=1e-12)


def test_reset_restores_prior():
    op = BayesianRegression(alpha=0.1)
    rng = np.random.default_rng(1)
    x = rng.standard_normal(50); y = 2 * x + 1
    first = op(y, x)
    op.reset()
    again = op(y, x)
    np.testing.assert_allclose(first, again, rtol=1e-12, atol=1e-12)


def test_batch_equals_stream():
    rng = np.random.default_rng(2)
    n = 300
    x = rng.standard_normal(n); y = 0.7 * x - 0.3 + rng.standard_normal(n) * 0.4
    batch = BayesianRegression(alpha=0.05)(y, x)
    op = BayesianRegression(alpha=0.05)
    stream = np.array([op(float(y[i]), float(x[i])) for i in range(n)])
    np.testing.assert_allclose(np.nan_to_num(stream), np.nan_to_num(batch), rtol=1e-11)
```

- [ ] **Step 2: Run to verify failure**

Run: `poetry run python -m pytest tests/test_bayesian_regression.py -q`
Expected: FAIL (`cannot import name 'BayesianRegression'`).

- [ ] **Step 3: Create `include/screamer/bayesian_regression.h`**

```cpp
#ifndef SCREAMER_BAYESIAN_REGRESSION_H
#define SCREAMER_BAYESIAN_REGRESSION_H

// BayesianRegression: online Bayesian simple regression y = b0 + b1*x + eps,
// eps ~ N(0, sigma^2) with sigma^2 unknown. Maintains a conjugate
// Normal-Inverse-Gamma posterior over (intercept, slope, sigma^2) and updates it
// recursively with an exponential forgetting factor (stabilized forgetting: each
// step relaxes the posterior toward the prior by (1 - lambda), then folds in the new
// observation). Emits a causal one-step-ahead Student-t predictive for y_t from x_t
// using data before t, then updates. 2 -> 4:
//   [pred_mean, pred_std, slope, intercept].
// pred_mean / pred_std use data before t (the forecast); slope / intercept are the
// posterior means after folding in t (the current model). O(1) per step (2x2).
// nan_policy: ignore - a NaN x or y emits an all-NaN row and leaves state untouched.

#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <tuple>
#include "screamer/common/functor_base.h"
#include "screamer/common/float_info.h"

namespace screamer {

class BayesianRegression : public FunctorBase<BayesianRegression, 2, 4> {
public:
    BayesianRegression(
        std::optional<double> com = std::nullopt,
        std::optional<double> span = std::nullopt,
        std::optional<double> halflife = std::nullopt,
        std::optional<double> alpha = std::nullopt,
        double prior_precision = 1.0,
        double prior_sigma = 1.0)
        : prior_precision_(prior_precision), prior_sigma_(prior_sigma)
    {
        const int provided = (com.has_value() ? 1 : 0) + (span.has_value() ? 1 : 0)
                           + (halflife.has_value() ? 1 : 0) + (alpha.has_value() ? 1 : 0);
        if (provided != 1)
            throw std::invalid_argument(
                "Exactly one of com, span, halflife, or alpha must be provided");
        double a;
        if (alpha.has_value())        a = alpha.value();
        else if (com.has_value())     a = 1.0 / (1.0 + com.value());
        else if (span.has_value())    a = 2.0 / (span.value() + 1.0);
        else                          a = 1.0 - std::exp(-std::log(2.0) / halflife.value());
        if (a <= 0.0 || a >= 1.0)
            throw std::invalid_argument("Alpha must be between 0 and 1 (exclusive)");
        if (prior_precision_ <= 0.0)
            throw std::invalid_argument("prior_precision must be positive");
        if (prior_sigma_ <= 0.0)
            throw std::invalid_argument("prior_sigma must be positive");
        lambda_ = 1.0 - a;                             // forgetting factor
        a0_ = 2.0;                                     // weak proper IG shape (predictive defined from step 1)
        b0_ = prior_sigma_ * prior_sigma_ * (a0_ - 1.0);   // prior E[sigma^2] = prior_sigma^2
        L0_11_ = prior_precision_; L0_12_ = 0.0; L0_22_ = prior_precision_;   // Lambda0 = prior_precision * I
        eta0_1_ = 0.0; eta0_2_ = 0.0;                  // eta0 = Lambda0 * m0, m0 = 0
        s0_ = 2.0 * b0_;                               // s0 = 2 b0 + m0^T Lambda0 m0 = 2 b0
        reset();
    }

    void reset() override {
        L11_ = L0_11_; L12_ = L0_12_; L22_ = L0_22_;
        eta1_ = eta0_1_; eta2_ = eta0_2_;
        s_ = s0_; a_ = a0_;
    }

    ResultTuple call(const InputArray& inputs) override {
        const double y = inputs[0];
        const double x = inputs[1];
        const double nan = std::numeric_limits<double>::quiet_NaN();
        if (isnan2(y) || isnan2(x))
            return std::make_tuple(nan, nan, nan, nan);   // ignore: state untouched

        const double one_minus = 1.0 - lambda_;

        // 1) stabilized forgetting: relax the posterior toward the prior.
        L11_ = lambda_ * L11_ + one_minus * L0_11_;
        L12_ = lambda_ * L12_ + one_minus * L0_12_;
        L22_ = lambda_ * L22_ + one_minus * L0_22_;
        eta1_ = lambda_ * eta1_ + one_minus * eta0_1_;
        eta2_ = lambda_ * eta2_ + one_minus * eta0_2_;
        s_   = lambda_ * s_   + one_minus * s0_;
        a_   = lambda_ * a_   + one_minus * a0_;

        // 2) one-step-ahead predictive for y from phi = [1, x], using state before t.
        double det = L11_ * L22_ - L12_ * L12_;
        double m1 = ( L22_ * eta1_ - L12_ * eta2_) / det;   // intercept mean
        double m2 = (-L12_ * eta1_ + L11_ * eta2_) / det;   // slope mean
        double pred_mean = m1 + m2 * x;
        double etaLinv_eta = (L22_ * eta1_ * eta1_ - 2.0 * L12_ * eta1_ * eta2_
                              + L11_ * eta2_ * eta2_) / det;
        double b = 0.5 * (s_ - etaLinv_eta);
        double phiLinv_phi = (L22_ - 2.0 * L12_ * x + L11_ * x * x) / det;
        double pred_var = (b / a_) * (1.0 + phiLinv_phi) * a_ / (a_ - 1.0);
        double pred_std = std::sqrt(pred_var);

        // 3) update with (phi, y): Lambda += phi phi^T, eta += phi y, s += y^2, a += 1/2.
        L11_ += 1.0;
        L12_ += x;
        L22_ += x * x;
        eta1_ += y;
        eta2_ += x * y;
        s_   += y * y;
        a_   += 0.5;

        // 4) posterior mean after the update: the reported model.
        det = L11_ * L22_ - L12_ * L12_;
        double slope     = (-L12_ * eta1_ + L11_ * eta2_) / det;
        double intercept = ( L22_ * eta1_ - L12_ * eta2_) / det;
        return std::make_tuple(pred_mean, pred_std, slope, intercept);
    }

private:
    double prior_precision_, prior_sigma_;
    double lambda_{};
    double a0_{}, b0_{}, s0_{};
    double L0_11_{}, L0_12_{}, L0_22_{};
    double eta0_1_{}, eta0_2_{};
    double L11_{}, L12_{}, L22_{};
    double eta1_{}, eta2_{};
    double s_{}, a_{};
};

}  // namespace screamer

#endif  // SCREAMER_BAYESIAN_REGRESSION_H
```

- [ ] **Step 4: Register the binding** in `bindings/bindings_fin.cpp`

Add the include near the other fin includes:
```cpp
#include "screamer/bayesian_regression.h"
```
Add the class registration next to `RollingLinearRegression` (mirroring `EwCov`'s optional-arg init plus the two prior args):
```cpp
py::class_<screamer::BayesianRegression, screamer::EvalOp>(m, "BayesianRegression")
    .def(py::init<std::optional<double>, std::optional<double>, std::optional<double>,
                  std::optional<double>, double, double>(),
         py::arg("com") = std::nullopt, py::arg("span") = std::nullopt,
         py::arg("halflife") = std::nullopt, py::arg("alpha") = std::nullopt,
         py::arg("prior_precision") = 1.0, py::arg("prior_sigma") = 1.0)
    .def("__call__", &screamer::BayesianRegression::handle_input)
    .def("reset", &screamer::BayesianRegression::reset, "Reset to the prior.");
```
If `bindings_fin.cpp` does not already include `<optional>`, add it near the top.

- [ ] **Step 5: Build, regen, run**

Run: `make install-dev && make regen-init && poetry run python -m pytest tests/test_bayesian_regression.py -q`
Expected: PASS (all 6 core tests). `make regen-init` exports `screamer.BayesianRegression`.

- [ ] **Step 6: Commit**

```bash
git add include/screamer/bayesian_regression.h bindings/bindings_fin.cpp screamer/__init__.py tests/test_bayesian_regression.py
git commit -m "feat(stats): BayesianRegression online Bayesian regression with uncertainty"
```

---

## Task 2: statistical-correctness tests

**Files:**
- Modify: `tests/test_bayesian_regression.py`

**Interfaces:**
- Consumes: `BayesianRegression` from Task 1.

- [ ] **Step 1: Write the tests**

```python
def test_predictive_intervals_are_calibrated():
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.standard_normal(n)
    y = 1.5 - 0.8 * x + rng.standard_normal(n) * 0.5
    out = BayesianRegression(alpha=1e-4)(y, x)
    pm, ps = out[200:, 0], out[200:, 1]            # skip the early learning
    within1 = (np.abs(y[200:] - pm) <= ps).mean()
    within2 = (np.abs(y[200:] - pm) <= 2 * ps).mean()
    assert 0.63 < within1 < 0.73                    # ~68%
    assert 0.92 < within2 < 0.97                    # ~95%


def test_intervals_shrink_as_evidence_grows():
    rng = np.random.default_rng(3)
    n = 3000
    x = rng.standard_normal(n)
    y = 2.0 * x + rng.standard_normal(n) * 0.3
    out = BayesianRegression(alpha=1e-4)(y, x)     # long memory -> certainty grows
    early = out[10:60, 1].mean()
    late = out[-50:, 1].mean()
    assert late < early                             # predictive std tightens with data


def test_forgetting_tracks_a_slope_change():
    rng = np.random.default_rng(4)
    n = 4000; h = n // 2
    x = rng.standard_normal(n)
    y = np.empty(n)
    y[:h] = 1.5 - 0.8 * x[:h] + rng.standard_normal(h) * 0.5
    y[h:] = 1.5 + 2.0 * x[h:] + rng.standard_normal(n - h) * 0.5
    out = BayesianRegression(alpha=0.02)(y, x)     # finite memory
    assert abs(out[h - 50, 2] - (-0.8)) < 0.2       # tracked the first regime
    assert abs(out[-1, 2] - 2.0) < 0.2              # followed the change


def test_prior_regularizes_a_degenerate_sample():
    # near-constant x: OLS slope is ill-posed; the prior keeps it finite and shrunk
    x = np.full(40, 3.0) + 1e-9 * np.arange(40)
    y = np.arange(40, dtype=float)
    out = BayesianRegression(alpha=0.05, prior_precision=1.0)(y, x)
    assert np.all(np.isfinite(out))
    assert abs(out[-1, 2]) < 50.0                   # slope stays bounded, not a blow-up


def test_parity_with_rolling_linear_regression_in_the_limit():
    from screamer import RollingLinearRegression
    rng = np.random.default_rng(5)
    n = 6000
    x = rng.standard_normal(n)
    y = 1.5 - 0.8 * x + rng.standard_normal(n) * 0.5
    br = BayesianRegression(alpha=1e-4, prior_precision=1e-6)(y, x)   # long memory, weak prior
    rlr = np.asarray(RollingLinearRegression(2000)(y, x))            # OLS over a long window
    # both estimate the same stationary line; compare end slope/intercept loosely
    assert abs(br[-1, 2] - rlr[-1, 0]) < 0.05        # slope
    assert abs(br[-1, 3] - rlr[-1, 1]) < 0.05        # intercept
```

- [ ] **Step 2: Run to verify pass**

Run: `poetry run python -m pytest tests/test_bayesian_regression.py -q`
Expected: PASS (all core + statistical tests). If the parity tolerance is tight because the two weighting schemes differ, confirm both estimates are within tolerance of the true `(-0.8, 1.5)` and adjust the comparison to the true params (keep the assertion meaningful, not loosened to triviality).

- [ ] **Step 3: Commit**

```bash
git add tests/test_bayesian_regression.py
git commit -m "test(stats): BayesianRegression calibration, drift, prior, parity"
```

---

## Task 3: docs, help, param sweep, changelog, and full verification

**Files:**
- Create: `docs/functions_fin/BayesianRegression.md`
- Modify: `tests/param_cases.py`, `CHANGELOG.md`

- [ ] **Step 1: Exclude from the no-arg sweep** in `tests/param_cases.py`

`BayesianRegression` cannot be default-constructed (it requires exactly one forgetting arg), exactly like `EwCov`. Find how `EwCov` is handled in `param_cases.py` (an exclusion set or a provided-args entry) and give `BayesianRegression` the same treatment (for example a constructor-args entry `{"span": 20}` so the sweep can build it, or add it to the exclusion set if that is the pattern). Match whatever `EwCov` / the `Ew*` ops do.

- [ ] **Step 2: Create `docs/functions_fin/BayesianRegression.md`**

Front-matter matching `RollingLinearRegression.md` (same `topics:` value so it homes in the same group; check `docs/functions_fin/RollingLinearRegression.md` and copy its `topics`/`implementation_family` fields), with `name: BayesianRegression`, `title`, `short`, `inputs: 2`, `outputs: 4`, the parameters (`com`, `span`, `halflife`, `alpha`, `prior_precision`, `prior_sigma`), `nan_policy: ignore`. Body: describe the online Bayesian regression, the causal predict-then-update loop, the Student-t predictive, the forgetting factor, and the prior (defined from step 1). Include a `.. plotly::` example that fits a drifting synthetic line and plots `y`, `pred_mean`, and a `pred_mean +/- pred_std` band, showing the interval tighten as data accumulates. No em-dashes. Run `poetry run python devtools/build_help_registry.py` then `poetry run python devtools/build_topic_pages.py` to register it.

- [ ] **Step 3: Add the `[Unreleased] ### Added` changelog entry** in `CHANGELOG.md`:
```markdown
* `BayesianRegression`: online Bayesian univariate regression that emits a causal
  one-step-ahead predictive mean and standard deviation plus the current slope and
  intercept. Normal-Inverse-Gamma posterior (noise learned online, Student-t
  predictive) with exponential forgetting (`com`/`span`/`halflife`/`alpha`) and a weak
  prior, so estimates and intervals are defined from the first sample.
```

- [ ] **Step 4: Full regen + suite**

Run:
```bash
poetry run python devtools/build_help_registry.py
poetry run python devtools/build_topic_pages.py
make regen-init
poetry run python -m pytest -q
```
Expected: the registry validates the new page; the whole suite passes (the `BayesianRegression` tests plus the NaN-policy and param sweeps picking it up).

- [ ] **Step 5: Docs build**

Run: `make docs`
Expected: exit 0; the `BayesianRegression` page renders (plotly iframe) and is homed; no new orphan warnings.

- [ ] **Step 6: Commit**

```bash
git add docs/functions_fin/BayesianRegression.md tests/param_cases.py CHANGELOG.md screamer/data/help.json docs/by_group docs/by_group_index.rst
git commit -m "docs(stats): BayesianRegression page + changelog"
```

---

## Self-Review

**Spec coverage:**
- Online Bayesian univariate regression, NIG conjugate, online `sigma^2`, Student-t predictive: Task 1 header (the verified recursion). Covered.
- Exponential forgetting `com/span/halflife/alpha` like the `Ew*` family (`lambda = 1 - alpha`): Task 1 constructor. Covered.
- Weak proper prior, defined from step 1 (`a0 = 2`): Task 1 constructor + the `defined_from_first_step` and `intervals shrink`/`calibrated` tests (Tasks 1-2). Covered.
- Inputs `(y, x)`, outputs `[pred_mean, pred_std, slope, intercept]`, causal predict-then-update: Task 1 `call`. Covered.
- Contract (2->4, O(1), batch==stream, nan ignore, reset restores prior): Task 1 tests. Covered.
- Validation (recovery, calibration, drift, prior shrink, parity with `RollingLinearRegression`): Tasks 1-2. Covered.
- Docs/help/changelog: Task 3. Covered.
- Out of scope (multivariate, other Bayesian families): no task, correct per spec.

**Placeholder scan:** No TBD/TODO. The full header is given (transcribed from the numpy-verified recursion). Test constants come from the validated run. The two "match what `EwCov` does" steps (param_cases treatment, docs `topics` field) point at a concrete exemplar to copy rather than leaving a blank; the parity-tolerance note gives a concrete fallback (compare to the true params) with the meaningful-assertion guardrail.

**Type consistency:** `BayesianRegression(com, span, halflife, alpha, prior_precision, prior_sigma)` and the `(y, x)` input order and `[pred_mean, pred_std, slope, intercept]` output order are identical across the header, the binding, and every test. `lambda_ = 1 - alpha`, `a0_ = 2.0`, `b0_ = prior_sigma^2 (a0-1)` are consistent between the Global Constraints, the algorithm summary, and the header.

---

## Notes for the implementer

- The header is a direct transcription of a numpy recursion that was verified to recover the true coefficients, produce calibrated intervals (68/95 percent), stay finite from step 1, and track a mid-series slope change. Do not re-derive the math; transcribe it and let the tests confirm.
- Everything is 2x2 analytic; there is no matrix library and none is needed. `det > 0` and `b >= b0 > 0` are guaranteed by the proper prior, so `pred_std` is always real.
- Model the constructor and NaN handling on `include/screamer/ew_cov.h`; model the binding on the `EwCov` / `RollingLinearRegression` entries in `bindings/bindings_ew.cpp` / `bindings/bindings_fin.cpp`.
