# Research Toolkit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the gaps the first screamer-bots research loop hit and add probabilistic regime models: a clearer `permute_bars`/`permute_returns` error, `Hold(cooldown=)`, ten new operators (`CusumFilter`, `RollingUpsideDeviation`, `RollingSeasonalMean/Std/Zscore`, `RollingBipowerVar`, `RollingJumpShare`, `RollingVarianceRatio`, `RollingPermutationEntropy`, `BayesianChangepoint`, `RollingHMM`), `backtest_trades`, `backtest_report(by=)`, and `walk_forward(carry=, carry_multiplier=)` plus portfolio engines in `walk_forward`, in Python and JavaScript.

**Architecture:** Every operator is a header-only C++ node in `include/screamer/` (a `ScreamerBase` for 1 -> 1, a `FunctorBase<_, 1, 3>` for 1 -> 3), bound once in `bindings/*.cpp` and reached from JavaScript through the generated WASM manifest; the registry-driven suites enrol it from its docs page. The backtest helpers are C++ functions in `include/screamer/backtest_summary.h`, `include/screamer/backtest_trade_list.h` and `include/screamer/validation/position_test.h`, called by thin wrappers in `screamer/backtest.py`, `screamer/validation.py`, `wasm/backtest_bindings.cpp`, `wasm/validation_bindings.cpp`, `js/src/backtest.ts` and `js/src/validation.ts`. JavaScript values are checked against a Python-generated fixture, `wasm/smoke/oracle_research.json`.

**Tech Stack:** C++17 header-only core, nanobind, Emscripten/Embind + TypeScript (`js/`), pytest, node:test, Sphinx + myst (plotly and exec_code directives).

**Spec:** docs/superpowers/specs/2026-10-17-research-toolkit-design.md

## Global Constraints

- All operator and helper logic is C++ under `include/screamer/`; Python (`screamer/*.py`) and TypeScript (`js/src/*.ts`) only label, check arguments, and dispatch (CONTRIBUTING.md "Design principles").
- Every new operator ships in Python and JavaScript, works eager, per event, in a `Pipeline` (graph) and lazily, and batch equals per-event output bit for bit.
- Every output at step t depends on inputs up to t only (no backfill, no lookahead).
- A NaN input gives a NaN output and leaves the state unchanged (`nan_policy: ignore`), except `RollingSeasonal*` (`nan-aware`: a NaN step advances the slot clock and enters no window).
- `start_policy="zero"` for every new rolling operator means: the output equals the `"strict"` output on the input preceded by enough zeros to fill every window (the state starts as if it had seen zeros).
- Per-operator checklist (each operator task repeats its concrete paths): header `include/screamer/<name>.h`; binding in `bindings/bindings_<file>.cpp` (include + `nb::class_`); every integer setting in `include/screamer/common/int_settings.h` (alphabetical, minimum = the constructor's smallest accepted value); docs page `docs/functions_<family>/<Name>.md` (frontmatter with `min:` on every integer setting, `nan_policy`, `topics`, `see_also`, `output_names` for 1 -> 3, a plotly usage plot); baseline `devtools/baselines/<Name>.py` with class `<Name>_numpy`; entries in `tests/param_cases.py`; a test file with hand-computed numbers, a baseline comparison on random input, `assert_regimes_agree`, and NaN behaviour; cases in `devtools/wasm/gen_research_oracle.py`; regenerated files; JS test.
- Regeneration after a C++ or docs change, in this order (run from the repo root; `REGEN` below means exactly this block):
  ```bash
  make install-dev
  .venv/bin/python devtools/inject_nan_footnote.py
  .venv/bin/python devtools/build_help_registry.py
  .venv/bin/python devtools/build_topic_pages.py
  .venv/bin/python devtools/build_function_index.py
  .venv/bin/python devtools/build_gallery.py
  .venv/bin/python devtools/wasm/gen_wasm_manifest.py
  .venv/bin/python devtools/wasm/gen_wasm_bindings.py
  .venv/bin/python devtools/wasm/gen_ts_api.py
  .venv/bin/python devtools/wasm/gen_js_docs.py
  .venv/bin/python devtools/wasm/gen_null_oracle.py
  .venv/bin/python devtools/wasm/gen_field_axis_oracle.py
  .venv/bin/python devtools/wasm/gen_research_oracle.py
  ```
- JavaScript build and tests (`JSTEST` below): `source /Users/thijs/Projects/emsdk/emsdk_env.sh && cd js && npm run build:wasm && npm run build && npm test 2>&1 | tail -15`.
- Registry suites for one operator `<Name>` (`REGISTRY <Name>` below): `.venv/bin/python -m pytest tests/test_baselines.py tests/test_contract_compliance.py tests/test_stream_vs_batch.py tests/test_nan_input_compliance.py tests/test_nan_start_policy_compliance.py tests/test_doc_coverage.py tests/test_parity_registration.py tests/test_documented_defaults.py tests/test_int_settings.py tests/test_setting_kinds.py tests/test_batch_stream_bitwise.py tests/test_tensor.py tests/test_view.py tests/test_matrix.py tests/test_io_size.py -q -k "<Name>" 2>&1 | tail -5`.
- Freshness suites (`FRESH` below): `.venv/bin/python -m pytest tests/test_help_json_fresh.py tests/test_js_docs_fresh.py tests/test_ts_api_fresh.py tests/test_null_oracle_fresh.py tests/test_wasm_codegen_fresh.py tests/test_field_axis_order.py tests/test_research_oracle_fresh.py -q 2>&1 | tail -5` (the last file exists from Task 2 on).
- After any C++ change run `make install-dev` (not only `make build`) before Python tests; run `make tidy` before each commit that touches C++ and fix every finding.
- Targeted tests while iterating; the full Python suite (`.venv/bin/python -m pytest -q -x -n auto 2>&1 | tail -5`, drop `-n auto` if xdist is missing) and `JSTEST` once at the end of each task.
- Existing numbers do not change: `Hold` with `cooldown=0`, `BacktestStats`, `backtest_report` without `by`, and `walk_forward` without `carry` give identical results (the compat probe in Task 15 checks this).
- Never edit version strings (`pyproject.toml` version, `js/package.json` version, `screamer/__init__.py` version); never run `make patch/minor/major`; never `git stash`; never `git add -A` or `git add .`; never push or merge. Commit only the paths a task names, plus any other path `git status --short` shows as changed by that task's regeneration commands, each named explicitly.
- Docs prose follows CONTRIBUTING.md "Writing documentation": lead with what the thing is, no em dashes (ASCII hyphens only), no editorializing, report not narrate; state batch == live once, in a streaming context.
- Commit messages end with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01PZMDN2ijvnSZ7ZnrJUxnd8
  ```

## Review Focus

1. A NaN in a series on a regular clock fed to `RollingSeasonalMean/Std/Zscore`: the NaN step must still advance the slot counter, so every later value lands in its own slot (`[1, 10, nan, 20, 5, 30]` with `period=2` gives slot means `15` and `3`, not a shifted pairing). Test in Task 5.
2. Degenerate windows: a constant return window in `RollingVarianceRatio` (NaN, not inf), all-zero returns in `RollingJumpShare` (0), a constant window in `RollingHMM` (the variance floor keeps the fit finite, no exception, outputs finite). Tests in Tasks 6, 7 and 10.
3. Ties in `RollingPermutationEntropy` (prices that repeat on consecutive ticks): ties rank by order of appearance, and a constant or monotone series gives exactly 0, not a rounding residue. Test in Task 8.
4. Engine output with NaN rows, a sign flip, and a trade still open on the last row, fed to `backtest_trades` and `backtest_report(by=)`: flips close one trade and open the next, the open trade has `closed=False` and `exit = T - 1`, NaN rows count in `bars` but not in `steps`, and the closed trades agree with `BacktestStats`. Tests in Tasks 11 and 12.
5. `walk_forward(carry=)` with an engine whose last market input is not a price (`BacktestOHLCVTarget`'s volume, `BacktestTradesTarget`'s size, `BacktestL1Target`'s ask size): the carry is marked with the last price input (close, trade price, ask), never the volume or size; and a portfolio engine with K = 1 gives the single-asset scores. Tests in Tasks 13 and 14.

---

### Task 1: `permute_bars` / `permute_returns` error message (A5)

**Files:**
- Modify: `include/screamer/validation/surrogates.h` (`BarsPermutation` constructor ~line 504, `ReturnsPermutation` constructor ~line 641, a new `detail::not_positive_message`)
- Modify: `js/test/validation.test.ts:277-278` (the exact message)
- Test: `tests/test_validation.py` (append)

**Interfaces:**
- Consumes: nothing new.
- Produces: the `ValueError` (Python) / `RangeError` (JavaScript) message `<who> with mode="log" needs positive <prices|values>, but <n> value(s) is/are not positive; the first is <field>[<row>] = <value>; use mode="diff".`, with `[<row>, <asset>]` when there are several assets. `<field>` is `open`, `high`, `low` or `close` for `permute_bars` and `x` for `permute_returns`; `<value>` is the shortest round-trip decimal (`std::to_chars`).

- [ ] **Step 0: Record the compat probe of the unchanged build** (Task 15 diffs against it)

Run: `make install-dev && .venv/bin/python devtools/compat_probe.py --json build/compat_before.json`
Expected: the file is written (it stays untracked; `build/` is ignored).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_validation.py`; add `import re` at the top if it is not imported)

```python
def test_permute_returns_log_error_names_the_first_bad_value():
    x = np.array([1.0, 2.0, -2.0, 3.0, 0.0])
    msg = ('permute_returns with mode="log" needs positive values, but 2 values are not '
           'positive; the first is x[2] = -2; use mode="diff".')
    with pytest.raises(ValueError, match=re.escape(msg)):
        s.permute_returns(x, 3)


def test_permute_returns_log_error_names_the_asset():
    x = np.array([[1.0, 2.0], [3.0, -0.5], [2.0, 1.0]])
    msg = ('permute_returns with mode="log" needs positive values, but 1 value is not '
           'positive; the first is x[1, 1] = -0.5; use mode="diff".')
    with pytest.raises(ValueError, match=re.escape(msg)):
        s.permute_returns(x, 3)


def test_permute_bars_log_error_names_the_field_and_row():
    o = np.array([1.0, 2.0, 3.0])
    h = o + 1.0
    l = o - 0.5
    c = np.array([1.5, -1.0, 2.5])
    msg = ('permute_bars with mode="log" needs positive prices, but 1 value is not '
           'positive; the first is close[1] = -1; use mode="diff".')
    with pytest.raises(ValueError, match=re.escape(msg)):
        s.permute_bars(o, h, l, c, 2)


def test_permute_finite_error_keeps_its_precedence():
    # a non-finite row before any non-positive one still gives the finite message
    with pytest.raises(ValueError, match="needs finite values after the NaN warmup"):
        s.permute_returns(np.array([1.0, 2.0, np.inf, -1.0]), 3)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_validation.py -q -k "permute_returns_log_error or permute_bars_log_error or finite_error_keeps" 2>&1 | tail -5`
Expected: 3 FAIL (the old message), 1 PASS.

- [ ] **Step 3: Implement**

In `include/screamer/validation/surrogates.h` add `#include <charconv>` and, in `namespace detail` (before `BarsPermutation`), the message builder:

```cpp
        // The mode="log" refusal of a series with values at or below zero:
        // "<who> with mode=\"log\" needs positive <what>, but 2 values are not
        // positive; the first is close[3] = -1; use mode=\"diff\"." The
        // position is [row] for one asset and [row, asset] for several.
        inline std::string not_positive_message(const char* who, const char* what,
                                                std::size_t count, const char* field,
                                                std::size_t row, std::size_t asset,
                                                std::size_t assets, double value) {
            char num[32];
            const auto end = std::to_chars(num, num + sizeof num, value).ptr;
            std::string pos = std::to_string(row);
            if (assets > 1) pos += ", " + std::to_string(asset);
            return std::string(who) + " with mode=\"log\" needs positive " + what + ", but " +
                   std::to_string(count) + (count == 1 ? " value is" : " values are") +
                   " not positive; the first is " + field + "[" + pos + "] = " +
                   std::string(num, end) + "; use mode=\"diff\".";
        }
```

Replace the check loop of `BarsPermutation`'s constructor with:

```cpp
            const bool log = mode == PermuteMode::Log;
            const bool positive = log || (notional && vol);
            const std::size_t TK = T * K;
            static constexpr const char* kFields[4] = {"open", "high", "low", "close"};
            std::size_t bad = 0, first = 0;
            int first_field = 0;
            for (std::size_t i = 0; i < TK; ++i) {
                if (!std::isfinite(o[i] + h[i] + l[i] + c[i])) {
                    if (bad == 0)
                        throw std::invalid_argument(
                            "permute_bars needs finite open, high, low, and close prices.");
                    continue;
                }
                if (!positive) continue;
                const double v[4] = {o[i], h[i], l[i], c[i]};
                for (int f = 0; f < 4; ++f)
                    if (!(v[f] > 0.0) && bad++ == 0) { first = i; first_field = f; }
            }
            if (bad > 0) {
                if (!log) throw std::invalid_argument("volume_mode=\"notional\" needs positive prices.");
                const double v[4] = {o[first], h[first], l[first], c[first]};
                throw std::invalid_argument(detail::not_positive_message(
                    "permute_bars", "prices", bad, kFields[first_field], first / K, first % K, K,
                    v[first_field]));
            }
```

and the check loop of `ReturnsPermutation`'s constructor with:

```cpp
            const bool log = mode == PermuteMode::Log;
            std::size_t bad = 0, first = 0;
            for (std::size_t i = p * K; i < T * K; ++i) {
                if (!std::isfinite(x[i])) {
                    if (bad == 0)
                        throw std::invalid_argument(
                            "permute_returns needs finite values after the NaN warmup.");
                    continue;
                }
                if ((log || (notional && vol)) && !(x[i] > 0.0) && bad++ == 0) first = i;
            }
            if (bad > 0) {
                if (!log) throw std::invalid_argument("volume_mode=\"notional\" needs positive prices.");
                throw std::invalid_argument(detail::not_positive_message(
                    "permute_returns", "values", bad, "x", first / K, first % K, K, x[first]));
            }
```

(keep the `const bool log` declaration only once in each constructor; the code after the loop that already declares or uses `log` keeps using it). In `js/test/validation.test.ts` change the expected message at line 278 to:

```typescript
    { name: "RangeError", message: 'permute_returns with mode="log" needs positive values, but 1 value is not positive; the first is x[1] = -2; use mode="diff".' });
```

- [ ] **Step 4: Run the tests**

Run: `make install-dev && .venv/bin/python -m pytest tests/test_validation.py -q 2>&1 | tail -5`
Expected: all pass.

- [ ] **Step 5: Tidy, full suite, JavaScript**

Run: `make tidy`, the full Python suite, then `JSTEST`.
Expected: clean, all pass.

- [ ] **Step 6: Commit**

```bash
git add include/screamer/validation/surrogates.h tests/test_validation.py js/test/validation.test.ts
git commit -m "fix(validation): permute log-mode error names the first bad value and the count

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PZMDN2ijvnSZ7ZnrJUxnd8"
```

---

### Task 2: `Hold(cooldown=)` and the research oracle

**Files:**
- Modify: `include/screamer/hold.h`, `bindings/bindings_signal.cpp:125-132`, `include/screamer/common/int_settings.h` (add `{"Hold", "cooldown", 0},` before `{"Hold", "n", 1},`)
- Modify: `devtools/baselines/Hold.py`, `docs/functions_signal/Hold.md`, `tests/param_cases.py:205`, `tests/test_setting_kinds.py:103`
- Modify: `tests/regime_helpers.py` (add `assert_regimes_agree`)
- Create: `devtools/wasm/gen_research_oracle.py`, `wasm/smoke/oracle_research.json` (generated), `tests/test_research_oracle_fresh.py`, `js/test/research.test.ts`
- Test: `tests/test_hold.py` (append)

**Interfaces:**
- Produces: `Hold(n, release=0.0, cooldown=0)`; JavaScript `Hold({ n, release, cooldown })` and `Hold(n, release, cooldown)`.
- Produces: `tests.regime_helpers.assert_regimes_agree(op_factory, x)`, used by Tasks 3 to 10.
- Produces: `devtools/wasm/gen_research_oracle.py` with a `CASES` list and a `_helpers()` function that Tasks 3 to 14 extend, and `js/test/research.test.ts` that Tasks 3 to 14 extend.

**Behaviour:** after a hold expires (the step after its last held step), nonzero inputs in the next `cooldown` steps are ignored and the output is `release`. A trigger during a hold restarts it. A NaN step changes nothing (it does not use up a cooldown step). `cooldown=0` is the previous behaviour exactly.

- [ ] **Step 1: Add the regime helper** (append to `tests/regime_helpers.py`)

```python
def assert_regimes_agree(op_factory, x):
    """Eager, per-event, graph (Pipeline) and lazy runs of a one-input operator
    give the same output bit for bit, NaN in the same places."""
    import screamer
    x = np.asarray(x, dtype=float)
    eager = np.asarray(op_factory()(x), dtype=float)
    op = op_factory()
    stream = np.array([np.asarray(op(float(v)), dtype=float) for v in x]).reshape(eager.shape)
    node = screamer.Input("x")
    pipeline = screamer.Pipeline(inputs=[node], outputs=[op_factory()(node)])
    graph, _ = pipeline((x, np.arange(len(x))))
    lazy = np.array([np.asarray(r[0], dtype=float)
                     for r in pipeline(((float(v), i) for i, v in enumerate(x)))])
    assert_bitwise(stream, eager)
    assert_bitwise(np.asarray(graph, dtype=float).reshape(eager.shape), eager)
    assert_bitwise(lazy.reshape(eager.shape), eager)
```

- [ ] **Step 2: Write the failing tests** (append to `tests/test_hold.py`)

```python
import importlib.util
import pathlib

from tests.regime_helpers import assert_regimes_agree

_B = pathlib.Path(__file__).resolve().parents[1] / "devtools" / "baselines" / "Hold.py"
_spec = importlib.util.spec_from_file_location("hold_baseline", _B)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
Hold_numpy = _mod.Hold_numpy


def _old_hold(x, n, release=0.0):
    """Hold before cooldown existed: NaN skipped, a trigger restarts the hold."""
    held, remaining, out = release, 0, []
    for v in x:
        if math.isnan(v):
            out.append(math.nan)
        elif v != 0.0:
            held, remaining = v, n - 1
            out.append(v)
        elif remaining > 0:
            remaining -= 1
            out.append(held)
        else:
            out.append(release)
    return np.array(out)


def _events(seed, n=400):
    rng = np.random.default_rng(seed)
    x = np.where(rng.random(n) < 0.2, rng.standard_normal(n), 0.0)
    x[::37] = np.nan
    return x


def test_hold_cooldown_ignores_triggers_after_expiry():
    out = np.asarray(Hold(n=2, cooldown=2)(np.array([5., 0, 7, 3, 0, 4, 0])))
    np.testing.assert_array_equal(out, [5, 5, 0, 0, 0, 4, 4])


def test_hold_cooldown_trigger_during_hold_restarts():
    out = np.asarray(Hold(n=3, cooldown=1)(np.array([1., 0, 2, 0, 0, 0, 3, 0])))
    np.testing.assert_array_equal(out, [1, 1, 2, 2, 2, 0, 3, 3])


def test_hold_cooldown_after_one_step_hold_outputs_release():
    out = np.asarray(Hold(n=1, release=-1.0, cooldown=1)(np.array([5., 6, 7, 0, 8])))
    np.testing.assert_array_equal(out, [5, -1, 7, -1, 8])


def test_hold_cooldown_nan_does_not_use_up_a_cooldown_step():
    out = np.asarray(Hold(n=1, cooldown=1)(np.array([5., np.nan, 6, 0])))
    assert math.isnan(out[1])
    np.testing.assert_array_equal(out[[0, 2, 3]], [5, 0, 0])


@pytest.mark.parametrize("n", [1, 2, 5])
def test_hold_cooldown_zero_is_the_previous_behaviour(n):
    x = _events(n)
    np.testing.assert_array_equal(np.asarray(Hold(n=n, cooldown=0)(x)), _old_hold(x, n))
    np.testing.assert_array_equal(np.asarray(Hold(n=n)(x)), _old_hold(x, n))


@pytest.mark.parametrize("n,cooldown,release", [(1, 1, 0.0), (3, 2, 0.0), (5, 7, -1.0)])
def test_hold_cooldown_matches_baseline(n, cooldown, release):
    x = _events(10 + n)
    np.testing.assert_array_equal(np.asarray(Hold(n=n, release=release, cooldown=cooldown)(x)),
                                  Hold_numpy(n=n, release=release, cooldown=cooldown)(x))


def test_hold_cooldown_regimes_agree():
    assert_regimes_agree(lambda: Hold(n=3, cooldown=2), _events(3))


def test_hold_rejects_negative_cooldown():
    with pytest.raises(ValueError, match=r"^Hold: cooldown must be at least 0, got -1$"):
        Hold(n=3, cooldown=-1)
```

Change `tests/test_setting_kinds.py:103` to:

```python
    assert s.Hold.__doc__.splitlines()[0] == "Hold(n, release=0.0, cooldown=0)"
```

- [ ] **Step 3: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_hold.py -q 2>&1 | tail -5`
Expected: FAIL (`unexpected keyword argument 'cooldown'`).

- [ ] **Step 4: Implement**

`include/screamer/hold.h`, the class (keep the includes; extend the header comment with a cooldown paragraph and worked example `n=2, cooldown=2: [5, 0, 7, 3, 0, 4, 0] -> [5, 5, 0, 0, 0, 4, 4]`):

```cpp
class Hold : public ScreamerBase {
public:
    Hold(int n, double release = 0.0, int cooldown = 0)
        : n_(n), release_(release), cooldown_(cooldown)
    {
        if (n < 1) {
            throw std::invalid_argument("n must be >= 1.");
        }
        if (cooldown < 0) {
            throw std::invalid_argument("cooldown must be >= 0.");
        }
        if (isinf2(release)) {
            throw std::invalid_argument("release must be finite or NaN.");
        }
        reset();
    }

    void reset() override {
        remaining_ = 0;
        cool_ = 0;
        held_ = release_;
    }

private:
    double process_scalar(double x) override {
        if (isnan2(x)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (remaining_ == 0 && cool_ > 0) {     // cooling down: triggers ignored
            --cool_;
            return release_;
        }
        if (x != 0.0) {
            held_ = x;
            remaining_ = n_ - 1;
            if (remaining_ == 0) cool_ = cooldown_;
            return x;
        }
        if (remaining_ > 0) {
            if (--remaining_ == 0) cool_ = cooldown_;
            return held_;
        }
        return release_;
    }

    const int n_;
    const double release_;
    const int cooldown_;
    double held_ = std::numeric_limits<double>::quiet_NaN();
    int remaining_ = 0;
    int cool_ = 0;
};
```

`bindings/bindings_signal.cpp`:

```cpp
    nb::class_<screamer::Hold, screamer::ScreamerBase>(m, "Hold")
        .def(nb::init<screamer::IntArg<int>, screamer::FloatArg, screamer::IntArg<int>>(),
             "n"_a, "release"_a = 0.0, "cooldown"_a = 0)
        .def("__call__", &screamer::screamer_call, "value"_a.none())
        .def("reset", &screamer::Hold::reset,
             "Reset to the initial state (remaining=0, cooldown=0, held=release).");
```

`devtools/baselines/Hold.py`:

```python
"""Time latch: a nonzero input is held for n steps total, then `cooldown`
steps in which nonzero inputs are ignored.

The trigger step counts as step 1, so a nonzero input at t is emitted at
t, t+1, ..., t+n-1. When no hold is active the output is `release`. A NaN
input gives NaN and changes nothing.
"""
import math

import numpy as np


class Hold_numpy:

    def __init__(self, n=3, release=0.0, cooldown=0):
        if n < 1 or cooldown < 0:
            raise ValueError("n must be >= 1 and cooldown >= 0")
        self.n, self.release, self.cooldown = n, release, cooldown

    def __call__(self, x):
        held, remaining, cool = self.release, 0, 0
        out = np.empty(len(x), dtype=float)
        for i, v in enumerate(np.asarray(x, dtype=float)):
            if math.isnan(v):
                out[i] = math.nan
            elif remaining == 0 and cool > 0:
                cool -= 1
                out[i] = self.release
            elif v != 0.0:
                held, remaining = v, self.n - 1
                if remaining == 0:
                    cool = self.cooldown
                out[i] = v
            elif remaining > 0:
                remaining -= 1
                if remaining == 0:
                    cool = self.cooldown
                out[i] = held
            else:
                out[i] = self.release
        return out
```

`tests/param_cases.py:205`: `( ('Hold',)                  , {"n": [5], "cooldown": [0, 3]}),`

`docs/functions_signal/Hold.md`: add the parameter after `release`,

```yaml
- name: cooldown
  type: int
  default: 0
  min: 0
  description: Steps after a hold expires in which nonzero inputs are ignored and the output
    is `release`. 0 turns it off.
```

add to the Description, after the paragraph about `release`: "After a hold expires, nonzero inputs in the next `cooldown` steps are ignored and the output is `release`; a trigger during a hold still restarts it. A `NaN` step does not count as a cooldown step. With `n=2, cooldown=2`, the input `[5, 0, 7, 3, 0, 4, 0]` gives `[5, 5, 0, 0, 0, 4, 4]`: the triggers 7 and 3 fall in the cooldown." Add the cooldown row to the formula block (`release` if in cooldown, decrement cooldown) and `cooldown` (steps left, initialized to 0) to Implementation Details and to the `reset()` sentence.

- [ ] **Step 5: Create the research oracle generator** (`devtools/wasm/gen_research_oracle.py`)

```python
#!/usr/bin/env python3
"""Write wasm/smoke/oracle_research.json: the operators and helpers of the
research toolkit, computed by the Python binding, for js/test/research.test.ts.

Each operator case names the operator, its settings in Python names, and one
of the input series below; js/test/research.test.ts builds the same operator
in JavaScript (settings in camelCase), runs it on the same series, and
compares the result, to 1e-9 relative unless the case gives a tolerance. The
helper cases hold the inputs and results of backtest_trades, backtest_report
with by=, and walk_forward with carry= and portfolio engines. A non-finite
number is written as null (NaN), "inf" or "-inf".

Modes:
    (no args)  write the fixture
    --stdout   print the fixture to stdout (freshness gate)
"""
import argparse
import json
import math
import os
import sys

import numpy as np

import screamer as s

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "wasm", "smoke", "oracle_research.json")


def _inputs():
    rng = np.random.default_rng(20261017)
    T = 600
    events = np.where(rng.random(T) < 0.15, np.round(rng.standard_normal(T), 3), 0.0)
    events[[40, 41, 300]] = np.nan
    scale = np.where((np.arange(T) // 150) % 2 == 0, 0.5, 2.0)
    returns = rng.standard_normal(T) * scale
    returns[[100, 450]] = np.nan
    return {"events": events, "returns": returns}


# (operator, settings, input series[, relative tolerance]). Each operator
# task adds its cases here.
CASES = [
    ("Hold", {"n": 3}, "events"),
    ("Hold", {"n": 3, "cooldown": 4}, "events"),
    ("Hold", {"n": 1, "release": -1.0, "cooldown": 2}, "events"),
]


def _helpers():
    """Inputs and results of the backtest helpers; Tasks 11 to 14 add entries."""
    return {}


def _num(v):
    v = float(v)
    if math.isnan(v):
        return None
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return v


def _flat(a):
    return [_num(v) for v in np.asarray(a, dtype=float).ravel()]


def build():
    inputs = _inputs()
    cases = []
    for case in CASES:
        name, settings, key = case[:3]
        out = np.asarray(getattr(s, name)(**settings)(inputs[key]), dtype=float)
        entry = {"name": name, "settings": settings, "input": key,
                 "shape": list(out.shape), "out": _flat(out)}
        if len(case) > 3:
            entry["tol"] = case[3]
        cases.append(entry)
    return {"inputs": {k: _flat(v) for k, v in inputs.items()}, "cases": cases,
            "helpers": _helpers()}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stdout", action="store_true", help="print instead of writing")
    args = p.parse_args()
    text = json.dumps(build(), separators=(",", ":")) + "\n"
    if args.stdout:
        sys.stdout.write(text)
    else:
        with open(OUT, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
```

`tests/test_research_oracle_fresh.py`:

```python
"""Freshness gate for wasm/smoke/oracle_research.json.

devtools/wasm/gen_research_oracle.py records the research-toolkit operators
and backtest helpers as the Python binding computes them;
js/test/research.test.ts checks that JavaScript gives the same values. This
test fails when the committed fixture no longer matches the Python binding.
"""
import pathlib

import pytest

from tests._oracle_fresh import assert_oracle_fresh

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
GEN = REPO_ROOT / "devtools" / "wasm" / "gen_research_oracle.py"
FIXTURE = REPO_ROOT / "wasm" / "smoke" / "oracle_research.json"

pytestmark = pytest.mark.skipif(not GEN.exists(), reason="needs the source tree")


def test_research_oracle_is_fresh():
    assert_oracle_fresh(GEN, FIXTURE, REPO_ROOT)
```

`js/test/research.test.ts`:

```typescript
// The research-toolkit operators and backtest helpers against the Python
// binding. Expected values come from wasm/smoke/oracle_research.json
// (devtools/wasm/gen_research_oracle.py).
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { ready, Hold } from "../src/index.ts";
import * as api from "../src/index.ts";

await ready();

type Num = number | null | "inf" | "-inf";
type Case = { name: string; settings: Record<string, unknown>; input: string; shape: number[];
              out: Num[]; tol?: number };
const oracle: { inputs: Record<string, Num[]>; cases: Case[]; helpers: Record<string, any> } =
  JSON.parse(readFileSync(new URL("../../wasm/smoke/oracle_research.json", import.meta.url), "utf8"));
const num = (v: Num): number => (v === null ? NaN : v === "inf" ? Infinity : v === "-inf" ? -Infinity : v);
const series = (k: string) => Float64Array.from(oracle.inputs[k], num);
const camel = (k: string) => k.replace(/_([a-z])/g, (_m, c) => c.toUpperCase());
const options = (s: Record<string, unknown>) =>
  Object.fromEntries(Object.entries(s).map(([k, v]) => [camel(k), v]));

export const close = (got: ArrayLike<number>, want: Num[], what: string, tol = 1e-9) => {
  assert.equal(got.length, want.length, `${what}: length`);
  for (let i = 0; i < want.length; i++) {
    const w = num(want[i]);
    if (Number.isNaN(w)) assert.ok(Number.isNaN(got[i]), `${what} [${i}]: ${got[i]} is not NaN`);
    else if (!Number.isFinite(w)) assert.equal(got[i], w, `${what} [${i}]`);
    else assert.ok(Math.abs(got[i] - w) <= tol * Math.max(1, Math.abs(w)), `${what} [${i}]: ${got[i]} vs ${w}`);
  }
};

const dataOf = (out: any): ArrayLike<number> => (ArrayBuffer.isView(out) ? out as Float64Array : out.data);

for (const c of oracle.cases) {
  test(`research: ${c.name}(${JSON.stringify(c.settings)}) matches Python`, () => {
    const make = (api as any)[c.name];
    const out = make(options(c.settings))(series(c.input));
    close(dataOf(out), c.out, c.name, c.tol ?? 1e-9);
  });
}

test("research: Hold cooldown worked example", () => {
  const out = Hold({ n: 2, cooldown: 2 })(Float64Array.from([5, 0, 7, 3, 0, 4, 0])) as Float64Array;
  assert.deepEqual(Array.from(out), [5, 5, 0, 0, 0, 4, 4]);
});
```

- [ ] **Step 6: Regenerate and run the tests**

Run: `REGEN`, then `.venv/bin/python -m pytest tests/test_hold.py tests/test_setting_kinds.py tests/test_research_oracle_fresh.py -q 2>&1 | tail -5`, then `REGISTRY Hold`, then `FRESH`.
Expected: all pass.

- [ ] **Step 7: Tidy, full suite, JavaScript**

Run: `make tidy`, the full Python suite, `JSTEST`.
Expected: clean, all pass (`research.test.ts` included).

- [ ] **Step 8: Commit**

```bash
git add include/screamer/hold.h bindings/bindings_signal.cpp include/screamer/common/int_settings.h \
  devtools/baselines/Hold.py docs/functions_signal/Hold.md tests/param_cases.py tests/test_setting_kinds.py \
  tests/regime_helpers.py tests/test_hold.py devtools/wasm/gen_research_oracle.py \
  wasm/smoke/oracle_research.json tests/test_research_oracle_fresh.py js/test/research.test.ts \
  screamer/data/help.json devtools/wasm/wasm_manifest.json wasm/generated/bindings_wasm.cpp \
  js/src/generated/ops.ts js/src/generated/ops.d.ts js/src/generated/docs_data.ts js/test/docs_oracle.json \
  wasm/smoke/null_oracle.json
git status --short   # add any other path the regeneration changed, by name
git commit -m "feat(signal): Hold cooldown, and the research oracle for JavaScript parity

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PZMDN2ijvnSZ7ZnrJUxnd8"
```

---

### Task 3: `CusumFilter` and the `regime` topic

**Files:**
- Create: `include/screamer/cusum_filter.h`, `docs/functions_signal/CusumFilter.md`, `devtools/baselines/CusumFilter.py`, `tests/test_cusum_filter.py`
- Modify: `bindings/bindings_signal.cpp` (include + class next to `Hold`), `docs/topics.yml` (new topic `regime` in group `machine-learning`), `tests/param_cases.py` (explicit entry), `tests/test_batch_stream_bitwise.py` (`REQUIRED_KWARGS`), `devtools/wasm/gen_null_oracle.py` (`FILL`), `devtools/wasm/gen_research_oracle.py` (`CASES`), `js/test/research.test.ts`

**Interfaces:**
- Produces: `CusumFilter(threshold, drift=0.0)`, 1 -> 1, output in `{-1, 0, +1}`; JavaScript `CusumFilter({ threshold, drift })`. Topic slug `regime`, used by Tasks 6 to 10.

**Behaviour (spec C3):** `S+ = max(0, S+ + x - drift)`, `S- = min(0, S- + x + drift)`; output `+1` when `S+ > threshold`, `-1` when `S- < -threshold`, and then both sums reset to 0; else 0. `threshold > 0` and finite, `drift >= 0` and finite. NaN: NaN output, sums unchanged.

- [ ] **Step 1: Write the baseline** (`devtools/baselines/CusumFilter.py`)

```python
"""Symmetric CUSUM event filter (Lopez de Prado 2018, section 2.5.2.1),
written as a plain loop over the input."""
import math

import numpy as np


class CusumFilter_numpy:
    def __init__(self, threshold=1.0, drift=0.0):
        self.h, self.drift = threshold, drift

    def __call__(self, x):
        pos = neg = 0.0
        out = np.empty(len(x), dtype=float)
        for i, v in enumerate(np.asarray(x, dtype=float)):
            if math.isnan(v):
                out[i] = math.nan
                continue
            pos = max(0.0, pos + v - self.drift)
            neg = min(0.0, neg + v + self.drift)
            if pos > self.h:
                out[i], pos, neg = 1.0, 0.0, 0.0
            elif neg < -self.h:
                out[i], pos, neg = -1.0, 0.0, 0.0
            else:
                out[i] = 0.0
        return out
```

- [ ] **Step 2: Write the failing tests** (`tests/test_cusum_filter.py`)

```python
import math

import numpy as np
import pytest

from devtools.baselines.CusumFilter import CusumFilter_numpy
from screamer import CusumFilter
from tests.regime_helpers import assert_regimes_agree


def test_hand_example():
    x = np.array([0.5, 0.6, -0.2, 0.3, -1.5, 0.1, np.nan, 0.4])
    out = np.asarray(CusumFilter(threshold=1.0)(x))
    np.testing.assert_array_equal(out[[0, 1, 2, 3, 4, 5, 7]], [0, 1, 0, 0, -1, 0, 0])
    assert math.isnan(out[6])


def test_threshold_is_strict():
    # 0.5 + 0.5 = 1.0 is not above 1.0; the third step is
    out = np.asarray(CusumFilter(threshold=1.0)(np.array([0.5, 0.5, 0.5])))
    np.testing.assert_array_equal(out, [0, 0, 1])


def test_drift():
    x = np.array([0.5, 0.5, 0.5, 0.25, -1.0, -1.0])
    out = np.asarray(CusumFilter(threshold=1.0, drift=0.25)(x))
    np.testing.assert_array_equal(out, [0, 0, 0, 0, 0, -1])


def test_both_sums_reset_after_an_event():
    # after the +1 at step 1, S- restarts from 0, so -0.9 alone does not fire
    out = np.asarray(CusumFilter(threshold=1.0)(np.array([-0.5, 1.6, -0.9, -0.2])))
    np.testing.assert_array_equal(out, [0, 1, 0, -1])


@pytest.mark.parametrize("threshold,drift", [(0.5, 0.0), (2.0, 0.1), (3.0, 0.5)])
def test_matches_baseline(threshold, drift):
    rng = np.random.default_rng(int(threshold * 10))
    x = rng.standard_normal(2000)
    x[::97] = np.nan
    np.testing.assert_array_equal(np.asarray(CusumFilter(threshold, drift)(x)),
                                  CusumFilter_numpy(threshold, drift)(x))


def test_regimes_agree():
    x = np.random.default_rng(1).standard_normal(500)
    x[[10, 200]] = np.nan
    assert_regimes_agree(lambda: CusumFilter(threshold=1.5, drift=0.1), x)


def test_settings_are_checked():
    with pytest.raises(ValueError):
        CusumFilter(threshold=0.0)
    with pytest.raises(ValueError):
        CusumFilter(threshold=1.0, drift=-0.1)
    with pytest.raises(ValueError, match="threshold must be finite"):
        CusumFilter(threshold=math.inf)
```

The `-0.5, 1.6, -0.9, -0.2` case: step 0 `S+ = 0, S- = -0.5`; step 1 `S+ = 1.6 > 1` fires +1 and resets both; step 2 `S- = -0.9`; step 3 `S- = -1.1 < -1` fires -1.

- [ ] **Step 3: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_cusum_filter.py -q 2>&1 | tail -5`
Expected: FAIL (`ImportError: cannot import name 'CusumFilter'`).

- [ ] **Step 4: Implement** `include/screamer/cusum_filter.h`

```cpp
#ifndef SCREAMER_CUSUM_FILTER_H
#define SCREAMER_CUSUM_FILTER_H

// CusumFilter: the symmetric CUSUM event filter (Lopez de Prado 2018,
// section 2.5.2.1). Two running sums of the input,
//     S+ = max(0, S+ + x - drift),   S- = min(0, S- + x + drift),
// give +1 when S+ > threshold and -1 when S- < -threshold, after which both
// sums restart from 0; otherwise 0. The input is normally returns or price
// differences, and an event marks a cumulative move of more than `threshold`
// since the last event. 1 -> 1, O(1) per step.
// nan_policy: ignore (a NaN gives NaN and leaves both sums unchanged).

#include <limits>
#include <stdexcept>
#include "screamer/common/base.h"
#include "screamer/common/finite_settings.h"
#include "screamer/common/float_info.h"

namespace screamer {

class CusumFilter : public ScreamerBase {
public:
    explicit CusumFilter(double threshold, double drift = 0.0)
        : threshold_(threshold), drift_(drift)
    {
        require_finite("threshold", threshold);
        require_finite("drift", drift);
        if (!(threshold > 0.0)) throw std::invalid_argument("threshold must be positive.");
        if (drift < 0.0) throw std::invalid_argument("drift must be 0 or more.");
        reset();
    }

    void reset() override {
        pos_ = 0.0;
        neg_ = 0.0;
    }

private:
    double process_scalar(double x) override {
        if (isnan2(x)) return std::numeric_limits<double>::quiet_NaN();
        const double up = pos_ + x - drift_;
        const double down = neg_ + x + drift_;
        pos_ = up > 0.0 ? up : 0.0;
        neg_ = down < 0.0 ? down : 0.0;
        if (pos_ > threshold_) {
            pos_ = 0.0;
            neg_ = 0.0;
            return 1.0;
        }
        if (neg_ < -threshold_) {
            pos_ = 0.0;
            neg_ = 0.0;
            return -1.0;
        }
        return 0.0;
    }

    const double threshold_;
    const double drift_;
    double pos_ = 0.0;
    double neg_ = 0.0;
};

}  // namespace screamer

#endif  // SCREAMER_CUSUM_FILTER_H
```

`bindings/bindings_signal.cpp`: `#include "screamer/cusum_filter.h"` and, after `Hold`,

```cpp
    // CusumFilter: symmetric CUSUM event filter, +1 / -1 / 0 per step.
    nb::class_<screamer::CusumFilter, screamer::ScreamerBase>(m, "CusumFilter")
        .def(nb::init<screamer::FloatArg, screamer::FloatArg>(), "threshold"_a, "drift"_a = 0.0)
        .def("__call__", &screamer::screamer_call, "value"_a.none())
        .def("reset", &screamer::CusumFilter::reset, "Reset both sums to 0.");
```

`docs/topics.yml`: in `groups: machine-learning:` set `topics: [supervised, regime]` and `desc: "Training sets and labels from time-series data, online regression and filtering that forecast as they update, and regime and change-point models."`; under `topics:` after `supervised:` add

```yaml
  regime:
    name: "Regimes & change points"
    desc: "Detect a change in a series' behaviour: hidden Markov regimes, Bayesian change points, CUSUM events, and tests for persistence, jumps, and randomness."
```

`tests/param_cases.py`, in `test_definitions` after the `SchmittTrigger` line: `( ('CusumFilter',)           , {"threshold": [0.5, 2.0], "drift": [0.0, 0.1]}),`

`tests/test_batch_stream_bitwise.py` `REQUIRED_KWARGS`: add `"CusumFilter": {"threshold": 1.0},`

`devtools/wasm/gen_null_oracle.py`: `FILL = {"n": 3, "lower": 1.0, "upper": 3.0, "threshold": 1.0}`

`devtools/wasm/gen_research_oracle.py` `CASES`: add

```python
    ("CusumFilter", {"threshold": 1.0}, "returns"),
    ("CusumFilter", {"threshold": 2.5, "drift": 0.2}, "returns"),
```

`js/test/research.test.ts`: add `CusumFilter` to the import and

```typescript
test("research: CusumFilter hand example", () => {
  const out = CusumFilter({ threshold: 1 })(Float64Array.from([0.5, 0.6, -0.2, 0.3, -1.5, 0.1, NaN, 0.4])) as Float64Array;
  assert.deepEqual(Array.from(out.subarray(0, 6)), [0, 1, 0, 0, -1, 0]);
  assert.ok(Number.isNaN(out[6]));
  assert.equal(out[7], 0);
});
```

`docs/functions_signal/CusumFilter.md`:

````markdown
---
name: CusumFilter
title: CUSUM event filter
implementation_family: signal
topics:
- signals
- regime
tags:
- cusum
- event filter
- event sampling
- structural break
- change detection
- labeling
short: 'Symmetric CUSUM event filter: +1 when the cumulative rise since the last event passes
  a threshold, -1 for a fall, 0 otherwise.'
inputs: 1
outputs: 1
parameters:
- name: threshold
  type: float
  default: 1.0
  required: true
  description: Cumulative move, in the units of the input, that triggers an event. Must be
    positive.
- name: drift
  type: float
  default: 0.0
  min: 0.0
  description: Amount subtracted from each rise and from each fall before it is accumulated,
    so moves smaller than `drift` per step do not build up. 0 or more.
nan_policy: ignore
see_also:
- Hold
- TripleBarrier
- BayesianChangepoint
- SchmittTrigger
references:
- 'Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*, section 2.5.2.1. Wiley.'
---

# `CusumFilter`

## Description

`CusumFilter` is the symmetric CUSUM event filter (Lopez de Prado 2018). It
keeps two running sums of the input,

$$
S^+_t = \max(0,\ S^+_{t-1} + x_t - \text{drift}), \qquad
S^-_t = \min(0,\ S^-_{t-1} + x_t + \text{drift}),
$$

and outputs `+1` when `S+ > threshold`, `-1` when `S- < -threshold`, and `0`
otherwise. After an event both sums restart from 0. The input is normally
returns or price differences, so an event marks a cumulative move of more than
`threshold` since the last event, in either direction.

The events are a sampling rule: label or trade only at the steps where the
output is not 0, for example with
[`TripleBarrier`](../functions_fin/TripleBarrier.md) entries at those steps.
To hold a position for a while after each event, follow it with
[`Hold`](Hold.md).

With `threshold=1`, the input `[0.5, 0.6, -0.2, 0.3, -1.5]` gives
`[0, 1, 0, 0, -1]`: the sum of the first two values passes 1, and after the
reset the fall to -1.5 passes -1.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input gives a `NaN` output at that step. The step is skipped: it does not enter the window or update the internal state. See [NaN and warmup](../nan_and_warmup.md).
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from screamer import CusumFilter, Diff

    rng = np.random.default_rng(5)
    price = 100 + np.cumsum(rng.standard_normal(400) * 0.5)
    events = np.asarray(CusumFilter(threshold=2.0)(Diff(1)(price)))
    t = np.arange(len(price))

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=price, name='price', line=dict(color='gray')))
    for value, color, name in [(1, 'seagreen', 'rise event (+1)'), (-1, 'crimson', 'fall event (-1)')]:
        at = t[events == value]
        fig.add_trace(go.Scatter(x=at, y=price[at], mode='markers', name=name,
                                 marker=dict(color=color, size=7)))
    fig.update_layout(title='CusumFilter: an event at each cumulative move of more than 2',
                      xaxis_title='step', yaxis_title='price',
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```

<!-- HELP_END -->
````

- [ ] **Step 5: Regenerate and run the tests**

Run: `REGEN`, then `.venv/bin/python -m pytest tests/test_cusum_filter.py -q 2>&1 | tail -5`, `REGISTRY CusumFilter`, `FRESH`.
Expected: all pass. `docs/by_group/machine-learning.rst` now has a "Regimes & change points" section listing `CusumFilter`.

- [ ] **Step 6: Tidy, full suite, JavaScript**

Run: `make tidy`, the full Python suite, `JSTEST`.
Expected: clean, all pass.

- [ ] **Step 7: Commit**

```bash
git add include/screamer/cusum_filter.h bindings/bindings_signal.cpp docs/functions_signal/CusumFilter.md \
  docs/topics.yml devtools/baselines/CusumFilter.py tests/test_cusum_filter.py tests/param_cases.py \
  tests/test_batch_stream_bitwise.py devtools/wasm/gen_null_oracle.py devtools/wasm/gen_research_oracle.py \
  js/test/research.test.ts screamer/__init__.py screamer/data/help.json screamer/data/topics.json \
  docs/by_group/machine-learning.rst docs/by_group_index.rst docs/by_group_cards.inc docs/function_index.txt \
  docs/gallery.rst docs/_static/gallery/manifest.json docs/_static/gallery/functions_signal__CusumFilter.webp \
  devtools/wasm/wasm_manifest.json wasm/generated/bindings_wasm.cpp js/src/generated/ops.ts \
  js/src/generated/ops.d.ts js/src/generated/docs_data.ts js/test/docs_oracle.json \
  wasm/smoke/null_oracle.json wasm/smoke/oracle_research.json
git status --short   # add any other path the regeneration changed, by name
git commit -m "feat(signal): CusumFilter event filter and the regime topic

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PZMDN2ijvnSZ7ZnrJUxnd8"
```

---

### Task 4: `RollingUpsideDeviation`

**Files:**
- Create: `include/screamer/rolling_upside_deviation.h`, `docs/functions_fin/RollingUpsideDeviation.md`, `devtools/baselines/RollingUpsideDeviation.py`, `tests/test_rolling_upside_deviation.py`
- Modify: `bindings/bindings_fin.cpp` (include + class after `RollingDownsideDeviation` ~line 1596), `include/screamer/common/int_settings.h` (`{"RollingUpsideDeviation", "window_size", 2},` after `RollingSum`/alphabetical place), `tests/param_cases.py` (`_ROLLING_AUTO_EXCLUDE` + explicit entry), `docs/functions_fin/RollingDownsideDeviation.md` (`see_also`), `devtools/wasm/gen_research_oracle.py`, `js/test/research.test.ts`

**Interfaces:**
- Produces: `RollingUpsideDeviation(window_size=20, mar=0.0, start_policy="strict")`, 1 -> 1; JavaScript `RollingUpsideDeviation({ windowSize, mar, startPolicy })`.

**Behaviour (spec B):** `sqrt(mean(max(x - mar, 0)^2))` over the window; the mirror of `RollingDownsideDeviation`, so `RollingUpsideDeviation(w, mar)(x) == RollingDownsideDeviation(w, -mar)(-x)`.

- [ ] **Step 1: Write the baselines helper and the baseline**

Append to `devtools/baselines/_windowing.py`:

```python
def ignoring_nan(values, compute):
    """compute() on the input with its NaN steps removed, with NaN put back at
    those steps: the output of an operator whose NaN policy is ``ignore``."""
    values = np.asarray(values, dtype=float)
    keep = ~np.isnan(values)
    part = np.asarray(compute(values[keep]), dtype=float)
    out = np.full((len(values),) + part.shape[1:], np.nan)
    out[keep] = part
    return out


def trailing(values, window, start_policy, minimum=1):
    """The trailing windows of ``values`` as a list (None where the policy
    gives no output): ``strict`` needs ``window`` values, ``expanding`` at
    least ``minimum``, and ``zero`` sees ``window`` zeros before the input."""
    values = np.asarray(values, dtype=float)
    pad = window if start_policy == "zero" else 0
    x = np.concatenate([np.zeros(pad), values])
    out = []
    for t in range(pad, len(x)):
        w = x[max(0, t - window + 1): t + 1]
        need = window if start_policy == "strict" else minimum
        out.append(w if len(w) >= need else None)
    return out
```

`devtools/baselines/RollingUpsideDeviation.py`:

```python
"""Upside semideviation over a trailing window, sqrt(mean(max(r - mar, 0)^2))."""
import numpy as np

from ._windowing import ignoring_nan, trailing


class RollingUpsideDeviation_numpy:
    def __init__(self, window_size=20, mar=0.0, start_policy="strict"):
        self.w, self.mar, self.sp = window_size, mar, start_policy

    def _run(self, x):
        out = []
        for win in trailing(x, self.w, self.sp):
            out.append(np.nan if win is None
                       else np.sqrt(np.mean(np.maximum(win - self.mar, 0.0) ** 2)))
        return np.array(out)

    def __call__(self, x):
        return ignoring_nan(x, self._run)
```

- [ ] **Step 2: Write the failing tests** (`tests/test_rolling_upside_deviation.py`)

```python
import math

import numpy as np
import pytest

from devtools.baselines.RollingUpsideDeviation import RollingUpsideDeviation_numpy
from screamer import RollingDownsideDeviation, RollingUpsideDeviation
from tests.regime_helpers import assert_regimes_agree


def test_hand_example_strict():
    out = np.asarray(RollingUpsideDeviation(3)(np.array([1.0, -2.0, 2.0, 0.0])))
    assert np.isnan(out[:2]).all()
    np.testing.assert_allclose(out[2:], [math.sqrt(5 / 3), math.sqrt(4 / 3)], rtol=1e-12)


def test_hand_example_mar():
    out = np.asarray(RollingUpsideDeviation(3, mar=1.0)(np.array([1.0, -2.0, 2.0])))
    np.testing.assert_allclose(out[2], math.sqrt(1 / 3), rtol=1e-12)


def test_hand_example_expanding_and_zero():
    x = np.array([1.0, -2.0, 2.0])
    exp = np.asarray(RollingUpsideDeviation(3, start_policy="expanding")(x))
    np.testing.assert_allclose(exp, [1.0, math.sqrt(1 / 2), math.sqrt(5 / 3)], rtol=1e-12)
    zero = np.asarray(RollingUpsideDeviation(3, start_policy="zero")(x))
    np.testing.assert_allclose(zero, [math.sqrt(1 / 3), math.sqrt(1 / 3), math.sqrt(5 / 3)], rtol=1e-12)


@pytest.mark.parametrize("mar", [0.0, 0.3, -0.2])
def test_mirror_of_downside_deviation(mar):
    x = np.random.default_rng(0).standard_normal(300)
    np.testing.assert_allclose(np.asarray(RollingUpsideDeviation(30, mar=mar)(x)),
                               np.asarray(RollingDownsideDeviation(30, mar=-mar)(-x)),
                               rtol=1e-12, equal_nan=True)


@pytest.mark.parametrize("start_policy", ["strict", "expanding", "zero"])
def test_matches_baseline(start_policy):
    x = np.random.default_rng(1).standard_normal(400)
    x[[5, 77, 300]] = np.nan
    got = np.asarray(RollingUpsideDeviation(25, mar=0.1, start_policy=start_policy)(x))
    want = RollingUpsideDeviation_numpy(25, 0.1, start_policy)(x)
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12, equal_nan=True)


def test_regimes_agree():
    x = np.random.default_rng(2).standard_normal(300)
    x[50] = np.nan
    assert_regimes_agree(lambda: RollingUpsideDeviation(20, mar=0.05), x)
```

- [ ] **Step 3: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_rolling_upside_deviation.py -q 2>&1 | tail -5`
Expected: FAIL (`ImportError`).

- [ ] **Step 4: Implement** `include/screamer/rolling_upside_deviation.h`

```cpp
#ifndef SCREAMER_ROLLING_UPSIDE_DEVIATION_H
#define SCREAMER_ROLLING_UPSIDE_DEVIATION_H

#include "screamer/common/finite_settings.h"
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include "screamer/common/base.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/rolling_sum.h"
#include "screamer/detail/start_policy.h"

namespace screamer {

    // RollingUpsideDeviation: the trailing-window upside semideviation of a
    // return series, the root-mean-square of the excesses above a minimum
    // acceptable return `mar`,
    //     sqrt( mean( max(x - mar, 0)^2 ) ).
    // The mirror of RollingDownsideDeviation: only returns above `mar`
    // contribute. 1 -> 1; one detail::RollingSum of the squared excess.
    // nan_policy: ignore (a NaN input yields NaN and leaves the window
    // untouched).
    class RollingUpsideDeviation : public ScreamerBase {
    public:
        RollingUpsideDeviation(int window_size, double mar = 0.0,
                               const std::string& start_policy = "strict")
            : window_size_(window_size),
              mar_(mar),
              start_policy_(detail::parse_start_policy(start_policy)),
              sum_e2_(window_size >= 2 ? static_cast<std::size_t>(window_size) : 2, start_policy)
        {
            require_finite("mar", mar);
            if (window_size_ < 2) {
                throw std::invalid_argument("Window size must be 2 or more.");
            }
            reset();
        }

        void reset() override {
            sum_e2_.reset();
            n_ = (start_policy_ != detail::StartPolicy::Zero) ? 0 : static_cast<std::size_t>(window_size_);
        }

        double process_scalar(double x) override {
            if (isnan2(x)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            if ((n_ < static_cast<std::size_t>(window_size_)) &&
                (start_policy_ != detail::StartPolicy::Zero)) {
                n_++;
            }
            double excess = x - mar_;
            if (excess < 0.0) excess = 0.0;               // upside only
            const double sum = sum_e2_.append(excess * excess);
            return std::sqrt(sum / static_cast<double>(n_));
        }

    private:
        const int window_size_;
        const double mar_;
        const detail::StartPolicy start_policy_;
        std::size_t n_ = 0;
        detail::RollingSum sum_e2_;
    };

} // namespace screamer

#endif // SCREAMER_ROLLING_UPSIDE_DEVIATION_H
```

`bindings/bindings_fin.cpp`: `#include "screamer/rolling_upside_deviation.h"` and

```cpp
    nb::class_<screamer::RollingUpsideDeviation, screamer::ScreamerBase>(m, "RollingUpsideDeviation")
        .def(nb::init<screamer::IntArg<int>, screamer::FloatArg, const std::string&>(),
             "window_size"_a = 20, "mar"_a = 0.0,
             "start_policy"_a = "strict")
        .def("__call__", &screamer::screamer_call, "value"_a.none())
        .def("reset", &screamer::RollingUpsideDeviation::reset, "Reset.");
```

`tests/param_cases.py`: add `'RollingUpsideDeviation'` to `_ROLLING_AUTO_EXCLUDE` (on the `'RollingDownsideDeviation', ...` line) and to `test_definitions`:
`( ('RollingUpsideDeviation',)  , {"window_size": [20], "mar": [0.0, 0.5], "start_policy": ["strict", "expanding", "zero"]}),`

`devtools/wasm/gen_research_oracle.py` `CASES`:

```python
    ("RollingUpsideDeviation", {"window_size": 20}, "returns"),
    ("RollingUpsideDeviation", {"window_size": 15, "mar": 0.2, "start_policy": "expanding"}, "returns"),
    ("RollingUpsideDeviation", {"window_size": 15, "start_policy": "zero"}, "returns"),
```

`js/test/research.test.ts`:

```typescript
test("research: RollingUpsideDeviation hand example", () => {
  const out = (api as any).RollingUpsideDeviation({ windowSize: 3 })(Float64Array.from([1, -2, 2, 0])) as Float64Array;
  assert.ok(Number.isNaN(out[0]) && Number.isNaN(out[1]));
  assert.ok(Math.abs(out[2] - Math.sqrt(5 / 3)) < 1e-12 && Math.abs(out[3] - Math.sqrt(4 / 3)) < 1e-12);
});
```

`docs/functions_fin/RollingDownsideDeviation.md`: add `- RollingUpsideDeviation` to `see_also`.

`docs/functions_fin/RollingUpsideDeviation.md`:

````markdown
---
name: RollingUpsideDeviation
title: Rolling Upside Deviation
implementation_family: fin
topics:
- risk
- volatility
tags:
- upside deviation
- upside risk
- semideviation
- upside volatility
- gain-loss ratio
short: 'Trailing-window upside semideviation: the RMS of the returns above a minimum acceptable
  return.'
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Number of most recent returns in the window.
- name: mar
  type: float
  default: 0.0
  description: Minimum acceptable return. Only returns above it contribute to the deviation.
- name: start_policy
  type: str
  default: strict
  enum:
  - strict
  - expanding
  - zero
  description: 'Output before the window is full: "strict" gives NaN, "expanding" computes
    from the samples seen so far, and "zero" treats the window as pre-filled with zeros.'
nan_policy: ignore
see_also:
- RollingDownsideDeviation
- RollingStd
- RollingSortino
- RollingOmega
---

# `RollingUpsideDeviation`

## Description

`RollingUpsideDeviation` is the upside semideviation over a trailing window:
the root-mean-square of the excesses above a minimum acceptable return `mar`,

    sqrt( mean( max(return - mar, 0)^2 ) ).

Only returns above `mar` contribute; anything at or below it counts as zero.
It is the mirror of [`RollingDownsideDeviation`](RollingDownsideDeviation.md):
`RollingUpsideDeviation(w, mar)(r)` equals `RollingDownsideDeviation(w, -mar)(-r)`.
The ratio of the two measures the asymmetry of a return distribution, for
example a strategy whose gains are larger than its losses.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input gives a `NaN` output at that step. The step is skipped: it does not enter the window or update the internal state. See [NaN and warmup](../nan_and_warmup.md).
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from screamer import RollingDownsideDeviation, RollingUpsideDeviation

    rng = np.random.default_rng(0)
    n = 500
    # returns with occasional sharp gains (right skew)
    r = rng.standard_normal(n) * 0.01 + (rng.random(n) < 0.05) * 0.03
    up = RollingUpsideDeviation(60)(r)
    down = RollingDownsideDeviation(60)(r)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=up, name='upside deviation', line=dict(color='seagreen')))
    fig.add_trace(go.Scatter(y=down, name='downside deviation', line=dict(color='crimson')))
    fig.update_layout(title='RollingUpsideDeviation: the gains above mar, against the losses below it',
                      xaxis_title='step', yaxis_title='per-step deviation',
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```

<!-- HELP_END -->
````

- [ ] **Step 5: Regenerate and run the tests**

Run: `REGEN`, `.venv/bin/python -m pytest tests/test_rolling_upside_deviation.py -q 2>&1 | tail -5`, `REGISTRY RollingUpsideDeviation`, `FRESH`.
Expected: all pass.

- [ ] **Step 6: Tidy, full suite, JavaScript** (`make tidy`, full suite, `JSTEST`). Expected: clean, all pass.

- [ ] **Step 7: Commit**

```bash
git add include/screamer/rolling_upside_deviation.h bindings/bindings_fin.cpp include/screamer/common/int_settings.h \
  docs/functions_fin/RollingUpsideDeviation.md docs/functions_fin/RollingDownsideDeviation.md \
  devtools/baselines/_windowing.py devtools/baselines/RollingUpsideDeviation.py \
  tests/test_rolling_upside_deviation.py tests/param_cases.py devtools/wasm/gen_research_oracle.py \
  js/test/research.test.ts screamer/__init__.py screamer/data/help.json docs/by_group/backtesting-risk.rst \
  docs/by_group/statistics.rst docs/function_index.txt docs/gallery.rst docs/_static/gallery/manifest.json \
  docs/_static/gallery/functions_fin__RollingUpsideDeviation.webp devtools/wasm/wasm_manifest.json \
  wasm/generated/bindings_wasm.cpp js/src/generated/ops.ts js/src/generated/ops.d.ts \
  js/src/generated/docs_data.ts js/test/docs_oracle.json wasm/smoke/null_oracle.json wasm/smoke/oracle_research.json
git status --short   # add any other path the regeneration changed, by name
git commit -m "feat(risk): RollingUpsideDeviation

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PZMDN2ijvnSZ7ZnrJUxnd8"
```

---

### Task 5: `RollingSeasonalMean`, `RollingSeasonalStd`, `RollingSeasonalZscore`

**Files:**
- Create: `include/screamer/rolling_seasonal.h` (the shared `detail::SeasonalMomentsOp` and the three classes), `docs/functions_rolling/RollingSeasonalMean.md`, `docs/functions_rolling/RollingSeasonalStd.md`, `docs/functions_rolling/RollingSeasonalZscore.md`, `devtools/baselines/RollingSeasonalMean.py`, `devtools/baselines/RollingSeasonalStd.py`, `devtools/baselines/RollingSeasonalZscore.py`, `devtools/baselines/_seasonal.py`, `tests/test_rolling_seasonal.py`
- Modify: `bindings/bindings_rolling.cpp` (include + three classes), `include/screamer/common/int_settings.h` (`{"RollingSeasonalMean", "period", 1}`, `{"RollingSeasonalMean", "window_size", 1}`, `{"RollingSeasonalStd", "period", 1}`, `{"RollingSeasonalStd", "window_size", 2}`, `{"RollingSeasonalZscore", "period", 1}`, `{"RollingSeasonalZscore", "window_size", 2}`), `tests/param_cases.py` (`_ROLLING_AUTO_EXCLUDE` + explicit entry), `devtools/wasm/gen_research_oracle.py`, `js/test/research.test.ts`

**Interfaces:**
- Produces: `RollingSeasonalMean(period=24, window_size=20, start_policy="strict")`, `RollingSeasonalStd(...)`, `RollingSeasonalZscore(...)`, each 1 -> 1; JavaScript `RollingSeasonalMean({ period, windowSize, startPolicy })` and the same for the other two.

**Behaviour (spec B, with the NaN rule of Global Constraints):** step t (counted from the first step, NaN steps included) belongs to slot `t mod period`. Each slot keeps its last `window_size` finite values. Mean, sample standard deviation (ddof 1), and `(x_t - mean) / std` with `x_t` in the window, as `RollingZscore`. A NaN step advances the slot, enters no window, and outputs NaN (`nan_policy: nan-aware`). `"strict"`: NaN until the slot holds `window_size` values; `"expanding"`: the values the slot holds (std and zscore need 2); `"zero"`: each slot starts with `window_size` zeros.

- [ ] **Step 1: Write the baselines**

`devtools/baselines/_seasonal.py`:

```python
"""The slot windows of the seasonal operators: step t belongs to slot
t mod period, and each slot keeps its last window_size finite values. A NaN
step advances the slot and enters no window."""
import numpy as np


def slot_windows(x, period, window_size, start_policy, minimum):
    """For each step, the values of its slot's window (newest last), or None
    where the step is NaN or the policy gives no output."""
    x = np.asarray(x, dtype=float)
    slots = [[0.0] * window_size if start_policy == "zero" else [] for _ in range(period)]
    out = []
    for t, v in enumerate(x):
        s = slots[t % period]
        if np.isnan(v):
            out.append(None)
            continue
        s.append(v)
        if len(s) > window_size:
            s.pop(0)
        need = window_size if start_policy == "strict" else minimum
        out.append(np.array(s) if len(s) >= need else None)
    return out
```

`devtools/baselines/RollingSeasonalMean.py`:

```python
import numpy as np

from ._seasonal import slot_windows


class RollingSeasonalMean_numpy:
    def __init__(self, period=24, window_size=20, start_policy="strict"):
        self.p, self.w, self.sp = period, window_size, start_policy

    def __call__(self, x):
        return np.array([np.nan if w is None else w.mean()
                         for w in slot_windows(x, self.p, self.w, self.sp, 1)])
```

`devtools/baselines/RollingSeasonalStd.py`:

```python
import numpy as np

from ._seasonal import slot_windows


class RollingSeasonalStd_numpy:
    def __init__(self, period=24, window_size=20, start_policy="strict"):
        self.p, self.w, self.sp = period, window_size, start_policy

    def __call__(self, x):
        return np.array([np.nan if w is None else w.std(ddof=1)
                         for w in slot_windows(x, self.p, self.w, self.sp, 2)])
```

`devtools/baselines/RollingSeasonalZscore.py`:

```python
import numpy as np

from ._seasonal import slot_windows


class RollingSeasonalZscore_numpy:
    def __init__(self, period=24, window_size=20, start_policy="strict"):
        self.p, self.w, self.sp = period, window_size, start_policy

    def __call__(self, x):
        return np.array([np.nan if w is None else (w[-1] - w.mean()) / w.std(ddof=1)
                         for w in slot_windows(x, self.p, self.w, self.sp, 2)])
```

- [ ] **Step 2: Write the failing tests** (`tests/test_rolling_seasonal.py`)

```python
import math

import numpy as np
import pytest

from devtools.baselines.RollingSeasonalMean import RollingSeasonalMean_numpy
from devtools.baselines.RollingSeasonalStd import RollingSeasonalStd_numpy
from devtools.baselines.RollingSeasonalZscore import RollingSeasonalZscore_numpy
from screamer import RollingSeasonalMean, RollingSeasonalStd, RollingSeasonalZscore, RollingZscore
from tests.regime_helpers import assert_regimes_agree

X = np.array([1.0, 10.0, 3.0, 20.0, 5.0, 30.0])
S2 = math.sqrt(2.0)
S50 = math.sqrt(50.0)


def test_mean_hand_example():
    out = np.asarray(RollingSeasonalMean(period=2, window_size=2)(X))
    assert np.isnan(out[:2]).all()
    np.testing.assert_allclose(out[2:], [2.0, 15.0, 4.0, 25.0], rtol=1e-12)


def test_std_and_zscore_hand_example():
    sd = np.asarray(RollingSeasonalStd(period=2, window_size=2)(X))
    np.testing.assert_allclose(sd[2:], [S2, S50, S2, S50], rtol=1e-12)
    z = np.asarray(RollingSeasonalZscore(period=2, window_size=2)(X))
    np.testing.assert_allclose(z[2:], [1 / S2] * 4, rtol=1e-12)


def test_expanding_and_zero():
    exp = np.asarray(RollingSeasonalMean(period=2, window_size=2, start_policy="expanding")(X))
    np.testing.assert_allclose(exp, [1.0, 10.0, 2.0, 15.0, 4.0, 25.0], rtol=1e-12)
    sd = np.asarray(RollingSeasonalStd(period=2, window_size=2, start_policy="expanding")(X))
    assert np.isnan(sd[:2]).all()
    zero = np.asarray(RollingSeasonalMean(period=2, window_size=2, start_policy="zero")(X))
    np.testing.assert_allclose(zero[:3], [0.5, 5.0, 2.0], rtol=1e-12)


def test_nan_advances_the_slot_clock():
    x = np.array([1.0, 10.0, np.nan, 20.0, 5.0, 30.0])
    out = np.asarray(RollingSeasonalMean(period=2, window_size=2, start_policy="expanding")(x))
    assert math.isnan(out[2])
    np.testing.assert_allclose(out[[0, 1, 3, 4, 5]], [1.0, 10.0, 15.0, 3.0, 25.0], rtol=1e-12)


def test_period_one_is_the_rolling_zscore():
    x = np.random.default_rng(0).standard_normal(200)
    np.testing.assert_allclose(np.asarray(RollingSeasonalZscore(period=1, window_size=20)(x)),
                               np.asarray(RollingZscore(20)(x)), rtol=1e-9, equal_nan=True)


@pytest.mark.parametrize("cls,ref", [(RollingSeasonalMean, RollingSeasonalMean_numpy),
                                     (RollingSeasonalStd, RollingSeasonalStd_numpy),
                                     (RollingSeasonalZscore, RollingSeasonalZscore_numpy)])
@pytest.mark.parametrize("start_policy", ["strict", "expanding", "zero"])
def test_matches_baseline(cls, ref, start_policy):
    rng = np.random.default_rng(3)
    t = np.arange(24 * 40)
    x = 1000 + 5 * np.sin(2 * np.pi * t / 24) + rng.standard_normal(len(t))
    x[[7, 300, 301, 650]] = np.nan
    got = np.asarray(cls(period=24, window_size=10, start_policy=start_policy)(x))
    want = ref(24, 10, start_policy)(x)
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9, equal_nan=True)


@pytest.mark.parametrize("cls", [RollingSeasonalMean, RollingSeasonalStd, RollingSeasonalZscore])
def test_regimes_agree(cls):
    x = np.random.default_rng(4).standard_normal(300)
    x[[3, 150]] = np.nan
    assert_regimes_agree(lambda: cls(period=7, window_size=5), x)


def test_settings_are_checked():
    with pytest.raises(ValueError):
        RollingSeasonalMean(period=0)
    with pytest.raises(ValueError):
        RollingSeasonalStd(window_size=1)
    RollingSeasonalMean(window_size=1)            # one value has a mean
```

- [ ] **Step 3: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_rolling_seasonal.py -q 2>&1 | tail -5`
Expected: FAIL (`ImportError`).

- [ ] **Step 4: Implement** `include/screamer/rolling_seasonal.h`

```cpp
#ifndef SCREAMER_ROLLING_SEASONAL_H
#define SCREAMER_ROLLING_SEASONAL_H

// RollingSeasonalMean, RollingSeasonalStd, RollingSeasonalZscore: the mean,
// sample standard deviation (ddof 1) and z-score of the same slot of the last
// `window_size` cycles, x_t, x_{t - period}, ..., x_{t - (window_size - 1)
// period}. Step t (counted from the first step) belongs to slot t mod period,
// so the input must be on a regular clock. Each slot keeps its own window
// (detail::ShiftedWindowSums<2>, sums about an origin near the slot mean).
// The z-score includes x_t in its window, as RollingZscore does.
//
// start_policy: "strict" gives NaN until the slot holds window_size values,
// "expanding" uses the values the slot holds (the standard deviation needs
// two), and "zero" starts each slot with window_size zeros.
//
// nan_policy: nan-aware. A NaN step advances the slot clock, enters no
// window, and gives NaN, so a missing value on the clock does not shift the
// later values into the wrong slots. O(1) per step; memory period *
// window_size values.

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "screamer/common/base.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/moment_formulas.h"
#include "screamer/detail/shifted_window_sums.h"
#include "screamer/detail/start_policy.h"

namespace screamer {
namespace detail {

template <class Derived>
class SeasonalMomentsOp : public ScreamerBase {
protected:
    using Sums = ShiftedWindowSums<2>;
    using State = Sums::State;

    SeasonalMomentsOp(int period, int window_size, const std::string& start_policy,
                      int min_window)
        : window_size_(window_size),
          start_policy_(parse_start_policy(start_policy))
    {
        if (period < 1) throw std::invalid_argument("period must be 1 or more.");
        if (window_size < min_window)
            throw std::invalid_argument(min_window == 1 ? "Window size must be 1 or more."
                                                        : "Window size must be 2 or more.");
        slots_.reserve(static_cast<std::size_t>(period));
        for (int i = 0; i < period; ++i)
            slots_.emplace_back(static_cast<std::size_t>(window_size), start_policy_);
        reset();
    }

public:
    void reset() override {
        for (Sums& s : slots_) s.reset();
        slot_ = 0;
    }

    double process_scalar(double x) override {
        Sums& s = slots_[slot_];
        if (++slot_ == slots_.size()) slot_ = 0;
        if (isnan2(x)) return std::numeric_limits<double>::quiet_NaN();
        Sums::Cursor c = s.cursor();
        c.push(x);
        s.set_state(c.st);
        return static_cast<const Derived&>(*this).value(c.st, x);
    }

protected:
    bool ready(const State& st, std::size_t minimum) const {
        if (start_policy_ == StartPolicy::Strict)
            return st.count == static_cast<std::size_t>(window_size_);
        return st.count >= minimum;
    }

    const int window_size_;
    const StartPolicy start_policy_;
    std::vector<Sums> slots_;
    std::size_t slot_ = 0;
};

}  // namespace detail

class RollingSeasonalMean : public detail::SeasonalMomentsOp<RollingSeasonalMean> {
public:
    RollingSeasonalMean(int period = 24, int window_size = 20,
                        const std::string& start_policy = "strict")
        : SeasonalMomentsOp(period, window_size, start_policy, 1) {}

    double value(const State& st, double) const {
        if (!ready(st, 1)) return std::numeric_limits<double>::quiet_NaN();
        return st.origin + st.s[0] / static_cast<double>(st.count);
    }
};

class RollingSeasonalStd : public detail::SeasonalMomentsOp<RollingSeasonalStd> {
public:
    RollingSeasonalStd(int period = 24, int window_size = 20,
                       const std::string& start_policy = "strict")
        : SeasonalMomentsOp(period, window_size, start_policy, 2) {}

    double value(const State& st, double) const {
        if (!ready(st, 2)) return std::numeric_limits<double>::quiet_NaN();
        const double n = static_cast<double>(st.count);
        return std::sqrt(detail::var_from_sums(st.s[0], st.s[1], n));
    }
};

class RollingSeasonalZscore : public detail::SeasonalMomentsOp<RollingSeasonalZscore> {
public:
    RollingSeasonalZscore(int period = 24, int window_size = 20,
                          const std::string& start_policy = "strict")
        : SeasonalMomentsOp(period, window_size, start_policy, 2) {}

    double value(const State& st, double x) const {
        if (!ready(st, 2)) return std::numeric_limits<double>::quiet_NaN();
        const double n = static_cast<double>(st.count);
        const double var = detail::var_from_sums(st.s[0], st.s[1], n);
        const double mean_d = st.s[0] / n;
        return ((x - st.origin) - mean_d) / std::sqrt(var);
    }
};

}  // namespace screamer

#endif  // SCREAMER_ROLLING_SEASONAL_H
```

`SeasonalMomentsOp`'s constructor validates before the slots are built (a negative `period` must raise, not allocate); `reset()` is called from the base constructor, which only touches the base's members.

`bindings/bindings_rolling.cpp`: `#include "screamer/rolling_seasonal.h"` and

```cpp
    // Seasonal statistics: the same slot of the last window_size cycles.
    nb::class_<screamer::RollingSeasonalMean, screamer::ScreamerBase>(m, "RollingSeasonalMean")
        .def(nb::init<screamer::IntArg<int>, screamer::IntArg<int>, const std::string&>(),
             "period"_a = 24, "window_size"_a = 20, "start_policy"_a = "strict")
        .def("__call__", &screamer::screamer_call, "value"_a.none())
        .def("reset", &screamer::RollingSeasonalMean::reset, "Reset.");
    nb::class_<screamer::RollingSeasonalStd, screamer::ScreamerBase>(m, "RollingSeasonalStd")
        .def(nb::init<screamer::IntArg<int>, screamer::IntArg<int>, const std::string&>(),
             "period"_a = 24, "window_size"_a = 20, "start_policy"_a = "strict")
        .def("__call__", &screamer::screamer_call, "value"_a.none())
        .def("reset", &screamer::RollingSeasonalStd::reset, "Reset.");
    nb::class_<screamer::RollingSeasonalZscore, screamer::ScreamerBase>(m, "RollingSeasonalZscore")
        .def(nb::init<screamer::IntArg<int>, screamer::IntArg<int>, const std::string&>(),
             "period"_a = 24, "window_size"_a = 20, "start_policy"_a = "strict")
        .def("__call__", &screamer::screamer_call, "value"_a.none())
        .def("reset", &screamer::RollingSeasonalZscore::reset, "Reset.");
```

`tests/param_cases.py`: add the three names to `_ROLLING_AUTO_EXCLUDE` (comment: `# period x window warm-up; explicit entry below`) and to `test_definitions`:
`( ('RollingSeasonalMean','RollingSeasonalStd','RollingSeasonalZscore'), {"period": [5], "window_size": [4], "start_policy": ["strict", "expanding", "zero"], "array_length": [200]}),`

`devtools/wasm/gen_research_oracle.py` `CASES`:

```python
    ("RollingSeasonalMean", {"period": 24, "window_size": 5}, "returns"),
    ("RollingSeasonalStd", {"period": 24, "window_size": 5, "start_policy": "expanding"}, "returns"),
    ("RollingSeasonalZscore", {"period": 7, "window_size": 6, "start_policy": "zero"}, "returns"),
```

`js/test/research.test.ts`:

```typescript
test("research: RollingSeasonalMean NaN advances the slot clock", () => {
  const out = (api as any).RollingSeasonalMean({ period: 2, windowSize: 2, startPolicy: "expanding" })(
    Float64Array.from([1, 10, NaN, 20, 5, 30])) as Float64Array;
  assert.ok(Number.isNaN(out[2]));
  assert.deepEqual([out[0], out[1], out[3], out[4], out[5]], [1, 10, 15, 3, 25]);
});
```

`docs/functions_rolling/RollingSeasonalMean.md` (the Std and Zscore pages are the same with the changes listed below):

````markdown
---
name: RollingSeasonalMean
title: Rolling Seasonal Mean
implementation_family: rolling
topics:
- statistics
tags:
- seasonal
- seasonality
- intraday
- time of day
- day of week
- profile
- slot
short: Mean of the same slot of the last `window_size` cycles, such as the same hour on each
  of the last 20 days.
inputs: 1
outputs: 1
parameters:
- name: period
  type: int
  default: 24
  min: 1
  description: Steps per cycle, for example 24 for hourly data with a daily cycle.
- name: window_size
  type: int
  default: 20
  min: 1
  description: Number of cycles in the window.
- name: start_policy
  type: str
  default: strict
  enum:
  - strict
  - expanding
  - zero
  description: 'Output before a slot holds window_size values: "strict" gives NaN, "expanding"
    uses the values the slot holds, and "zero" starts each slot with window_size zeros.'
nan_policy: nan-aware
see_also:
- RollingSeasonalStd
- RollingSeasonalZscore
- RollingMean
---

# `RollingSeasonalMean`

## Description

`RollingSeasonalMean` is the mean of one slot of a cycle over the last
`window_size` cycles:

$$
\text{mean}_t = \frac{1}{w} \sum_{j=0}^{w-1} x_{t - j \cdot \text{period}}
$$

Step `t` belongs to slot `t mod period`, counted in steps from the first input,
so the input must be on a regular clock: one value per hour for `period=24`,
for example. On hourly volume it gives the typical volume of the current hour
of the day over the last `window_size` days.

A `NaN` step advances the slot clock and enters no window, and its output is
`NaN`: a missing hour is treated as missing, and the hours after it stay in
their own slots. With `period=2, window_size=2` and `start_policy="expanding"`,
the input `[1, 10, NaN, 20, 5, 30]` gives `[1, 10, NaN, 15, 3, 25]`.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `nan-aware`.** This function is built to process `NaN` inputs, as the Description explains. See [NaN and warmup](../nan_and_warmup.md).
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from screamer import RollingSeasonalMean

    rng = np.random.default_rng(1)
    hours = np.arange(24 * 30)
    # hourly volume with a daily profile: busy around hour 14
    profile = 100 + 80 * np.exp(-0.5 * ((hours % 24 - 14) / 3.0) ** 2)
    volume = profile * rng.lognormal(0.0, 0.3, len(hours))
    typical = RollingSeasonalMean(period=24, window_size=10)(volume)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=volume, name='volume', line=dict(color='lightgray')))
    fig.add_trace(go.Scatter(y=typical, name='same hour, last 10 days', line=dict(color='navy')))
    fig.update_layout(title='RollingSeasonalMean: the typical value of each hour of the day',
                      xaxis_title='hour', yaxis_title='volume',
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```

<!-- HELP_END -->
````

`RollingSeasonalStd.md`: `name`/`title` "Rolling Seasonal Standard Deviation", `topics: [statistics, volatility]`, `short: Sample standard deviation of the same slot of the last window_size cycles.`, `window_size` `min: 2`, `see_also: [RollingSeasonalMean, RollingSeasonalZscore, RollingStd]`; the formula is the sample standard deviation (ddof 1) of the same values; "expanding" needs two values; the plot shows `RollingSeasonalStd(period=24, window_size=10)` of the same volume. `RollingSeasonalZscore.md`: title "Rolling Seasonal Z-score", `topics: [standardization]`, `short: (x - mean) / std over the same slot of the last window_size cycles, x included.`, `window_size` `min: 2`, `see_also: [RollingSeasonalMean, RollingSeasonalStd, RollingZscore]`; the formula `(x_t - mean_t) / std_t` with `x_t` in the window, as `RollingZscore`; with `period=1` it equals `RollingZscore(window_size)`; the plot shows the z-score of the same volume with a horizontal line at 2. Both carry the same NaN paragraph and `nan_policy: nan-aware`.

- [ ] **Step 5: Regenerate and run the tests**

Run: `REGEN`, `.venv/bin/python -m pytest tests/test_rolling_seasonal.py -q 2>&1 | tail -5`, `REGISTRY RollingSeasonal`, `FRESH`.
Expected: all pass.

- [ ] **Step 6: Tidy, full suite, JavaScript** (`make tidy`, full suite, `JSTEST`). Expected: clean, all pass.

- [ ] **Step 7: Commit**

```bash
git add include/screamer/rolling_seasonal.h bindings/bindings_rolling.cpp include/screamer/common/int_settings.h \
  docs/functions_rolling/RollingSeasonalMean.md docs/functions_rolling/RollingSeasonalStd.md \
  docs/functions_rolling/RollingSeasonalZscore.md devtools/baselines/_seasonal.py \
  devtools/baselines/RollingSeasonalMean.py devtools/baselines/RollingSeasonalStd.py \
  devtools/baselines/RollingSeasonalZscore.py tests/test_rolling_seasonal.py tests/param_cases.py \
  devtools/wasm/gen_research_oracle.py js/test/research.test.ts screamer/__init__.py screamer/data/help.json \
  docs/by_group/statistics.rst docs/function_index.txt docs/gallery.rst docs/_static/gallery/manifest.json \
  docs/_static/gallery/functions_rolling__RollingSeasonalMean.webp \
  docs/_static/gallery/functions_rolling__RollingSeasonalStd.webp \
  docs/_static/gallery/functions_rolling__RollingSeasonalZscore.webp devtools/wasm/wasm_manifest.json \
  wasm/generated/bindings_wasm.cpp js/src/generated/ops.ts js/src/generated/ops.d.ts \
  js/src/generated/docs_data.ts js/test/docs_oracle.json wasm/smoke/null_oracle.json \
  wasm/smoke/oracle_research.json
git status --short   # add any other path the regeneration changed, by name
git commit -m "feat(rolling): RollingSeasonalMean, RollingSeasonalStd, RollingSeasonalZscore

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PZMDN2ijvnSZ7ZnrJUxnd8"
```

---

### Task 6: `RollingBipowerVar` and `RollingJumpShare`

**Files:**
- Create: `include/screamer/rolling_bipower.h` (the shared `detail::BipowerWindow` and both classes), `docs/functions_rolling/RollingBipowerVar.md`, `docs/functions_rolling/RollingJumpShare.md`, `devtools/baselines/_bipower.py`, `devtools/baselines/RollingBipowerVar.py`, `devtools/baselines/RollingJumpShare.py`, `tests/test_rolling_bipower.py`
- Modify: `bindings/bindings_rolling.cpp`, `include/screamer/common/int_settings.h` (`{"RollingBipowerVar", "window_size", 2}`, `{"RollingJumpShare", "window_size", 2}`), `tests/param_cases.py` (`_ROLLING_AUTO_EXCLUDE` + explicit entry), `devtools/wasm/gen_research_oracle.py`, `js/test/research.test.ts`

**Interfaces:**
- Produces: `RollingBipowerVar(window_size=20, start_policy="strict")` and `RollingJumpShare(window_size=20, start_policy="strict")`, 1 -> 1; JavaScript `RollingBipowerVar({ windowSize, startPolicy })`, `RollingJumpShare({ ... })`.

**Behaviour (spec B):** input returns. With `n` returns in the window (`n = window_size` once full; fewer with `"expanding"`), `BV = (pi / 2) * sum over the n - 1 adjacent pairs of |r_i| |r_{i-1}| * n / (n - 1)` and `RV = sum r_i^2`. `RollingJumpShare = max(0, 1 - BV / RV)`, 0 when `RV == 0`. A pair spans a NaN step (the NaN is not in the window). `"expanding"` needs two returns. `"zero"`: window of zeros before the input (the first return pairs with a 0).

- [ ] **Step 1: Write the baselines**

`devtools/baselines/_bipower.py`:

```python
"""Bipower and realized variance of trailing windows of returns."""
import math

import numpy as np

from ._windowing import ignoring_nan, trailing


def bipower_and_realized(x, window, start_policy):
    """(BV, RV) per step, NaN where the policy gives no output."""
    def run(v):
        bv, rv = [], []
        for w in trailing(v, window, start_policy, minimum=2):
            if w is None:
                bv.append(np.nan)
                rv.append(np.nan)
                continue
            n = len(w)
            bv.append(math.pi / 2 * np.sum(np.abs(w[1:]) * np.abs(w[:-1])) * n / (n - 1))
            rv.append(np.sum(w * w))
        return np.column_stack([bv, rv])
    return ignoring_nan(x, run)
```

`devtools/baselines/RollingBipowerVar.py`:

```python
from ._bipower import bipower_and_realized


class RollingBipowerVar_numpy:
    def __init__(self, window_size=20, start_policy="strict"):
        self.w, self.sp = window_size, start_policy

    def __call__(self, x):
        return bipower_and_realized(x, self.w, self.sp)[:, 0]
```

`devtools/baselines/RollingJumpShare.py`:

```python
import numpy as np

from ._bipower import bipower_and_realized


class RollingJumpShare_numpy:
    def __init__(self, window_size=20, start_policy="strict"):
        self.w, self.sp = window_size, start_policy

    def __call__(self, x):
        bv, rv = bipower_and_realized(x, self.w, self.sp).T
        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.maximum(0.0, 1.0 - bv / rv)
        share[rv == 0.0] = 0.0
        share[np.isnan(bv) | np.isnan(rv)] = np.nan
        return share
```

- [ ] **Step 2: Write the failing tests** (`tests/test_rolling_bipower.py`)

```python
import math

import numpy as np
import pytest

from devtools.baselines.RollingBipowerVar import RollingBipowerVar_numpy
from devtools.baselines.RollingJumpShare import RollingJumpShare_numpy
from screamer import RollingBipowerVar, RollingJumpShare
from tests.regime_helpers import assert_regimes_agree


def test_bipower_hand_example():
    out = np.asarray(RollingBipowerVar(3)(np.array([1.0, -2.0, 3.0, -1.0])))
    assert np.isnan(out[:2]).all()
    np.testing.assert_allclose(out[2:], [6 * math.pi, 6.75 * math.pi], rtol=1e-12)


def test_jump_share_hand_example():
    out = np.asarray(RollingJumpShare(4)(np.array([1.0, 1.0, 1.0, 10.0])))
    np.testing.assert_allclose(out[3], 1 - 8 * math.pi / 103, rtol=1e-12)
    smooth = np.asarray(RollingJumpShare(3)(np.array([1.0, -2.0, 3.0])))
    assert smooth[2] == 0.0                      # BV above RV: no jump


def test_jump_share_of_zero_returns_is_zero():
    out = np.asarray(RollingJumpShare(5)(np.zeros(10)))
    np.testing.assert_array_equal(out[4:], 0.0)


def test_expanding_needs_two_returns_and_zero_pairs_with_zero():
    exp = np.asarray(RollingBipowerVar(3, start_policy="expanding")(np.array([2.0, 3.0])))
    assert math.isnan(exp[0])
    np.testing.assert_allclose(exp[1], math.pi / 2 * 6 * 2, rtol=1e-12)
    zero = np.asarray(RollingBipowerVar(3, start_policy="zero")(np.array([2.0, 3.0])))
    np.testing.assert_allclose(zero, [0.0, math.pi / 2 * 6 * 1.5], rtol=1e-12)


def test_bipower_estimates_the_variance_without_jumps():
    r = np.random.default_rng(0).standard_normal(200_000) * 0.01
    bv = np.asarray(RollingBipowerVar(1000)(r))
    rv = np.convolve(r * r, np.ones(1000), mode="valid")
    assert abs(np.nanmean(bv[999:]) / rv.mean() - 1.0) < 0.01


@pytest.mark.parametrize("cls,ref", [(RollingBipowerVar, RollingBipowerVar_numpy),
                                     (RollingJumpShare, RollingJumpShare_numpy)])
@pytest.mark.parametrize("start_policy", ["strict", "expanding", "zero"])
def test_matches_baseline(cls, ref, start_policy):
    rng = np.random.default_rng(5)
    r = rng.standard_normal(500) * 0.01
    r[rng.random(500) < 0.02] *= 15                # jumps
    r[[9, 250]] = np.nan
    got = np.asarray(cls(30, start_policy=start_policy)(r))
    np.testing.assert_allclose(got, ref(30, start_policy)(r), rtol=1e-9, atol=1e-15, equal_nan=True)


@pytest.mark.parametrize("cls", [RollingBipowerVar, RollingJumpShare])
def test_regimes_agree(cls):
    r = np.random.default_rng(6).standard_normal(300)
    r[40] = np.nan
    assert_regimes_agree(lambda: cls(25), r)
```

Hand numbers: window `[1, -2, 3]` has pairs `2, 6`, so `BV = pi/2 * 8 * 3/2 = 6 pi`; `[-2, 3, -1]` has `6, 3`, so `6.75 pi`; `[1, 1, 1, 10]` has pairs `1, 1, 10`, `BV = pi/2 * 12 * 4/3 = 8 pi` and `RV = 103`. Expanding `[2, 3]`: one pair `6`, `n = 2`. Zero `[0, 0, 2]` then `[0, 2, 3]`: pairs `0, 0` then `0, 6`.

- [ ] **Step 3: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_rolling_bipower.py -q 2>&1 | tail -5`
Expected: FAIL (`ImportError`).

- [ ] **Step 4: Implement** `include/screamer/rolling_bipower.h`

```cpp
#ifndef SCREAMER_ROLLING_BIPOWER_H
#define SCREAMER_ROLLING_BIPOWER_H

// RollingBipowerVar and RollingJumpShare (Barndorff-Nielsen and Shephard
// 2004). On a window of n returns (n = window_size once full),
//     BV = (pi / 2) * sum_{i=2..n} |r_i| |r_{i-1}| * n / (n - 1)
//     RV = sum_{i=1..n} r_i^2
// BV estimates the variance of the continuous part of the returns and RV the
// whole variance, jumps included; the factor n / (n - 1) scales the n - 1
// pairs to the n terms of RV. RollingJumpShare = max(0, 1 - BV / RV), the
// share of RV that comes from jumps, and 0 when RV is 0.
//
// start_policy: "strict" gives NaN until the window holds window_size
// returns, "expanding" uses the returns seen so far (two at least), "zero"
// starts from a window of zeros. nan_policy: ignore (a NaN is not stored;
// the pair after it joins the returns on either side). O(1) per step.

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include "screamer/common/base.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/rolling_sum.h"
#include "screamer/detail/start_policy.h"

namespace screamer {
namespace detail {

class BipowerWindow {
public:
    BipowerWindow(int window_size, const std::string& start_policy)
        : window_size_(window_size),
          policy_(parse_start_policy(start_policy)),
          pairs_(window_size >= 2 ? static_cast<std::size_t>(window_size - 1) : 1, start_policy),
          squares_(window_size >= 2 ? static_cast<std::size_t>(window_size) : 2, start_policy)
    {
        if (window_size < 2) throw std::invalid_argument("Window size must be 2 or more.");
        reset();
    }

    void reset() {
        pairs_.reset();
        squares_.reset();
        const bool zero = policy_ == StartPolicy::Zero;
        n_ = zero ? static_cast<std::size_t>(window_size_) : 0;
        has_prev_ = zero;               // the zero window's last return is 0
        prev_abs_ = 0.0;
    }

    // Adds a finite return; bv and rv are NaN where the policy gives none.
    void push(double r, double& bv, double& rv) {
        constexpr double kHalfPi = 1.5707963267948966;
        const double nan = std::numeric_limits<double>::quiet_NaN();
        if (n_ < static_cast<std::size_t>(window_size_)) ++n_;
        rv = squares_.append(r * r);
        const double a = std::fabs(r);
        const double pair_sum = has_prev_ ? pairs_.append(prev_abs_ * a) : nan;
        prev_abs_ = a;
        has_prev_ = true;
        if (n_ < 2) {
            bv = nan;
            return;
        }
        const double n = static_cast<double>(n_);
        const double scale = kHalfPi * n / (n - 1.0);
        bv = pair_sum * scale;
    }

private:
    const int window_size_;
    const StartPolicy policy_;
    RollingSum pairs_, squares_;
    std::size_t n_ = 0;
    bool has_prev_ = false;
    double prev_abs_ = 0.0;
};

}  // namespace detail

class RollingBipowerVar : public ScreamerBase {
public:
    RollingBipowerVar(int window_size = 20, const std::string& start_policy = "strict")
        : window_(window_size, start_policy) {}

    void reset() override { window_.reset(); }

    double process_scalar(double x) override {
        if (isnan2(x)) return std::numeric_limits<double>::quiet_NaN();
        double bv = 0.0, rv = 0.0;
        window_.push(x, bv, rv);
        return bv;
    }

private:
    detail::BipowerWindow window_;
};

class RollingJumpShare : public ScreamerBase {
public:
    RollingJumpShare(int window_size = 20, const std::string& start_policy = "strict")
        : window_(window_size, start_policy) {}

    void reset() override { window_.reset(); }

    double process_scalar(double x) override {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        if (isnan2(x)) return nan;
        double bv = 0.0, rv = 0.0;
        window_.push(x, bv, rv);
        if (isnan2(bv) || isnan2(rv)) return nan;
        if (rv == 0.0) return 0.0;
        const double share = 1.0 - bv / rv;
        return share > 0.0 ? share : 0.0;
    }

private:
    detail::BipowerWindow window_;
};

}  // namespace screamer

#endif  // SCREAMER_ROLLING_BIPOWER_H
```

The strict `RollingSum` of the `window_size - 1` pairs is full on the same step as the `window_size` squares, so both give their first value together.

`bindings/bindings_rolling.cpp`: `#include "screamer/rolling_bipower.h"` and

```cpp
    nb::class_<screamer::RollingBipowerVar, screamer::ScreamerBase>(m, "RollingBipowerVar")
        .def(nb::init<screamer::IntArg<int>, const std::string&>(),
             "window_size"_a = 20, "start_policy"_a = "strict")
        .def("__call__", &screamer::screamer_call, "value"_a.none())
        .def("reset", &screamer::RollingBipowerVar::reset, "Reset.");
    nb::class_<screamer::RollingJumpShare, screamer::ScreamerBase>(m, "RollingJumpShare")
        .def(nb::init<screamer::IntArg<int>, const std::string&>(),
             "window_size"_a = 20, "start_policy"_a = "strict")
        .def("__call__", &screamer::screamer_call, "value"_a.none())
        .def("reset", &screamer::RollingJumpShare::reset, "Reset.");
```

`tests/param_cases.py`: add both names to `_ROLLING_AUTO_EXCLUDE` and to `test_definitions`:
`( ('RollingBipowerVar','RollingJumpShare'), {"window_size": [20], "start_policy": ["strict", "expanding", "zero"]}),`

`devtools/wasm/gen_research_oracle.py` `CASES`:

```python
    ("RollingBipowerVar", {"window_size": 30}, "returns"),
    ("RollingJumpShare", {"window_size": 30, "start_policy": "expanding"}, "returns"),
    ("RollingJumpShare", {"window_size": 12, "start_policy": "zero"}, "returns"),
```

`js/test/research.test.ts`:

```typescript
test("research: RollingJumpShare hand example", () => {
  const out = (api as any).RollingJumpShare({ windowSize: 4 })(Float64Array.from([1, 1, 1, 10])) as Float64Array;
  assert.ok(Math.abs(out[3] - (1 - 8 * Math.PI / 103)) < 1e-12);
});
```

`docs/functions_rolling/RollingBipowerVar.md`:

````markdown
---
name: RollingBipowerVar
title: Rolling Bipower Variation
implementation_family: rolling
topics:
- volatility
tags:
- bipower variation
- realized variance
- jumps
- jump-robust volatility
- barndorff-nielsen
- shephard
short: Jump-robust variance of a window of returns, (pi/2) times the sum of products of adjacent
  absolute returns, on the scale of the realized variance.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Number of most recent returns in the window.
- name: start_policy
  type: str
  default: strict
  enum:
  - strict
  - expanding
  - zero
  description: 'Output before the window is full: "strict" gives NaN, "expanding" uses the
    returns seen so far (at least two), and "zero" treats the window as pre-filled with zeros.'
nan_policy: ignore
see_also:
- RollingJumpShare
- RollingVar
- RollingRms
references:
- 'Barndorff-Nielsen, O. E. and Shephard, N. (2004). Power and bipower variation with stochastic
  volatility and jumps. *Journal of Financial Econometrics* 2(1), 1-37.'
---

# `RollingBipowerVar`

## Description

`RollingBipowerVar` is the bipower variation of a window of `n` returns
(Barndorff-Nielsen and Shephard 2004):

$$
BV = \frac{\pi}{2} \sum_{i=2}^{n} |r_i|\,|r_{i-1}| \cdot \frac{n}{n-1}
$$

The input is returns. The realized variance `RV = sum r_i^2` counts a jump at
its full size squared; in `BV` a jump is multiplied by its neighbours, which
are of normal size, so `BV` estimates the variance of the continuous part of
the returns. The factor `n / (n - 1)` scales the `n - 1` pairs to the `n` terms
of `RV`, so without jumps `BV` and `RV` estimate the same quantity.
[`RollingJumpShare`](RollingJumpShare.md) compares the two.

A `NaN` return is left out, and the returns on either side of it form a pair.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input gives a `NaN` output at that step. The step is skipped: it does not enter the window or update the internal state. See [NaN and warmup](../nan_and_warmup.md).
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from screamer import RollingBipowerVar, RollingSum

    rng = np.random.default_rng(2)
    r = rng.standard_normal(600) * 0.01
    r[[150, 300, 450]] = [0.08, -0.07, 0.09]            # three jumps
    bv = RollingBipowerVar(60)(r)
    rv = RollingSum(60)(r * r)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=rv, name='realized variance', line=dict(color='gray')))
    fig.add_trace(go.Scatter(y=bv, name='bipower variation', line=dict(color='navy')))
    fig.update_layout(title='RollingBipowerVar: the jumps raise RV, not BV',
                      xaxis_title='step', yaxis_title='variance over 60 steps',
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```

<!-- HELP_END -->
````

`docs/functions_rolling/RollingJumpShare.md`: frontmatter as above with `name: RollingJumpShare`, `title: Rolling Jump Share`, `topics: [volatility, regime]`, tags `jumps, jump share, jump test, bipower variation, realized variance`, `short: 'Share of the realized variance of a window of returns that comes from jumps: max(0, 1 - BV / RV).'`, `see_also: [RollingBipowerVar, RollingVar, RollingVarianceRatio]`, the same `window_size` (`min: 2`), `start_policy` and reference. Description: "`RollingJumpShare` is `max(0, 1 - BV / RV)`, with `BV` the bipower variation of [`RollingBipowerVar`](RollingBipowerVar.md) and `RV = sum r_i^2` the realized variance of the same window. It is the share of the window's variance that comes from jumps (Barndorff-Nielsen and Shephard 2004): near 0 for diffusive returns and near 1 when a few returns carry most of the variance. On a window of returns `[1, 1, 1, 10]`, `BV = 8 pi`, `RV = 103`, and the share is `0.756`. It is 0 when `RV` is 0, and 0 when `BV` exceeds `RV`, which sampling noise causes on windows without jumps." The plot: the returns of the `RollingBipowerVar` example and `RollingJumpShare(60)` on a second row (`make_subplots(rows=2, shared_xaxes=True)`), title `'RollingJumpShare: the share of variance from jumps'`.

- [ ] **Step 5: Regenerate and run the tests**

Run: `REGEN`, `.venv/bin/python -m pytest tests/test_rolling_bipower.py -q 2>&1 | tail -5`, `REGISTRY RollingBipowerVar`, `REGISTRY RollingJumpShare`, `FRESH`.
Expected: all pass.

- [ ] **Step 6: Tidy, full suite, JavaScript** (`make tidy`, full suite, `JSTEST`). Expected: clean, all pass.

- [ ] **Step 7: Commit**

```bash
git add include/screamer/rolling_bipower.h bindings/bindings_rolling.cpp include/screamer/common/int_settings.h \
  docs/functions_rolling/RollingBipowerVar.md docs/functions_rolling/RollingJumpShare.md \
  devtools/baselines/_bipower.py devtools/baselines/RollingBipowerVar.py devtools/baselines/RollingJumpShare.py \
  tests/test_rolling_bipower.py tests/param_cases.py devtools/wasm/gen_research_oracle.py js/test/research.test.ts \
  screamer/__init__.py screamer/data/help.json docs/by_group/statistics.rst docs/by_group/machine-learning.rst \
  docs/function_index.txt docs/gallery.rst docs/_static/gallery/manifest.json \
  docs/_static/gallery/functions_rolling__RollingBipowerVar.webp \
  docs/_static/gallery/functions_rolling__RollingJumpShare.webp devtools/wasm/wasm_manifest.json \
  wasm/generated/bindings_wasm.cpp js/src/generated/ops.ts js/src/generated/ops.d.ts \
  js/src/generated/docs_data.ts js/test/docs_oracle.json wasm/smoke/null_oracle.json wasm/smoke/oracle_research.json
git status --short   # add any other path the regeneration changed, by name
git commit -m "feat(rolling): RollingBipowerVar and RollingJumpShare

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PZMDN2ijvnSZ7ZnrJUxnd8"
```

---

### Task 7: `RollingVarianceRatio`

**Files:**
- Create: `include/screamer/rolling_variance_ratio.h`, `docs/functions_rolling/RollingVarianceRatio.md`, `devtools/baselines/RollingVarianceRatio.py`, `tests/test_rolling_variance_ratio.py`
- Modify: `bindings/bindings_rolling.cpp`, `include/screamer/common/int_settings.h` (`{"RollingVarianceRatio", "lag", 2}`, `{"RollingVarianceRatio", "window_size", 5}`), `tests/param_cases.py` (`_ROLLING_AUTO_EXCLUDE` + explicit entry), `tests/test_batch_stream_bitwise.py` (`OUTPUT_CHOICES["RollingVarianceRatio"] = ("zscore",)`), `devtools/wasm/gen_research_oracle.py`, `js/test/research.test.ts`

**Interfaces:**
- Produces: `RollingVarianceRatio(window_size=256, lag=2, output="ratio", start_policy="strict")`, 1 -> 1, `output` in `{"ratio", "zscore"}`; JavaScript `RollingVarianceRatio({ windowSize, lag, output, startPolicy })`.

**Behaviour (Lo and MacKinlay 1988, spec B):** on the window `r_1..r_n` of returns (`n = window_size` once full; with `"expanding"` the returns seen, output from `n > 2 * lag`), with `q = lag`:

- `mu = mean(r)`, `sigma_a^2 = sum (r_i - mu)^2 / (n - 1)`
- `sigma_c^2 = sum_{k=q..n} (r_k + ... + r_{k-q+1} - q mu)^2 / m`, `m = q (n - q + 1)(1 - q / n)`
- `ratio = sigma_c^2 / sigma_a^2`
- `delta_j = n sum_{i=j+1..n} (r_i - mu)^2 (r_{i-j} - mu)^2 / [sum_i (r_i - mu)^2]^2`, `theta = sum_{j=1..q-1} (2 (q - j) / q)^2 delta_j`
- `zscore = sqrt(n) (ratio - 1) / sqrt(theta)` (heteroskedasticity-robust z*(q))

NaN when `sum (r_i - mu)^2` is 0. Checks: `lag >= 2`, `window_size > 2 * lag`, `output` one of the two strings. O(window_size * lag) per step.

- [ ] **Step 1: Write the baseline** (`devtools/baselines/RollingVarianceRatio.py`)

```python
"""Lo-MacKinlay variance ratio of overlapping lag-sums over a trailing window
of returns, with the bias-corrected variances, and the heteroskedasticity-
robust z*(q) (Lo and MacKinlay 1988, equations 12, 13, 18 to 20)."""
import math

import numpy as np

from ._windowing import ignoring_nan, trailing


def variance_ratio(r, q, output):
    n = len(r)
    mu = r.mean()
    e = r - mu
    s2 = np.sum(e * e)
    if not s2 > 0.0:
        return math.nan
    var_a = s2 / (n - 1)
    sums = np.convolve(r, np.ones(q), mode="valid")          # the n - q + 1 overlapping q-sums
    m = q * (n - q + 1) * (1.0 - q / n)
    ratio = np.sum((sums - q * mu) ** 2) / m / var_a
    if output == "ratio":
        return ratio
    theta = sum((2.0 * (q - j) / q) ** 2 * n * np.sum(e[j:] ** 2 * e[:-j] ** 2) / s2 ** 2
                for j in range(1, q))
    return math.sqrt(n) * (ratio - 1.0) / math.sqrt(theta)


class RollingVarianceRatio_numpy:
    def __init__(self, window_size=256, lag=2, output="ratio", start_policy="strict"):
        self.w, self.q, self.output, self.sp = window_size, lag, output, start_policy

    def _run(self, x):
        return np.array([math.nan if w is None else variance_ratio(w, self.q, self.output)
                         for w in trailing(x, self.w, self.sp, minimum=2 * self.q + 1)])

    def __call__(self, x):
        return ignoring_nan(x, self._run)
```

- [ ] **Step 2: Write the failing tests** (`tests/test_rolling_variance_ratio.py`)

```python
import math

import numpy as np
import pytest

from devtools.baselines.RollingVarianceRatio import RollingVarianceRatio_numpy
from screamer import RollingVarianceRatio
from tests.regime_helpers import assert_regimes_agree

ALT = np.array([1.0, -1.0, 1.0, -1.0, 1.0])


def test_hand_example():
    # mu = 0.2, sum (r - mu)^2 = 4.8, the four 2-sums are 0: ratio = (0.64 / 4.8) / 1.2 = 1/9
    ratio = np.asarray(RollingVarianceRatio(5, lag=2)(ALT))
    assert np.isnan(ratio[:4]).all()
    np.testing.assert_allclose(ratio[4], 1 / 9, rtol=1e-12)
    # delta_1 = 5 * 4 * 0.9216 / 4.8^2 = 0.8 = theta: z = sqrt(5) (1/9 - 1) / sqrt(0.8) = -20/9
    z = np.asarray(RollingVarianceRatio(5, lag=2, output="zscore")(ALT))
    np.testing.assert_allclose(z[4], -20 / 9, rtol=1e-12)


def test_random_walk_ratio_is_near_one():
    r = np.random.default_rng(0).standard_normal(20_000)
    vr = np.asarray(RollingVarianceRatio(2000, lag=4)(r))
    assert abs(np.nanmean(vr) - 1.0) < 0.02


def test_trending_and_mean_reverting_returns():
    rng = np.random.default_rng(1)
    e = rng.standard_normal(5000)
    ar = np.zeros(5000)
    for t in range(1, 5000):
        ar[t] = 0.3 * ar[t - 1] + e[t]
    assert np.nanmean(np.asarray(RollingVarianceRatio(500, lag=5)(ar))) > 1.2
    assert np.nanmean(np.asarray(RollingVarianceRatio(500, lag=5)(np.diff(e)))) < 0.8


def test_constant_window_is_nan():
    # 0.5 is exact in binary, so the window mean is exactly 0.5 and the dispersion exactly 0
    out = np.asarray(RollingVarianceRatio(10, lag=2)(np.full(30, 0.5)))
    assert np.isnan(out).all()
    out = np.asarray(RollingVarianceRatio(10, lag=2, output="zscore")(np.full(30, 0.5)))
    assert np.isnan(out).all()


@pytest.mark.parametrize("output", ["ratio", "zscore"])
@pytest.mark.parametrize("lag", [2, 3, 7])
@pytest.mark.parametrize("start_policy", ["strict", "expanding", "zero"])
def test_matches_baseline(output, lag, start_policy):
    r = np.random.default_rng(lag).standard_normal(400)
    r[[3, 120, 121]] = np.nan
    got = np.asarray(RollingVarianceRatio(60, lag=lag, output=output, start_policy=start_policy)(r))
    want = RollingVarianceRatio_numpy(60, lag, output, start_policy)(r)
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12, equal_nan=True)


def test_expanding_starts_after_two_lags():
    r = np.random.default_rng(2).standard_normal(20)
    out = np.asarray(RollingVarianceRatio(10, lag=3, start_policy="expanding")(r))
    assert np.isnan(out[:6]).all() and np.isfinite(out[6:]).all()


@pytest.mark.parametrize("output", ["ratio", "zscore"])
def test_regimes_agree(output):
    r = np.random.default_rng(3).standard_normal(300)
    r[77] = np.nan
    assert_regimes_agree(lambda: RollingVarianceRatio(50, lag=4, output=output), r)


def test_settings_are_checked():
    with pytest.raises(ValueError):
        RollingVarianceRatio(10, lag=1)
    with pytest.raises(ValueError):
        RollingVarianceRatio(8, lag=4)                        # needs window_size > 2 lag
    with pytest.raises(ValueError, match="ratio.*zscore"):
        RollingVarianceRatio(10, lag=2, output="z")
```

- [ ] **Step 3: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_rolling_variance_ratio.py -q 2>&1 | tail -5`
Expected: FAIL (`ImportError`).

- [ ] **Step 4: Implement** `include/screamer/rolling_variance_ratio.h`

```cpp
#ifndef SCREAMER_ROLLING_VARIANCE_RATIO_H
#define SCREAMER_ROLLING_VARIANCE_RATIO_H

// RollingVarianceRatio: the Lo-MacKinlay (1988) variance ratio of a window
// of returns r_1..r_n, with q = lag:
//     mu        = mean(r)
//     sigma_a^2 = sum (r_i - mu)^2 / (n - 1)
//     sigma_c^2 = sum_{k=q..n} (r_k + ... + r_{k-q+1} - q mu)^2 / m,
//                 m = q (n - q + 1) (1 - q / n)
//     ratio     = sigma_c^2 / sigma_a^2
// the overlapping q-step sums with the bias-corrected variances. Near 1 for
// uncorrelated returns, above 1 when returns trend, below 1 when they revert.
// output "zscore" is the heteroskedasticity-robust statistic
//     z*(q) = sqrt(n) (ratio - 1) / sqrt(theta),
//     theta = sum_{j=1..q-1} (2 (q - j) / q)^2 delta_j,
//     delta_j = n sum_{i>j} (r_i - mu)^2 (r_{i-j} - mu)^2 / (sum_i (r_i - mu)^2)^2.
// NaN when the window has no dispersion.
//
// start_policy: "strict" gives NaN until the window holds window_size
// returns, "expanding" uses the returns seen so far once there are more than
// 2 lag, "zero" starts from a window of zeros. nan_policy: ignore.
// O(window_size * lag) per step, from a ring of the window and one scratch
// array allocated at construction.

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "screamer/common/base.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/start_policy.h"

namespace screamer {

class RollingVarianceRatio : public ScreamerBase {
public:
    static bool parse_output(const std::string& s) {
        if (s == "ratio") return false;
        if (s == "zscore") return true;
        throw std::invalid_argument("output must be \"ratio\" or \"zscore\".");
    }

    RollingVarianceRatio(int window_size = 256, int lag = 2,
                         const std::string& output = "ratio",
                         const std::string& start_policy = "strict")
        : w_(window_size), q_(lag), zscore_(parse_output(output)),
          policy_(detail::parse_start_policy(start_policy))
    {
        if (lag < 2) throw std::invalid_argument("lag must be 2 or more.");
        if (window_size <= 2 * lag)
            throw std::invalid_argument("window_size must be more than 2 lag.");
        ring_.assign(static_cast<std::size_t>(w_), 0.0);
        dev_.assign(static_cast<std::size_t>(w_), 0.0);
        reset();
    }

    void reset() override {
        std::fill(ring_.begin(), ring_.end(), 0.0);
        head_ = 0;
        count_ = policy_ == detail::StartPolicy::Zero ? static_cast<std::size_t>(w_) : 0;
    }

    double process_scalar(double x) override {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        if (isnan2(x)) return nan;
        const std::size_t W = ring_.size();
        ring_[head_] = x;
        if (++head_ == W) head_ = 0;
        if (count_ < W) ++count_;
        if (policy_ == detail::StartPolicy::Strict && count_ < W) return nan;
        if (count_ <= 2 * static_cast<std::size_t>(q_)) return nan;
        return value();
    }

private:
    // The statistic of the count_ newest values, oldest first in dev_.
    double value() {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        const std::size_t W = ring_.size(), n = count_;
        std::size_t start = (head_ + W - n) % W;   // the oldest value in the window
        double sum = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double v = ring_[(start + i) % W];
            dev_[i] = v;
            sum += v;
        }
        const double nd = static_cast<double>(n);
        const double mu = sum / nd;
        const std::size_t q = static_cast<std::size_t>(q_);
        const double qd = static_cast<double>(q_);
        // overlapping q-sums of the returns, about q mu
        double s = 0.0;
        for (std::size_t i = 0; i < q; ++i) s += dev_[i];
        double acc = (s - qd * mu) * (s - qd * mu);
        for (std::size_t k = q; k < n; ++k) {
            s += dev_[k] - dev_[k - q];
            const double d = s - qd * mu;
            acc += d * d;
        }
        double s2 = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            dev_[i] -= mu;
            s2 += dev_[i] * dev_[i];
        }
        if (!(s2 > 0.0)) return nan;
        const double var_a = s2 / (nd - 1.0);
        const double m = qd * (nd - qd + 1.0) * (1.0 - qd / nd);
        const double ratio = acc / m / var_a;
        if (!zscore_) return ratio;
        double theta = 0.0;
        for (std::size_t j = 1; j < q; ++j) {
            double d = 0.0;
            for (std::size_t i = j; i < n; ++i) d += dev_[i] * dev_[i] * dev_[i - j] * dev_[i - j];
            const double wj = 2.0 * static_cast<double>(q - j) / qd;
            theta += wj * wj * nd * d / (s2 * s2);
        }
        return std::sqrt(nd) * (ratio - 1.0) / std::sqrt(theta);
    }

    const int w_;
    const int q_;
    const bool zscore_;
    const detail::StartPolicy policy_;
    std::vector<double> ring_, dev_;
    std::size_t head_ = 0, count_ = 0;
};

}  // namespace screamer

#endif  // SCREAMER_ROLLING_VARIANCE_RATIO_H
```

Add `#include <algorithm>` for `std::fill`.

`bindings/bindings_rolling.cpp`: `#include "screamer/rolling_variance_ratio.h"` and

```cpp
    nb::class_<screamer::RollingVarianceRatio, screamer::ScreamerBase>(m, "RollingVarianceRatio")
        .def(nb::init<screamer::IntArg<int>, screamer::IntArg<int>, const std::string&, const std::string&>(),
             "window_size"_a = 256, "lag"_a = 2, "output"_a = "ratio", "start_policy"_a = "strict")
        .def("__call__", &screamer::screamer_call, "value"_a.none())
        .def("reset", &screamer::RollingVarianceRatio::reset, "Reset.");
```

`tests/param_cases.py`: add `'RollingVarianceRatio'` to `_ROLLING_AUTO_EXCLUDE` and to `test_definitions`:
`( ('RollingVarianceRatio',)    , {"window_size": [50], "lag": [2, 5], "output": ["ratio", "zscore"], "start_policy": ["strict", "expanding", "zero"]}),`

`tests/test_batch_stream_bitwise.py`: `OUTPUT_CHOICES` gains `"RollingVarianceRatio": ("zscore",),` (the variant sweep looks up every operator with an `output` setting there).

`devtools/wasm/gen_research_oracle.py` `CASES`:

```python
    ("RollingVarianceRatio", {"window_size": 60, "lag": 4}, "returns"),
    ("RollingVarianceRatio", {"window_size": 60, "lag": 4, "output": "zscore", "start_policy": "expanding"}, "returns"),
```

`js/test/research.test.ts`:

```typescript
test("research: RollingVarianceRatio hand example", () => {
  const x = Float64Array.from([1, -1, 1, -1, 1]);
  const ratio = (api as any).RollingVarianceRatio({ windowSize: 5, lag: 2 })(x) as Float64Array;
  const z = (api as any).RollingVarianceRatio({ windowSize: 5, lag: 2, output: "zscore" })(x) as Float64Array;
  assert.ok(Math.abs(ratio[4] - 1 / 9) < 1e-12 && Math.abs(z[4] + 20 / 9) < 1e-12);
});
```

`docs/functions_rolling/RollingVarianceRatio.md`:

````markdown
---
name: RollingVarianceRatio
title: Rolling Variance Ratio
implementation_family: rolling
topics:
- statistics
- regime
tags:
- variance ratio
- lo-mackinlay
- random walk test
- autocorrelation
- trend
- mean reversion
- efficiency
short: Lo-MacKinlay variance ratio of a window of returns, near 1 for a random walk, above 1
  when returns trend and below 1 when they revert, or its robust z-statistic.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 256
  min: 5
  description: Number of most recent returns in the window. Must be more than 2 lag.
- name: lag
  type: int
  default: 2
  min: 2
  description: Number of returns summed into each long-horizon return (q).
- name: output
  type: str
  default: ratio
  enum:
  - ratio
  - zscore
  description: '"ratio" for the variance ratio VR(q), "zscore" for the heteroskedasticity-robust
    statistic z*(q).'
- name: start_policy
  type: str
  default: strict
  enum:
  - strict
  - expanding
  - zero
  description: 'Output before the window is full: "strict" gives NaN, "expanding" uses the
    returns seen so far once there are more than 2 lag, and "zero" treats the window as pre-filled
    with zeros.'
nan_policy: ignore
see_also:
- RollingHurst
- RollingADF
- RollingPermutationEntropy
- RollingOU
references:
- 'Lo, A. W. and MacKinlay, A. C. (1988). Stock market prices do not follow random walks: evidence
  from a simple specification test. *Review of Financial Studies* 1(1), 41-66.'
---

# `RollingVarianceRatio`

## Description

`RollingVarianceRatio` is the variance ratio of Lo and MacKinlay (1988) on a
trailing window of `n` returns: the variance of the overlapping `q`-step sums
of returns, divided by `q` times the variance of one-step returns, with
`q = lag`. Returns that are uncorrelated give a ratio near 1; returns that
continue in the same direction (a trend) give more than 1; returns that
reverse (mean reversion) give less than 1.

With `mu` the window mean, the one-step variance is
`sum (r_i - mu)^2 / (n - 1)` and the `q`-step variance is the sum of the
squared deviations of the `n - q + 1` overlapping `q`-sums from `q mu`,
divided by `m = q (n - q + 1)(1 - q / n)`; both are the authors'
bias-corrected estimators. On the returns `[1, -1, 1, -1, 1]` with `lag=2`,
every 2-sum is 0 and the ratio is `1/9`.

`output="zscore"` gives the heteroskedasticity-robust statistic `z*(q)`, which
is approximately standard normal when the returns are uncorrelated, even with
changing volatility; `|z*| > 1.96` rejects a random walk at the 5% level. The
output is `NaN` on a window whose returns are all equal.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input gives a `NaN` output at that step. The step is skipped: it does not enter the window or update the internal state. See [NaN and warmup](../nan_and_warmup.md).
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from screamer import RollingVarianceRatio

    rng = np.random.default_rng(4)
    e = rng.standard_normal(3000)
    r = np.empty(3000)
    r[0] = e[0]
    for t in range(1, 3000):
        phi = 0.3 if t < 1500 else -0.3         # trending, then reverting
        r[t] = phi * r[t - 1] + e[t]
    vr = RollingVarianceRatio(250, lag=5)(r)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=vr, name='VR(5)', line=dict(color='navy')))
    fig.add_hline(y=1.0, line=dict(color='gray', dash='dot'))
    fig.update_layout(title='RollingVarianceRatio: above 1 while returns trend, below 1 while they revert',
                      xaxis_title='step', yaxis_title='variance ratio',
                      margin=dict(l=20, r=20, t=60, b=20), showlegend=False)
    fig.show()
```

<!-- HELP_END -->
````

- [ ] **Step 5: Regenerate and run the tests**

Run: `REGEN`, `.venv/bin/python -m pytest tests/test_rolling_variance_ratio.py -q 2>&1 | tail -5`, `REGISTRY RollingVarianceRatio`, `FRESH`.
Expected: all pass.

- [ ] **Step 6: Tidy, full suite, JavaScript** (`make tidy`, full suite, `JSTEST`). Expected: clean, all pass.

- [ ] **Step 7: Commit**

```bash
git add include/screamer/rolling_variance_ratio.h bindings/bindings_rolling.cpp include/screamer/common/int_settings.h \
  docs/functions_rolling/RollingVarianceRatio.md devtools/baselines/RollingVarianceRatio.py \
  tests/test_rolling_variance_ratio.py tests/param_cases.py tests/test_batch_stream_bitwise.py \
  devtools/wasm/gen_research_oracle.py js/test/research.test.ts screamer/__init__.py screamer/data/help.json \
  docs/by_group/statistics.rst docs/by_group/machine-learning.rst docs/function_index.txt docs/gallery.rst \
  docs/_static/gallery/manifest.json docs/_static/gallery/functions_rolling__RollingVarianceRatio.webp \
  devtools/wasm/wasm_manifest.json wasm/generated/bindings_wasm.cpp js/src/generated/ops.ts \
  js/src/generated/ops.d.ts js/src/generated/docs_data.ts js/test/docs_oracle.json wasm/smoke/null_oracle.json \
  wasm/smoke/oracle_research.json
git status --short   # add any other path the regeneration changed, by name
git commit -m "feat(rolling): RollingVarianceRatio (Lo-MacKinlay ratio and robust z-statistic)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PZMDN2ijvnSZ7ZnrJUxnd8"
```

---

### Task 8: `RollingPermutationEntropy`

**Files:**
- Create: `include/screamer/rolling_permutation_entropy.h`, `docs/functions_rolling/RollingPermutationEntropy.md`, `devtools/baselines/RollingPermutationEntropy.py`, `tests/test_rolling_permutation_entropy.py`
- Modify: `bindings/bindings_rolling.cpp`, `include/screamer/common/int_settings.h` (`{"RollingPermutationEntropy", "delay", 1}`, `{"RollingPermutationEntropy", "order", 2}`, `{"RollingPermutationEntropy", "window_size", 2}`), `tests/param_cases.py` (`_ROLLING_AUTO_EXCLUDE` + explicit entry), `devtools/wasm/gen_research_oracle.py`, `js/test/research.test.ts`

**Interfaces:**
- Produces: `RollingPermutationEntropy(window_size=100, order=3, delay=1, start_policy="strict")`, 1 -> 1, output in `[0, 1]`; JavaScript `RollingPermutationEntropy({ windowSize, order, delay, startPolicy })`.

**Behaviour (Bandt and Pompe 2002, spec B):** at each step with `(order - 1) * delay` earlier values, the ordinal pattern of `x_{t-(order-1) delay}, ..., x_{t-delay}, x_t` (ranks, ties ranked by order of appearance, the earlier value lower). The output is the Shannon entropy of the patterns ending at the last `window_size` steps, divided by `log(order!)`. `"strict"`: NaN until `window_size` patterns; `"expanding"`: the patterns seen (one at least); `"zero"`: the input is preceded by zeros, so the history is full and the window starts with `window_size` all-tie patterns. Exactly 0 when the window holds one distinct pattern. Checks: `2 <= order <= 7`, `delay >= 1`, `window_size >= 2`.

- [ ] **Step 1: Write the baseline** (`devtools/baselines/RollingPermutationEntropy.py`)

```python
"""Bandt-Pompe permutation entropy of the ordinal patterns ending in a
trailing window, normalized by log(order!). A pattern is the stable argsort
of order values spaced delay apart, so ties rank by order of appearance."""
import math
from collections import Counter

import numpy as np

from ._windowing import ignoring_nan


class RollingPermutationEntropy_numpy:
    def __init__(self, window_size=100, order=3, delay=1, start_policy="strict"):
        self.w, self.m, self.tau, self.sp = window_size, order, delay, start_policy

    def _run(self, x):
        span = (self.m - 1) * self.tau
        pad = self.w + span if self.sp == "zero" else 0
        x = np.concatenate([np.zeros(pad), x])
        codes = [None] * len(x)
        for t in range(span, len(x)):
            codes[t] = tuple(np.argsort(x[t - span: t + 1: self.tau], kind="stable"))
        norm = math.log(math.factorial(self.m))
        out = np.full(len(x), np.nan)
        for t in range(span, len(x)):
            win = codes[max(span, t - self.w + 1): t + 1]
            if self.sp == "strict" and len(win) < self.w:
                continue
            p = np.array(list(Counter(win).values()), dtype=float) / len(win)
            out[t] = max(0.0, -np.sum(p * np.log(p)) / norm)
        return out[pad:]

    def __call__(self, x):
        return ignoring_nan(x, self._run)
```

- [ ] **Step 2: Write the failing tests** (`tests/test_rolling_permutation_entropy.py`)

```python
import math

import numpy as np
import pytest

from devtools.baselines.RollingPermutationEntropy import RollingPermutationEntropy_numpy
from screamer import RollingPermutationEntropy
from tests.regime_helpers import assert_regimes_agree


def test_monotone_and_constant_series_are_exactly_zero():
    up = np.asarray(RollingPermutationEntropy(10, order=3)(np.arange(50.0)))
    assert (up[11:] == 0.0).all()
    flat = np.asarray(RollingPermutationEntropy(10, order=4)(np.full(50, 7.0)))
    assert (flat[12:] == 0.0).all()


def test_order_two_alternating():
    # patterns up, down, up, down, up from step 1; a window of 3 holds 2:1 of them
    out = np.asarray(RollingPermutationEntropy(3, order=2)(np.array([1.0, 2, 1, 2, 1, 2])))
    assert np.isnan(out[:3]).all()
    h = -(2 / 3 * math.log(2 / 3) + 1 / 3 * math.log(1 / 3)) / math.log(2)
    np.testing.assert_allclose(out[3:], [h, h, h], rtol=1e-12)


def test_ties_rank_by_appearance():
    # (2, 1, 1) and (1, 1, 2) are two different patterns: entropy log 2 / log 3!
    out = np.asarray(RollingPermutationEntropy(2, order=3)(np.array([2.0, 1.0, 1.0, 2.0])))
    np.testing.assert_allclose(out[3], math.log(2) / math.log(6), rtol=1e-12)


def test_white_noise_is_near_one():
    x = np.random.default_rng(0).standard_normal(5000)
    out = np.asarray(RollingPermutationEntropy(1000, order=4)(x))
    assert np.nanmin(out) > 0.97


def test_expanding_and_zero():
    x = np.array([1.0, 2.0, 1.0, 2.0])
    exp = np.asarray(RollingPermutationEntropy(3, order=2, start_policy="expanding")(x))
    assert math.isnan(exp[0])
    np.testing.assert_allclose(exp[1:], [0.0, 1.0, -(2 / 3 * math.log(2 / 3) + 1 / 3 * math.log(1 / 3)) / math.log(2)],
                               rtol=1e-12)
    zero = np.asarray(RollingPermutationEntropy(3, order=2, start_policy="zero")(x))
    np.testing.assert_allclose(zero, RollingPermutationEntropy_numpy(3, 2, 1, "zero")(x), rtol=1e-12)


@pytest.mark.parametrize("order,delay", [(2, 1), (3, 1), (3, 2), (5, 1), (7, 1)])
@pytest.mark.parametrize("start_policy", ["strict", "expanding", "zero"])
def test_matches_baseline(order, delay, start_policy):
    rng = np.random.default_rng(order * 10 + delay)
    x = np.round(rng.standard_normal(400).cumsum(), 1)        # rounding makes ties
    x[[11, 200]] = np.nan
    got = np.asarray(RollingPermutationEntropy(40, order=order, delay=delay, start_policy=start_policy)(x))
    want = RollingPermutationEntropy_numpy(40, order, delay, start_policy)(x)
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12, equal_nan=True)


def test_long_stream_has_no_drift():
    x = np.random.default_rng(9).standard_normal(100_000)
    got = np.asarray(RollingPermutationEntropy(50, order=3)(x))[-200:]
    want = RollingPermutationEntropy_numpy(50, 3, 1, "strict")(x[-400:])[-200:]
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-13)


def test_regimes_agree():
    x = np.random.default_rng(1).standard_normal(300)
    x[5] = np.nan
    assert_regimes_agree(lambda: RollingPermutationEntropy(30, order=4, delay=2), x)


def test_settings_are_checked():
    with pytest.raises(ValueError):
        RollingPermutationEntropy(10, order=1)
    with pytest.raises(ValueError, match="order"):
        RollingPermutationEntropy(10, order=8)
    with pytest.raises(ValueError):
        RollingPermutationEntropy(10, delay=0)
    with pytest.raises(ValueError):
        RollingPermutationEntropy(1)
```

Expanding hand values for `[1, 2, 1, 2]`, order 2: no pattern at step 0; one "up" at step 1 (0); up, down at step 2 (`log 2 / log 2 = 1`); up, down, up at step 3.

- [ ] **Step 3: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_rolling_permutation_entropy.py -q 2>&1 | tail -5`
Expected: FAIL (`ImportError`).

- [ ] **Step 4: Implement** `include/screamer/rolling_permutation_entropy.h`

```cpp
#ifndef SCREAMER_ROLLING_PERMUTATION_ENTROPY_H
#define SCREAMER_ROLLING_PERMUTATION_ENTROPY_H

// RollingPermutationEntropy: the Bandt-Pompe (2002) permutation entropy of a
// series over a trailing window. At each step with (order - 1) * delay earlier
// values, the values x_{t-(order-1) delay}, ..., x_{t-delay}, x_t form an
// ordinal pattern (their ranks; equal values rank by order of appearance, the
// earlier one lower). The output is the Shannon entropy of the patterns that
// end at the last window_size steps, divided by log(order!), so it lies in
// [0, 1]: 0 when one pattern repeats (a monotone or constant series), near 1
// for white noise.
//
// A pattern is stored as its Lehmer code, sum_i c_i (order - 1 - i)!, with c_i
// the number of later values strictly below value i, which is a bijection
// with the tie-broken ranks. The counts of the order! codes and
// S = sum_c n_c log n_c are updated as patterns enter and leave, so a step
// costs O(order^2); S is recomputed from the counts every window_size steps,
// which bounds the rounding the updates accumulate. Entropy =
// (W log W - S) / (W log(order!)), W the patterns in the window, and exactly 0
// when one distinct pattern fills the window.
//
// start_policy: "strict" gives NaN until window_size patterns, "expanding"
// uses the patterns seen so far, "zero" behaves as if the input were preceded
// by zeros (a full history and window_size all-tie patterns). nan_policy:
// ignore (a NaN is not stored and forms no pattern).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "screamer/common/base.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/start_policy.h"

namespace screamer {

class RollingPermutationEntropy : public ScreamerBase {
public:
    RollingPermutationEntropy(int window_size = 100, int order = 3, int delay = 1,
                              const std::string& start_policy = "strict")
        : W_(check(window_size, order, delay)), m_(order), tau_(delay),
          span_(static_cast<std::size_t>((order - 1) * delay)),
          policy_(detail::parse_start_policy(start_policy))
    {
        std::size_t f = 1;
        weights_.assign(static_cast<std::size_t>(m_), 1);
        for (int i = m_ - 1; i >= 0; --i) {          // weights_[i] = (m - 1 - i)!
            weights_[static_cast<std::size_t>(i)] = f;
            f *= static_cast<std::size_t>(m_ - i);
        }
        n_codes_ = f;                                 // order!
        log_norm_ = std::log(static_cast<double>(n_codes_));
        nlogn_.assign(W_ + 1, 0.0);
        for (std::size_t k = 2; k <= W_; ++k)
            nlogn_[k] = static_cast<double>(k) * std::log(static_cast<double>(k));
        values_.assign(span_ + 1, 0.0);
        codes_.assign(W_, 0);
        counts_.assign(n_codes_, 0);
        pattern_.assign(static_cast<std::size_t>(m_), 0.0);
        reset();
    }

    void reset() override {
        std::fill(values_.begin(), values_.end(), 0.0);
        std::fill(codes_.begin(), codes_.end(), 0);
        std::fill(counts_.begin(), counts_.end(), 0);
        vhead_ = 0;
        chead_ = 0;
        since_exact_ = 0;
        if (policy_ == detail::StartPolicy::Zero) {
            nvalues_ = span_ + 1;
            ncodes_ = W_;
            counts_[0] = W_;                          // W all-tie patterns, code 0
            S_ = nlogn_[W_];
            distinct_ = 1;
        } else {
            nvalues_ = 0;
            ncodes_ = 0;
            S_ = 0.0;
            distinct_ = 0;
        }
    }

    double process_scalar(double x) override {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        if (isnan2(x)) return nan;
        const std::size_t V = values_.size();
        values_[vhead_] = x;
        if (++vhead_ == V) vhead_ = 0;
        if (nvalues_ < V) ++nvalues_;
        if (nvalues_ < V) return nan;                 // no pattern yet
        add_pattern(code());
        if (policy_ == detail::StartPolicy::Strict && ncodes_ < W_) return nan;
        return entropy();
    }

private:
    static std::size_t check(int window_size, int order, int delay) {
        if (order < 2) throw std::invalid_argument("order must be 2 or more.");
        if (order > 7) throw std::invalid_argument("order must be 7 or less.");
        if (delay < 1) throw std::invalid_argument("delay must be 1 or more.");
        if (window_size < 2) throw std::invalid_argument("Window size must be 2 or more.");
        return static_cast<std::size_t>(window_size);
    }

    // The Lehmer code of the newest pattern.
    std::size_t code() {
        const std::size_t V = values_.size();
        const std::size_t newest = (vhead_ + V - 1) % V;
        const std::size_t m = static_cast<std::size_t>(m_);
        const std::size_t tau = static_cast<std::size_t>(tau_);
        for (std::size_t k = 0; k < m; ++k)          // oldest first
            pattern_[k] = values_[(newest + V - (m - 1 - k) * tau) % V];
        std::size_t c = 0;
        for (std::size_t i = 0; i + 1 < m; ++i) {
            std::size_t below = 0;
            for (std::size_t j = i + 1; j < m; ++j) below += pattern_[j] < pattern_[i] ? 1 : 0;
            c += below * weights_[i];
        }
        return c;
    }

    void add_pattern(std::size_t c) {
        if (ncodes_ == W_) {
            const std::size_t old = codes_[chead_];
            if (old == c) {                           // counts unchanged
                if (++chead_ == W_) chead_ = 0;
                tick();
                return;
            }
            const std::size_t n = counts_[old];
            S_ += nlogn_[n - 1] - nlogn_[n];
            counts_[old] = n - 1;
            if (n == 1) --distinct_;
        } else {
            ++ncodes_;
        }
        codes_[chead_] = c;
        if (++chead_ == W_) chead_ = 0;
        const std::size_t n = counts_[c];
        S_ += nlogn_[n + 1] - nlogn_[n];
        counts_[c] = n + 1;
        if (n == 0) ++distinct_;
        tick();
    }

    // Every W_ patterns, S from the counts (bounds the rounding of the updates).
    void tick() {
        if (++since_exact_ < W_) return;
        since_exact_ = 0;
        double s = 0.0;
        for (std::size_t n : counts_) s += nlogn_[n];
        S_ = s;
    }

    double entropy() const {
        if (distinct_ <= 1) return 0.0;
        const double w = static_cast<double>(ncodes_);
        const double h = (nlogn_[ncodes_] - S_) / (w * log_norm_);
        return h < 0.0 ? 0.0 : (h > 1.0 ? 1.0 : h);
    }

    const std::size_t W_;
    const int m_;
    const int tau_;
    const std::size_t span_;
    const detail::StartPolicy policy_;
    std::size_t n_codes_ = 1;
    double log_norm_ = 0.0;
    std::vector<std::size_t> weights_;
    std::vector<double> nlogn_;
    std::vector<double> values_;
    std::vector<std::size_t> codes_;
    std::vector<std::size_t> counts_;
    std::vector<double> pattern_;
    std::size_t vhead_ = 0, chead_ = 0, nvalues_ = 0, ncodes_ = 0, since_exact_ = 0, distinct_ = 0;
    double S_ = 0.0;
};

}  // namespace screamer

#endif  // SCREAMER_ROLLING_PERMUTATION_ENTROPY_H
```

`check()` runs in the first member initializer, so a bad setting raises before anything is allocated. `nlogn_[1]` is 0 (`1 log 1`).

`bindings/bindings_rolling.cpp`: `#include "screamer/rolling_permutation_entropy.h"` and

```cpp
    nb::class_<screamer::RollingPermutationEntropy, screamer::ScreamerBase>(m, "RollingPermutationEntropy")
        .def(nb::init<screamer::IntArg<int>, screamer::IntArg<int>, screamer::IntArg<int>, const std::string&>(),
             "window_size"_a = 100, "order"_a = 3, "delay"_a = 1, "start_policy"_a = "strict")
        .def("__call__", &screamer::screamer_call, "value"_a.none())
        .def("reset", &screamer::RollingPermutationEntropy::reset, "Reset.");
```

`tests/param_cases.py`: add `'RollingPermutationEntropy'` to `_ROLLING_AUTO_EXCLUDE` and to `test_definitions`:
`( ('RollingPermutationEntropy',), {"window_size": [30], "order": [2, 3, 5], "delay": [1, 2], "start_policy": ["strict", "expanding", "zero"]}),`

`devtools/wasm/gen_research_oracle.py` `CASES`:

```python
    ("RollingPermutationEntropy", {"window_size": 50, "order": 3}, "returns"),
    ("RollingPermutationEntropy", {"window_size": 40, "order": 5, "delay": 2, "start_policy": "expanding"}, "returns"),
    ("RollingPermutationEntropy", {"window_size": 20, "order": 4, "start_policy": "zero"}, "events"),
```

`js/test/research.test.ts`:

```typescript
test("research: RollingPermutationEntropy of a monotone series is exactly 0", () => {
  const x = Float64Array.from({ length: 50 }, (_, i) => i);
  const out = (api as any).RollingPermutationEntropy({ windowSize: 10, order: 3 })(x) as Float64Array;
  assert.ok(Array.from(out.subarray(11)).every((v) => v === 0));
});
```

`docs/functions_rolling/RollingPermutationEntropy.md`:

````markdown
---
name: RollingPermutationEntropy
title: Rolling Permutation Entropy
implementation_family: rolling
topics:
- statistics
- regime
tags:
- permutation entropy
- ordinal patterns
- bandt-pompe
- complexity
- randomness
- predictability
- entropy
short: Entropy of the ordinal patterns of a series over a trailing window, from 0 (one pattern
  repeats) to 1 (every pattern equally often).
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 100
  min: 2
  description: Number of patterns, ending at the most recent steps, in the window.
- name: order
  type: int
  default: 3
  min: 2
  description: Number of values in a pattern, 2 to 7. There are order! patterns.
- name: delay
  type: int
  default: 1
  min: 1
  description: Steps between the values of a pattern.
- name: start_policy
  type: str
  default: strict
  enum:
  - strict
  - expanding
  - zero
  description: 'Output before the window holds window_size patterns: "strict" gives NaN,
    "expanding" uses the patterns seen so far, and "zero" behaves as if the input were preceded
    by zeros.'
nan_policy: ignore
see_also:
- RollingVarianceRatio
- RollingHurst
- RollingJumpShare
references:
- 'Bandt, C. and Pompe, B. (2002). Permutation entropy: a natural complexity measure for time
  series. *Physical Review Letters* 88(17), 174102.'
---

# `RollingPermutationEntropy`

## Description

`RollingPermutationEntropy` is the permutation entropy of Bandt and Pompe
(2002) over a trailing window. At each step, the `order` values
`x[t - (order-1)*delay], ..., x[t - delay], x[t]` form an ordinal pattern: the
order of their ranks, one of `order!` possibilities. The output is the Shannon
entropy of the patterns that end at the last `window_size` steps, divided by
`log(order!)`:

$$
H = -\frac{1}{\log(\text{order}!)} \sum_{\pi} p_\pi \log p_\pi
$$

where `p_pi` is the share of pattern `pi` in the window. `H` is 0 when one
pattern fills the window, such as on a monotone or constant series, and near 1
when every pattern is equally frequent, as for white noise. Values in between
mean that some orderings are more common than others, which a forecast can
use.

Equal values rank by order of appearance, the earlier one lower, so on prices
that repeat a tick, `(2, 1, 1)` and `(1, 1, 2)` are two patterns. The input can
be prices or returns; a pattern of prices describes the signs of the moves
between them.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input gives a `NaN` output at that step. The step is skipped: it does not enter the window or update the internal state. See [NaN and warmup](../nan_and_warmup.md).
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from screamer import RollingPermutationEntropy

    rng = np.random.default_rng(6)
    noise = rng.standard_normal(1500)
    cycle = np.sin(np.arange(1500) * 2 * np.pi / 25) + 0.2 * rng.standard_normal(1500)
    x = np.where(np.arange(1500) < 750, noise, cycle)     # random, then a cycle
    h = RollingPermutationEntropy(200, order=4)(x)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=h, name='entropy', line=dict(color='navy')))
    fig.update_layout(title='RollingPermutationEntropy: near 1 on noise, lower on a cycle',
                      xaxis_title='step', yaxis_title='normalized entropy',
                      margin=dict(l=20, r=20, t=60, b=20), showlegend=False)
    fig.show()
```

<!-- HELP_END -->
````

- [ ] **Step 5: Regenerate and run the tests**

Run: `REGEN`, `.venv/bin/python -m pytest tests/test_rolling_permutation_entropy.py -q 2>&1 | tail -5`, `REGISTRY RollingPermutationEntropy`, `FRESH`.
Expected: all pass.

- [ ] **Step 6: Tidy, full suite, JavaScript** (`make tidy`, full suite, `JSTEST`). Expected: clean, all pass.

- [ ] **Step 7: Commit**

```bash
git add include/screamer/rolling_permutation_entropy.h bindings/bindings_rolling.cpp \
  include/screamer/common/int_settings.h docs/functions_rolling/RollingPermutationEntropy.md \
  devtools/baselines/RollingPermutationEntropy.py tests/test_rolling_permutation_entropy.py tests/param_cases.py \
  devtools/wasm/gen_research_oracle.py js/test/research.test.ts screamer/__init__.py screamer/data/help.json \
  docs/by_group/statistics.rst docs/by_group/machine-learning.rst docs/function_index.txt docs/gallery.rst \
  docs/_static/gallery/manifest.json docs/_static/gallery/functions_rolling__RollingPermutationEntropy.webp \
  devtools/wasm/wasm_manifest.json wasm/generated/bindings_wasm.cpp js/src/generated/ops.ts \
  js/src/generated/ops.d.ts js/src/generated/docs_data.ts js/test/docs_oracle.json wasm/smoke/null_oracle.json \
  wasm/smoke/oracle_research.json
git status --short   # add any other path the regeneration changed, by name
git commit -m "feat(rolling): RollingPermutationEntropy

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PZMDN2ijvnSZ7ZnrJUxnd8"
```

---

### Task 9: `BayesianChangepoint`

**Files:**
- Create: `include/screamer/bayesian_changepoint.h`, `docs/functions_fin/BayesianChangepoint.md`, `devtools/baselines/BayesianChangepoint.py`, `tests/test_bayesian_changepoint.py`
- Modify: `bindings/bindings_fin.cpp` (include + class after `BayesianRegression` ~line 1381), `include/screamer/common/int_settings.h` (`{"BayesianChangepoint", "max_run", 1}`), `tests/param_cases.py` (`multi_input_definitions` entry), `devtools/wasm/gen_research_oracle.py`, `js/test/research.test.ts`

**Interfaces:**
- Produces: `BayesianChangepoint(expected_run=250.0, max_run=1000, prior_var=1.0)`, 1 -> 3, `output_names: [run_length, pred_mean, pred_std]`; JavaScript `BayesianChangepoint({ expectedRun, maxRun, priorVar })`, a `(T, 3)` NdArray on a series.

**Behaviour (Adams and MacKay 2007, spec C2):** hazard `H = 1 / expected_run`; normal-gamma prior `mu0 = 0, kappa0 = 1, alpha0 = 1, beta0 = prior_var`. The value of a step belongs to the run it starts or continues, so a run of length `r` (0 at a changepoint) has seen `r + 1` values: `kappa_r = r + 2`, `alpha_r = 1.5 + r / 2`, with running `mu_r`, `beta_r`. For a value `x` after the first:

- predictive log densities `lp_r` of the Student-t with `nu = 2 alpha_r`, location `mu_r`, scale^2 `beta_r (kappa_r + 1) / (alpha_r kappa_r)`, and `lp0` of the prior predictive (`nu = 2`, location 0, scale^2 `2 prior_var`);
- growth `P'(r + 1) ∝ P(r) exp(lp_r) (1 - H)` for `r + 1 <= max_run` (the mass beyond `max_run` is dropped), changepoint `P'(0) ∝ H exp(lp0)`, normalized;
- updates `mu_{r+1} = (kappa_r mu_r + x) / (kappa_r + 1)`, `beta_{r+1} = beta_r + kappa_r (x - mu_r)^2 / (2 (kappa_r + 1))`, and the new run `mu_0 = x / 2`, `beta_0 = prior_var + x^2 / 4`.

The first value sets `P = [1]` and the new-run parameters. Outputs after each value: `run_length = sum r P(r)`; the predictive of the next value is the mixture of the runs' Student-t with means `mu_r` and variances `beta_r (kappa_r + 1) / (kappa_r (alpha_r - 1))`: `pred_mean = sum P(r) mu_r`, `pred_std = sqrt(sum P(r) (var_r + mu_r^2) - pred_mean^2)`. NaN input: NaN row, state unchanged. Checks: `expected_run >= 1`, `max_run >= 1`, `prior_var > 0`, all finite.

- [ ] **Step 1: Write the baseline** (`devtools/baselines/BayesianChangepoint.py`)

```python
"""Adams-MacKay online Bayesian changepoint detection, written as a direct
numpy transcription of the recursion: constant hazard 1 / expected_run, a
normal-gamma prior (mean 0, kappa 1, alpha 1, beta prior_var), Student-t
predictives, and the run-length distribution truncated at max_run. The value
of a step belongs to the run it starts or continues, so a run of length r has
seen r + 1 values."""
import math

import numpy as np


def _t_logpdf(x, mu, s2, nu):
    return (math.lgamma((nu + 1.0) / 2.0) - math.lgamma(nu / 2.0) - 0.5 * np.log(nu * math.pi * s2)
            - (nu + 1.0) / 2.0 * np.log1p((x - mu) ** 2 / (nu * s2)))


class BayesianChangepoint_numpy:
    def __init__(self, expected_run=250.0, max_run=1000, prior_var=1.0):
        self.h, self.R, self.b0 = 1.0 / expected_run, max_run + 1, prior_var

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        out = np.full((len(x), 3), np.nan)
        prob = mu = beta = None
        for t, v in enumerate(x):
            if math.isnan(v):
                continue
            if prob is None:
                prob = np.array([1.0])
                mu, beta = np.array([v / 2.0]), np.array([self.b0 + v * v / 4.0])
            else:
                r = np.arange(len(prob))
                kappa, alpha = r + 2.0, 1.5 + r / 2.0
                lp = np.array([_t_logpdf(v, mu[i], beta[i] * (kappa[i] + 1) / (alpha[i] * kappa[i]),
                                         2.0 * alpha[i]) for i in range(len(prob))])
                lp0 = _t_logpdf(v, 0.0, 2.0 * self.b0, 2.0)
                m = max(lp0, lp.max())
                grow = prob * np.exp(lp - m) * (1.0 - self.h)
                new = np.concatenate([[self.h * math.exp(lp0 - m)], grow])[: self.R]
                prob = new / new.sum()
                mu_g = (kappa * mu + v) / (kappa + 1.0)
                beta_g = beta + kappa * (v - mu) ** 2 / (2.0 * (kappa + 1.0))
                mu = np.concatenate([[v / 2.0], mu_g])[: self.R]
                beta = np.concatenate([[self.b0 + v * v / 4.0], beta_g])[: self.R]
            r = np.arange(len(prob))
            kappa, alpha = r + 2.0, 1.5 + r / 2.0
            var = beta * (kappa + 1.0) / (kappa * (alpha - 1.0))
            mean = prob @ mu
            out[t] = (prob @ r, mean, math.sqrt(max(prob @ (var + mu * mu) - mean * mean, 0.0)))
        return out
```

- [ ] **Step 2: Write the failing tests** (`tests/test_bayesian_changepoint.py`)

```python
import math

import numpy as np
import pytest

from devtools.baselines.BayesianChangepoint import BayesianChangepoint_numpy
from screamer import BayesianChangepoint
from tests.regime_helpers import assert_regimes_agree


def test_first_value():
    # one value 0.5: run length 0, mu = 0.25, beta = 1 + 0.25 / 4, var = beta * 3 / (2 * 0.5)
    out = np.asarray(BayesianChangepoint()(np.array([0.5])))
    np.testing.assert_allclose(out[0], [0.0, 0.25, math.sqrt(3.1875)], rtol=1e-12)


def test_run_length_drops_after_a_three_sigma_shift():
    rng = np.random.default_rng(3)
    x = np.r_[rng.standard_normal(300), 3.0 + rng.standard_normal(200)]
    rl = np.asarray(BayesianChangepoint(expected_run=250.0)(x))[:, 0]
    assert rl[299] > 100
    assert rl[305] < 20


def test_run_length_is_capped_by_max_run():
    x = np.random.default_rng(4).standard_normal(400)
    rl = np.asarray(BayesianChangepoint(expected_run=1000.0, max_run=50)(x))[:, 0]
    assert np.nanmax(rl) <= 50.0


@pytest.mark.parametrize("expected_run,max_run,prior_var", [(250.0, 1000, 1.0), (20.0, 30, 0.5),
                                                            (1.0, 5, 2.0)])
def test_matches_baseline(expected_run, max_run, prior_var):
    rng = np.random.default_rng(max_run)
    x = np.r_[rng.standard_normal(150), 2.0 + 0.5 * rng.standard_normal(150)]
    x[[7, 160]] = np.nan
    got = np.asarray(BayesianChangepoint(expected_run, max_run, prior_var)(x))
    want = BayesianChangepoint_numpy(expected_run, max_run, prior_var)(x)
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-10, equal_nan=True)


def test_extreme_value_does_not_break_the_state():
    x = np.r_[np.random.default_rng(5).standard_normal(50), 1e6, np.zeros(20)]
    out = np.asarray(BayesianChangepoint()(x))
    assert np.isfinite(out).all()


def test_regimes_agree():
    x = np.random.default_rng(6).standard_normal(200)
    x[[10, 100]] = np.nan
    assert_regimes_agree(lambda: BayesianChangepoint(expected_run=30.0, max_run=60), x)


def test_settings_are_checked():
    with pytest.raises(ValueError):
        BayesianChangepoint(expected_run=0.5)
    with pytest.raises(ValueError):
        BayesianChangepoint(max_run=0)
    with pytest.raises(ValueError):
        BayesianChangepoint(prior_var=0.0)
```

- [ ] **Step 3: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_bayesian_changepoint.py -q 2>&1 | tail -5`
Expected: FAIL (`ImportError`).

- [ ] **Step 4: Implement** `include/screamer/bayesian_changepoint.h`

```cpp
#ifndef SCREAMER_BAYESIAN_CHANGEPOINT_H
#define SCREAMER_BAYESIAN_CHANGEPOINT_H

// BayesianChangepoint: online Bayesian changepoint detection (Adams and
// MacKay 2007). The run length r is the number of steps since the last
// changepoint (0 on the step of a change); the value of a step belongs to the
// run it starts or continues, so run r has seen r + 1 values. Each run has a
// normal-gamma posterior from the prior (mean 0, kappa 1, alpha 1, beta
// prior_var): kappa_r = r + 2, alpha_r = 1.5 + r / 2, and running mu_r,
// beta_r. With hazard H = 1 / expected_run, a value x moves the run-length
// distribution P to
//     P'(r + 1) ~ P(r) t_r(x) (1 - H),   P'(0) ~ H t_0(x),
// normalized, where t_r is run r's Student-t predictive and t_0 the prior
// predictive. Runs longer than max_run are dropped. 1 -> 3:
//     [run_length, pred_mean, pred_std]
// run_length = sum r P(r); pred_mean and pred_std are the mean and standard
// deviation of the predictive mixture of the next value, with component
// means mu_r and variances beta_r (kappa_r + 1) / (kappa_r (alpha_r - 1)).
// Densities are scaled by their maximum before exponentiation, so an extreme
// value does not underflow every run. O(min(t, max_run)) per step; the
// arrays are allocated at construction. nan_policy: ignore.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>
#include "screamer/common/finite_settings.h"
#include "screamer/common/float_info.h"
#include "screamer/common/functor_base.h"

namespace screamer {

class BayesianChangepoint : public FunctorBase<BayesianChangepoint, 1, 3> {
public:
    BayesianChangepoint(double expected_run = 250.0, int max_run = 1000, double prior_var = 1.0)
        : hazard_(check(expected_run, max_run, prior_var)), prior_var_(prior_var),
          R_(static_cast<std::size_t>(max_run) + 1)
    {
        constexpr double kPi = 3.14159265358979323846;
        prob_.assign(R_, 0.0);
        next_.assign(R_, 0.0);
        mu_.assign(R_, 0.0);
        beta_.assign(R_, 0.0);
        lp_.assign(R_, 0.0);
        nu_.assign(R_, 0.0);
        c_.assign(R_, 0.0);
        g_.assign(R_, 0.0);
        v_.assign(R_, 0.0);
        k_.assign(R_, 0.0);
        for (std::size_t r = 0; r < R_; ++r) {
            const double kappa = static_cast<double>(r) + 2.0;
            const double alpha = 1.5 + 0.5 * static_cast<double>(r);
            k_[r] = kappa;
            nu_[r] = 2.0 * alpha;
            c_[r] = std::lgamma(alpha + 0.5) - std::lgamma(alpha) - 0.5 * std::log(nu_[r] * kPi);
            g_[r] = (kappa + 1.0) / (alpha * kappa);            // scale^2 = beta * g
            v_[r] = (kappa + 1.0) / (kappa * (alpha - 1.0));    // variance = beta * v
        }
        // prior predictive: nu 2, location 0, scale^2 2 prior_var
        c0_ = std::lgamma(1.5) - std::lgamma(1.0) - 0.5 * std::log(2.0 * kPi);
        s0_ = 2.0 * prior_var_;
        reset();
    }

    void reset() override { len_ = 0; }

    ResultTuple call(const InputArray& in) override {
        const double x = in[0];
        const double nan = std::numeric_limits<double>::quiet_NaN();
        if (isnan2(x)) return std::make_tuple(nan, nan, nan);
        if (len_ == 0) {
            prob_[0] = 1.0;
            len_ = 1;
        } else {
            step(x);
        }
        mu_[0] = 0.5 * x;                            // the run that starts at x
        beta_[0] = prior_var_ + 0.25 * x * x;
        return outputs();
    }

private:
    static double check(double expected_run, int max_run, double prior_var) {
        require_finite("expected_run", expected_run);
        require_finite("prior_var", prior_var);
        if (!(expected_run >= 1.0)) throw std::invalid_argument("expected_run must be 1 or more.");
        if (max_run < 1) throw std::invalid_argument("max_run must be 1 or more.");
        if (!(prior_var > 0.0)) throw std::invalid_argument("prior_var must be positive.");
        return 1.0 / expected_run;
    }

    // One value after the first: the run-length update and the grown runs'
    // parameters (index r + 1 from r, so the loop runs downwards).
    void step(double x) {
        const double lp0 = c0_ - 0.5 * std::log(s0_) - 1.5 * std::log1p(x * x / (2.0 * s0_));
        double m = lp0;
        for (std::size_t r = 0; r < len_; ++r) {
            const double s2 = beta_[r] * g_[r];
            const double d = x - mu_[r];
            lp_[r] = c_[r] - 0.5 * std::log(s2) - 0.5 * (nu_[r] + 1.0) * std::log1p(d * d / (nu_[r] * s2));
            if (lp_[r] > m) m = lp_[r];
        }
        const std::size_t n = std::min(len_ + 1, R_);
        double total = hazard_ * std::exp(lp0 - m);
        next_[0] = total;
        for (std::size_t r = n - 1; r >= 1; --r) {
            next_[r] = prob_[r - 1] * std::exp(lp_[r - 1] - m) * (1.0 - hazard_);
            total += next_[r];
        }
        if (!(total > 0.0) || !std::isfinite(total)) {      // every run impossible: a change
            std::fill(next_.begin(), next_.begin() + static_cast<std::ptrdiff_t>(n), 0.0);
            next_[0] = 1.0;
            total = 1.0;
        }
        for (std::size_t r = 0; r < n; ++r) prob_[r] = next_[r] / total;
        for (std::size_t r = n - 1; r >= 1; --r) {
            const double kappa = k_[r - 1];
            const double d = x - mu_[r - 1];
            mu_[r] = (kappa * mu_[r - 1] + x) / (kappa + 1.0);
            beta_[r] = beta_[r - 1] + kappa * d * d / (2.0 * (kappa + 1.0));
        }
        len_ = n;
    }

    ResultTuple outputs() const {
        double rl = 0.0, mean = 0.0, second = 0.0;
        for (std::size_t r = 0; r < len_; ++r) {
            const double p = prob_[r];
            rl += p * static_cast<double>(r);
            mean += p * mu_[r];
            second += p * (beta_[r] * v_[r] + mu_[r] * mu_[r]);
        }
        const double var = second - mean * mean;
        return std::make_tuple(rl, mean, std::sqrt(var > 0.0 ? var : 0.0));
    }

    const double hazard_;
    const double prior_var_;
    const std::size_t R_;
    double c0_ = 0.0, s0_ = 0.0;
    std::vector<double> prob_, next_, mu_, beta_, lp_, nu_, c_, g_, v_, k_;
    std::size_t len_ = 0;
};

}  // namespace screamer

#endif  // SCREAMER_BAYESIAN_CHANGEPOINT_H
```

The loops `for (std::size_t r = n - 1; r >= 1; --r)` stop at 1 because `n >= 2` after the first value.

`bindings/bindings_fin.cpp`: `#include "screamer/bayesian_changepoint.h"` and

```cpp
    // 1 -> 3 online Bayesian changepoint detection: (run_length, pred_mean, pred_std).
    nb::class_<screamer::BayesianChangepoint, screamer::EvalOp>(m, "BayesianChangepoint")
        .def(nb::init<screamer::FloatArg, screamer::IntArg<int>, screamer::FloatArg>(),
             "expected_run"_a = 250.0, "max_run"_a = 1000, "prior_var"_a = 1.0)
        .def("__call__", &screamer::functor_call<screamer::BayesianChangepoint>)
        .def("reset", &screamer::BayesianChangepoint::reset, "Reset to the prior.");
```

`tests/param_cases.py`, in `multi_input_definitions`:
`( ('BayesianChangepoint',)     , {"expected_run": [50.0], "max_run": [100], "prior_var": [1.0], "array_length": [300]}),`

`devtools/wasm/gen_research_oracle.py` `CASES`:

```python
    ("BayesianChangepoint", {"expected_run": 100.0, "max_run": 300}, "returns", 1e-7),
    ("BayesianChangepoint", {"expected_run": 20.0, "max_run": 40, "prior_var": 4.0}, "returns", 1e-7),
```

`js/test/research.test.ts`:

```typescript
test("research: BayesianChangepoint first value", () => {
  const out = (api as any).BayesianChangepoint()(Float64Array.from([0.5])) as { data: Float64Array; shape: number[] };
  assert.deepEqual(out.shape, [1, 3]);
  assert.ok(out.data[0] === 0 && Math.abs(out.data[1] - 0.25) < 1e-12 && Math.abs(out.data[2] - Math.sqrt(3.1875)) < 1e-12);
});
```

`docs/functions_fin/BayesianChangepoint.md`:

````markdown
---
name: BayesianChangepoint
title: Bayesian online changepoint detection
implementation_family: fin
topics:
- regime
tags:
- changepoint
- change point
- structural break
- bayesian
- adams-mackay
- run length
- regime change
short: Online Bayesian changepoint detection (Adams-MacKay) on a series with Gaussian segments
  of unknown mean and variance; the expected run length since the last change and a predictive
  mean and standard deviation.
inputs: 1
outputs: 3
parameters:
- name: expected_run
  type: float
  default: 250.0
  min: 1.0
  description: Expected number of steps between changes; the hazard of a change at each step
    is 1 / expected_run.
- name: max_run
  type: int
  default: 1000
  min: 1
  description: Longest run length tracked. Longer runs are dropped, which bounds the cost per
    step.
- name: prior_var
  type: float
  default: 1.0
  description: Scale of the prior on a segment's variance (the normal-gamma beta). Must be
    positive. 1 suits a standardized input.
nan_policy: ignore
see_also:
- CusumFilter
- RollingHMM
- RollingZscore
- EwStd
output_names:
- run_length
- pred_mean
- pred_std
references:
- 'Adams, R. P. and MacKay, D. J. C. (2007). Bayesian online changepoint detection. arXiv:0710.3742.'
---

# `BayesianChangepoint`

## Description

`BayesianChangepoint` is the online Bayesian changepoint detector of Adams and
MacKay (2007). It models the input as segments; within a segment the values
are Gaussian with a mean and variance it does not know, and at each step a new
segment starts with probability `1 / expected_run`. After each value it updates
the probability of every run length `r`, the number of steps since the last
change (0 on the step of a change), and returns three columns:

0. `run_length`: the posterior expected run length.
1. `pred_mean`: the mean of the predictive distribution of the next value.
2. `pred_std`: its standard deviation.

Each segment's mean and variance have a normal-gamma prior with mean 0,
`kappa=1`, `alpha=1`, and `beta=prior_var`, so the predictive of a run is a
Student-t. Feed a standardized series, such as a `RollingZscore` of returns or
returns divided by an `EwStd` of them: the prior assumes values of order 1. A
drop of `run_length` towards 0 marks a change; after a shift of the mean by 3
standard deviations it falls within a few steps.

A run longer than `max_run` is dropped and the rest renormalized, so
`run_length` never exceeds `max_run` and each step costs at most
`max_run + 1` updates.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input gives a `NaN` output at that step. The step is skipped: it does not enter the window or update the internal state. See [NaN and warmup](../nan_and_warmup.md).
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import BayesianChangepoint

    rng = np.random.default_rng(7)
    x = np.r_[rng.standard_normal(300), 2.5 + rng.standard_normal(300),
              0.5 * rng.standard_normal(300)]
    out = BayesianChangepoint(expected_run=200.0)(x)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06)
    fig.add_trace(go.Scatter(y=x, name='input', line=dict(color='lightgray')), row=1, col=1)
    fig.add_trace(go.Scatter(y=out[:, 1], name='pred_mean', line=dict(color='navy')), row=1, col=1)
    fig.add_trace(go.Scatter(y=out[:, 0], name='run_length', line=dict(color='crimson')), row=2, col=1)
    fig.update_layout(title='BayesianChangepoint: the run length restarts at each change',
                      xaxis2_title='step', margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```

<!-- HELP_END -->
````

- [ ] **Step 5: Regenerate and run the tests**

Run: `REGEN`, `.venv/bin/python -m pytest tests/test_bayesian_changepoint.py -q 2>&1 | tail -5`, `REGISTRY BayesianChangepoint`, `FRESH`.
Expected: all pass.

- [ ] **Step 6: Tidy, full suite, JavaScript** (`make tidy`, full suite, `JSTEST`). Expected: clean, all pass.

- [ ] **Step 7: Commit**

```bash
git add include/screamer/bayesian_changepoint.h bindings/bindings_fin.cpp include/screamer/common/int_settings.h \
  docs/functions_fin/BayesianChangepoint.md devtools/baselines/BayesianChangepoint.py \
  tests/test_bayesian_changepoint.py tests/param_cases.py devtools/wasm/gen_research_oracle.py \
  js/test/research.test.ts screamer/__init__.py screamer/data/help.json docs/by_group/machine-learning.rst \
  docs/function_index.txt docs/gallery.rst docs/_static/gallery/manifest.json \
  docs/_static/gallery/functions_fin__BayesianChangepoint.webp devtools/wasm/wasm_manifest.json \
  wasm/generated/bindings_wasm.cpp js/src/generated/ops.ts js/src/generated/ops.d.ts \
  js/src/generated/docs_data.ts js/test/docs_oracle.json wasm/smoke/null_oracle.json \
  wasm/smoke/field_axis_oracle.json wasm/smoke/oracle_research.json
git status --short   # add any other path the regeneration changed, by name
git commit -m "feat(regime): BayesianChangepoint (Adams-MacKay online changepoint detection)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PZMDN2ijvnSZ7ZnrJUxnd8"
```

---

### Task 10: `RollingHMM`

**Files:**
- Create: `include/screamer/rolling_hmm.h`, `docs/functions_rolling/RollingHMM.md`, `devtools/baselines/RollingHMM.py`, `tests/test_rolling_hmm.py`
- Modify: `bindings/bindings_rolling.cpp`, `include/screamer/common/int_settings.h` (`{"RollingHMM", "max_iter", 1}`, `{"RollingHMM", "refit_every", 1}`, `{"RollingHMM", "window_size", 10}`), `tests/param_cases.py` (`_ROLLING_AUTO_EXCLUDE`, two array types, `multi_input_definitions` entry), `tests/test_nan_input_compliance.py` (input length from the window), `tests/test_batch_stream_bitwise.py` (`CHOICES["emission"]`), `devtools/wasm/gen_research_oracle.py`, `js/test/research.test.ts`

**Interfaces:**
- Produces: `RollingHMM(window_size=720, refit_every=24, emission="gaussian", max_iter=50, tol=1e-6)`, 1 -> 3, `output_names: [prob, pred_mean, pred_std]`; JavaScript `RollingHMM({ windowSize, refitEvery, emission, maxIter, tol })`.

**Algorithm (spec C1; the header and the baseline below implement exactly this):**

1. *Window and schedule.* Keep the last `W = window_size` finite values. The first fit runs on the step the window first holds `W` values; afterwards a counter of steps since the last fit triggers a refit when it reaches `refit_every`. Steps before the first fit output NaN. A NaN input changes nothing.
2. *Observations.* `"gaussian"`: the `W` window values `y_i`. `"ar1"`: the `W - 1` pairs `(z_i, y_i) = (x_{i-1}, x_i)` of the window. State `k` emits `N(m_k(i), var_k)` with `m_k = mu_k` (`"gaussian"`) or `m_k = c_k + phi_k z_i` (`"ar1"`, `mu_k` holds `c_k`).
3. *Floor and centre.* `center` = window mean, `floor = max(1e-6 * window population variance, 1e-300)`; every variance estimate is `max(estimate, floor)`.
4. *First initialization.* `"gaussian"`: with `med` the median of `|y|` (the mean of the two middle values for even `W`), state 0 is the values with `|y| <= med`, state 1 the rest (both all values when the rest is empty); `mu_k` and `var_k` are the group mean and population variance. `"ar1"`: `rho = sum d_i d_{i-1} / sum d_i^2` over the window deviations `d` from its mean `xm` (0 when the sum of squares is 0); `phi = clip(rho - 0.5, rho + 0.5)` to `[-0.99, 0.99]`, `c_k = xm (1 - phi_k)`, `var_k = sum d^2 / W * (1 - rho^2)`. Both: `pi = (0.5, 0.5)`, `A = [[0.95, 0.05], [0.05, 0.95]]`.
5. *Warm start.* A later fit starts from the previous fit's parameters with `pi` reset to `(0.5, 0.5)`.
6. *E-step (scaled forward-backward).* `l_k(i) = -0.5 (log 2 pi + log var_k) - (y_i - m_k(i))^2 / (2 var_k)`, `mx_i = max_k l_k(i)`, `e_k(i) = exp(l_k(i) - mx_i)`. Forward: `q(1) = pi`, `q(i) = A^T alpha(i - 1)`; `a_k = q_k e_k(i)`, `s_i = a_0 + a_1`, `alpha(i) = a / s_i`; `loglik = sum (log s_i + mx_i)`; a step with `s_i` not positive and finite makes `loglik` NaN. Backward: `beta(N) = (1, 1)`, `beta_j(i) = sum_k A_jk e_k(i + 1) beta_k(i + 1) / s_{i+1}`.
7. *M-step.* `gamma_k(i) = alpha_k(i) beta_k(i) / sum_j alpha_j(i) beta_j(i)`; `pi = gamma(1)` normalized; `X_jk = sum_{i<N} alpha_j(i) A_jk e_k(i + 1) beta_k(i + 1) / s_{i+1}`, row `j` of `A` = `X_j / sum_k X_jk` (kept when that sum is not positive). Per state, with weights `w = gamma_k` and `S0 = sum w` (the state is kept when `S0` is not positive): `"gaussian"`: `mu = center + sum w (y - center) / S0`, `var = sum w (y - mu)^2 / S0`. `"ar1"` (weighted least squares on `yc = y - center`, `zc = z - center`): `Sz, Sy, Szz, Szy` the weighted sums, `den = S0 Szz - Sz^2`, `phi = (S0 Szy - Sz Sy) / den` if `den > 1e-12 S0 Szz` else 0, `c = (Sy - phi Sz) / S0 + center (1 - phi)`, `var = sum w (y - c - phi z)^2 / S0`.
8. *Iteration.* For `it = 0 .. max_iter - 1`: run the E-step; stop if `loglik` is NaN, or if `it > 0` and `loglik - previous loglik < tol`; else run the M-step and remember `loglik`.
9. *Identification.* Swap the states when `var_0 > var_1` (`"gaussian"`) or `phi_0 > phi_1` (`"ar1"`): swap `pi`, the emission parameters, and `A` to `[[A11, A10], [A01, A00]]`.
10. *Filter.* After a fit, a forward pass over the window with the final parameters gives `p = alpha(N)` (`(0.5, 0.5)` if it fails). Between fits, a new value `x` with predecessor `x_prev`: `q = A^T p`, `a_k = q_k exp(l_k(x) - max_k l_k(x))`, `p = a / sum a` (`q / sum q` when the sum is not positive and finite).
11. *Outputs at t.* `prob = p_1`; `q = A^T p`, `m_k = mu_k` or `c_k + phi_k x_t`; `pred_mean = sum q_k m_k`; `pred_std = sqrt(sum q_k (var_k + m_k^2) - pred_mean^2)`, computed as `sqrt(q_0 var_0 + q_1 var_1 + q_0 q_1 (m_0 - m_1)^2)`.

- [ ] **Step 1: Write the baseline** (`devtools/baselines/RollingHMM.py`)

```python
"""Two-state hidden Markov model refit on a trailing window, written as a
direct numpy transcription of the algorithm in include/screamer/rolling_hmm.h:
the deterministic first split, Baum-Welch with a scaled forward-backward pass,
the variance floor, the state ordering, the warm start, and forward filtering
between refits."""
import math

import numpy as np

_LOG2PI = math.log(2.0 * math.pi)
_A0 = ((0.95, 0.05), (0.05, 0.95))


class _Params:
    def __init__(self, pi, A, mu, phi, var):
        self.pi = np.array(pi, dtype=float)
        self.A = np.array(A, dtype=float)
        self.mu = np.array(mu, dtype=float)
        self.phi = np.array(phi, dtype=float)
        self.var = np.array(var, dtype=float)

    def copy(self):
        return _Params(self.pi, self.A, self.mu, self.phi, self.var)


class RollingHMM_numpy:
    def __init__(self, window_size=720, refit_every=24, emission="gaussian", max_iter=50, tol=1e-6):
        self.W, self.R = window_size, refit_every
        self.ar1 = emission == "ar1"
        self.max_iter, self.tol = max_iter, tol

    def _means(self, th, z):
        if self.ar1:
            return th.mu[None, :] + th.phi[None, :] * z[:, None]
        return np.broadcast_to(th.mu, (len(z), 2))

    def _logb(self, th, y, z):
        d = y[:, None] - self._means(th, z)
        return -0.5 * (_LOG2PI + np.log(th.var))[None, :] - d * d / (2.0 * th.var[None, :])

    def _forward(self, th, lb):
        N = len(lb)
        alpha, eb, scale = np.empty((N, 2)), np.empty((N, 2)), np.empty(N)
        ll, prev = 0.0, None
        for i in range(N):
            mx = max(lb[i, 0], lb[i, 1])
            eb[i] = np.exp(lb[i] - mx)
            q = th.pi if i == 0 else prev @ th.A
            a = q * eb[i]
            s = a[0] + a[1]
            if not (s > 0.0 and math.isfinite(s)):
                return math.nan, alpha, eb, scale
            alpha[i] = a / s
            scale[i] = s
            ll += math.log(s) + mx
            prev = alpha[i]
        return ll, alpha, eb, scale

    @staticmethod
    def _backward(th, eb, scale):
        N = len(eb)
        beta = np.ones((N, 2))
        for i in range(N - 2, -1, -1):
            beta[i] = (th.A @ (eb[i + 1] * beta[i + 1])) / scale[i + 1]
        return beta

    def _mstep(self, th, y, z, alpha, beta, eb, scale, center, floor):
        g = alpha * beta
        g = g / g.sum(axis=1, keepdims=True)
        new = th.copy()
        new.pi = g[0] / g[0].sum()
        X = np.zeros((2, 2))
        for i in range(len(y) - 1):
            X += alpha[i][:, None] * th.A * (eb[i + 1] * beta[i + 1])[None, :] / scale[i + 1]
        for j in range(2):
            den = X[j].sum()
            if den > 0.0:
                new.A[j] = X[j] / den
        for k in range(2):
            w = g[:, k]
            S0 = w.sum()
            if not S0 > 0.0:
                continue
            if self.ar1:
                yc, zc = y - center, z - center
                Sz, Sy = (w * zc).sum(), (w * yc).sum()
                Szz, Szy = (w * zc * zc).sum(), (w * zc * yc).sum()
                den = S0 * Szz - Sz * Sz
                phi = (S0 * Szy - Sz * Sy) / den if den > 1e-12 * S0 * Szz else 0.0
                c = (Sy - phi * Sz) / S0 + center * (1.0 - phi)
                r = y - c - phi * z
            else:
                phi = 0.0
                c = center + (w * (y - center)).sum() / S0
                r = y - c
            new.mu[k], new.phi[k] = c, phi
            new.var[k] = max((w * r * r).sum() / S0, floor)
        return new

    def _initial(self, win, floor):
        if self.ar1:
            xm = win.mean()
            d = win - xm
            c0, c1 = (d * d).sum(), (d[1:] * d[:-1]).sum()
            rho = c1 / c0 if c0 > 0.0 else 0.0
            phi = np.clip([rho - 0.5, rho + 0.5], -0.99, 0.99)
            v = max(c0 / len(win) * (1.0 - rho * rho), floor)
            return _Params((0.5, 0.5), _A0, xm * (1.0 - phi), phi, (v, v))
        a = np.abs(win)
        med = np.median(a)
        lo, hi = win[a <= med], win[a > med]
        if len(hi) == 0:
            lo = hi = win
        return _Params((0.5, 0.5), _A0, (lo.mean(), hi.mean()), (0.0, 0.0),
                       (max(lo.var(), floor), max(hi.var(), floor)))

    def _order(self, th):
        key = th.phi if self.ar1 else th.var
        if key[0] > key[1]:
            P = [1, 0]
            th = _Params(th.pi[P], th.A[np.ix_(P, P)], th.mu[P], th.phi[P], th.var[P])
        return th

    def _fit(self, win, warm):
        y, z = (win[1:], win[:-1]) if self.ar1 else (win, win)
        center = win.mean()
        floor = max(1e-6 * win.var(), 1e-300)
        if warm is None:
            th = self._initial(win, floor)
        else:
            th = warm.copy()
            th.pi = np.array([0.5, 0.5])
        ll_prev = -math.inf
        for it in range(self.max_iter):
            ll, alpha, eb, scale = self._forward(th, self._logb(th, y, z))
            if math.isnan(ll) or (it > 0 and not (ll - ll_prev >= self.tol)):
                break
            beta = self._backward(th, eb, scale)
            th = self._mstep(th, y, z, alpha, beta, eb, scale, center, floor)
            ll_prev = ll
        th = self._order(th)
        ll, alpha, _, _ = self._forward(th, self._logb(th, y, z))
        return th, (np.array([0.5, 0.5]) if math.isnan(ll) else alpha[-1].copy())

    def _filter(self, th, p, x, prev):
        m = th.mu + th.phi * prev if self.ar1 else th.mu
        lb = -0.5 * (_LOG2PI + np.log(th.var)) - (x - m) ** 2 / (2.0 * th.var)
        q = p @ th.A
        a = q * np.exp(lb - lb.max())
        s = a.sum()
        return a / s if (s > 0.0 and math.isfinite(s)) else q / q.sum()

    def _outputs(self, th, p, x):
        q = p @ th.A
        m = th.mu + th.phi * x if self.ar1 else th.mu
        mean = q @ m
        var = q @ (th.var + m * m) - mean * mean
        return p[1], mean, math.sqrt(max(var, 0.0))

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        out = np.full((len(x), 3), np.nan)
        buf, th, p, since, prev = [], None, None, 0, 0.0
        for t, v in enumerate(x):
            if math.isnan(v):
                continue
            buf.append(v)
            if len(buf) > self.W:
                buf.pop(0)
            before, prev = prev, v
            if len(buf) < self.W:
                continue
            win = np.array(buf)
            if th is None:
                th, p = self._fit(win, None)
                since = 0
            else:
                since += 1
                if since == self.R:
                    th, p = self._fit(win, th)
                    since = 0
                else:
                    p = self._filter(th, p, v, before)
            out[t] = self._outputs(th, p, v)
        return out
```

(This reference was run before the plan was written: on the simulated series of the tests below it reaches 0.97 state accuracy for `"gaussian"`, median `pred_std` 1.94 and 0.55 in the two states, 0.93 accuracy for `"ar1"`, and slopes 0.58 and -0.37 of `pred_mean` on `x_t`.)

- [ ] **Step 2: Write the failing tests** (`tests/test_rolling_hmm.py`)

```python
import numpy as np
import pytest

from devtools.baselines.RollingHMM import RollingHMM_numpy
from screamer import RollingHMM
from tests.regime_helpers import assert_regimes_agree

A_TRUE = [[0.99, 0.01], [0.02, 0.98]]


def _chain(T, rng):
    s = np.zeros(T, dtype=int)
    for t in range(1, T):
        stay = A_TRUE[s[t - 1]][s[t - 1]]
        s[t] = s[t - 1] if rng.random() < stay else 1 - s[t - 1]
    return s


def _gaussian_series(seed=1, T=4000):
    rng = np.random.default_rng(seed)
    s = _chain(T, rng)
    return rng.standard_normal(T) * np.where(s == 1, 2.0, 0.5), s


def _ar1_series(seed=2, T=4000):
    rng = np.random.default_rng(seed)
    s = _chain(T, rng)
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = (0.6 if s[t] == 1 else -0.4) * x[t - 1] + rng.standard_normal()
    return x, s


def test_gaussian_recovers_the_states_and_variances():
    x, s = _gaussian_series()
    out = np.asarray(RollingHMM(window_size=2000, refit_every=500)(x))
    assert np.isnan(out[:1999]).all() and np.isfinite(out[1999:]).all()
    o, st = out[2000:], s[2000:]
    assert np.mean((o[:, 0] > 0.5) == (st == 1)) > 0.9
    assert 1.7 < np.median(o[o[:, 0] > 0.9, 2]) < 2.3          # high-variance state: sigma 2
    assert 0.4 < np.median(o[o[:, 0] < 0.1, 2]) < 0.7          # low-variance state: sigma 0.5
    assert np.all(np.abs(o[:, 1]) < 0.2)                       # both means near 0


def test_ar1_identifies_the_persistent_state():
    x, s = _ar1_series()
    out = np.asarray(RollingHMM(window_size=2000, refit_every=500, emission="ar1")(x))
    o, st, xt = out[2000:], s[2000:], x[2000:]
    assert np.mean((o[:, 0] > 0.5) == (st == 1)) > 0.8
    hi, lo = o[:, 0] > 0.9, o[:, 0] < 0.1
    assert abs(np.polyfit(xt[hi], o[hi, 1], 1)[0] - 0.6) < 0.1     # state 1: phi 0.6
    assert abs(np.polyfit(xt[lo], o[lo, 1], 1)[0] + 0.4) < 0.1     # state 0: phi -0.4


@pytest.mark.parametrize("emission", ["gaussian", "ar1"])
def test_matches_baseline(emission):
    x, _ = _ar1_series(seed=5, T=700) if emission == "ar1" else _gaussian_series(seed=5, T=700)
    x[[50, 400]] = np.nan
    args = dict(window_size=200, refit_every=60, emission=emission, max_iter=40, tol=1e-8)
    got = np.asarray(RollingHMM(**args)(x))
    want = RollingHMM_numpy(**args)(x)
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-8, equal_nan=True)


@pytest.mark.parametrize("emission", ["gaussian", "ar1"])
def test_outputs_never_use_future_values(emission):
    x, _ = _gaussian_series(seed=7, T=900)
    t0 = 600
    y = x.copy()
    y[t0:] = 10.0 * x[t0:] + 3.0
    op = lambda: RollingHMM(window_size=300, refit_every=50, emission=emission)
    np.testing.assert_array_equal(np.asarray(op()(x))[:t0], np.asarray(op()(y))[:t0])


def test_nan_is_skipped():
    x, _ = _gaussian_series(seed=8, T=500)
    holed = np.insert(x, 300, np.nan)
    a = np.asarray(RollingHMM(window_size=100, refit_every=30)(x))
    b = np.asarray(RollingHMM(window_size=100, refit_every=30)(holed))
    assert np.isnan(b[300]).all()
    np.testing.assert_array_equal(np.delete(b, 300, axis=0), a)


def test_constant_window_stays_finite():
    x = np.r_[np.full(150, 0.25), np.random.default_rng(9).standard_normal(100)]
    for emission in ("gaussian", "ar1"):
        out = np.asarray(RollingHMM(window_size=100, refit_every=20, emission=emission)(x))
        assert np.isfinite(out[99:]).all()


@pytest.mark.parametrize("emission", ["gaussian", "ar1"])
def test_regimes_agree(emission):
    x, _ = _gaussian_series(seed=10, T=260)
    x[[20, 150]] = np.nan
    assert_regimes_agree(lambda: RollingHMM(window_size=50, refit_every=7, emission=emission), x)


def test_settings_are_checked():
    with pytest.raises(ValueError):
        RollingHMM(window_size=9)
    with pytest.raises(ValueError):
        RollingHMM(refit_every=0)
    with pytest.raises(ValueError):
        RollingHMM(max_iter=0)
    with pytest.raises(ValueError):
        RollingHMM(tol=-1.0)
    with pytest.raises(ValueError, match="gaussian.*ar1"):
        RollingHMM(emission="ar2")
```

- [ ] **Step 3: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_rolling_hmm.py -q 2>&1 | tail -5`
Expected: FAIL (`ImportError`).

- [ ] **Step 4: Implement** `include/screamer/rolling_hmm.h`

```cpp
#ifndef SCREAMER_ROLLING_HMM_H
#define SCREAMER_ROLLING_HMM_H

// RollingHMM: a two-state hidden Markov model of a series (normally
// returns), refit on a trailing window by Baum-Welch EM and filtered forward
// one step at a time. 1 -> 3: [prob, pred_mean, pred_std].
//
// Emissions: "gaussian", state k emits N(mu_k, var_k) (W observations, the
// window); "ar1", state k emits x_t = c_k + phi_k x_{t-1} + e_t, e_t ~
// N(0, var_k) (Hamilton's Markov-switching AR(1); the W - 1 pairs of the
// window), with c_k kept in mu_k.
//
// Schedule: the first fit runs on the step the window first holds
// window_size values, then every refit_every steps. The first fit starts
// from a deterministic split ("gaussian": values with |x| at or below the
// median |x| form state 0, the rest state 1; "ar1": phi = rho -/+ 0.5,
// clipped to [-0.99, 0.99], rho the window's lag-1 autocorrelation), later
// ones from the previous fit with the initial distribution reset to
// (0.5, 0.5). EM stops after max_iter M-steps or when the log-likelihood
// gains less than tol. Every variance is floored at 1e-6 times the window
// variance. After a fit the states are ordered so that state 1 has the
// larger variance ("gaussian") or the larger phi ("ar1"), and a forward pass
// over the window gives the filtered probability. Between fits the forward
// recursion runs with fixed parameters on each new value.
//
// Outputs at t use values up to t: prob = P(state 1 | x_1..x_t); with
// next-state probabilities q = A^T p and component means m_k = mu_k or
// c_k + phi_k x_t, pred_mean = sum q_k m_k and pred_std the standard
// deviation of the mixture, sqrt(sum q_k (var_k + m_k^2) - pred_mean^2),
// computed as sqrt(q_0 var_0 + q_1 var_1 + q_0 q_1 (m_0 - m_1)^2).
//
// NaN before the first fit. nan_policy: ignore (a NaN gives a NaN row and
// leaves the window, the schedule, and the filter unchanged). A fit costs
// O(max_iter * window_size); a step between fits O(1). Every buffer is
// allocated at construction.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "screamer/common/finite_settings.h"
#include "screamer/common/float_info.h"
#include "screamer/common/functor_base.h"

namespace screamer {

class RollingHMM : public FunctorBase<RollingHMM, 1, 3> {
public:
    RollingHMM(int window_size = 720, int refit_every = 24,
               const std::string& emission = "gaussian", int max_iter = 50, double tol = 1e-6)
        : W_(check_window(window_size)), refit_every_(refit_every),
          ar1_(parse_emission(emission)), max_iter_(max_iter), tol_(tol)
    {
        require_finite("tol", tol);
        if (refit_every < 1) throw std::invalid_argument("refit_every must be 1 or more.");
        if (max_iter < 1) throw std::invalid_argument("max_iter must be 1 or more.");
        if (tol < 0.0) throw std::invalid_argument("tol must be 0 or more.");
        buf_.assign(W_, 0.0);
        win_.assign(W_, 0.0);
        eb_.assign(2 * W_, 0.0);
        alpha_.assign(2 * W_, 0.0);
        beta_.assign(2 * W_, 0.0);
        scale_.assign(W_, 0.0);
        reset();
    }

    static bool parse_emission(const std::string& s) {
        if (s == "gaussian") return false;
        if (s == "ar1") return true;
        throw std::invalid_argument("emission must be \"gaussian\" or \"ar1\".");
    }

    void reset() override {
        head_ = 0;
        count_ = 0;
        since_ = 0;
        fitted_ = false;
        prev_ = 0.0;
        p_[0] = 0.5;
        p_[1] = 0.5;
        th_ = Params{};
    }

    ResultTuple call(const InputArray& in) override {
        const double x = in[0];
        const double nan = std::numeric_limits<double>::quiet_NaN();
        if (isnan2(x)) return std::make_tuple(nan, nan, nan);
        buf_[head_] = x;
        if (++head_ == W_) head_ = 0;
        if (count_ < W_) ++count_;
        const double prev = prev_;
        prev_ = x;
        if (count_ < W_) return std::make_tuple(nan, nan, nan);
        if (!fitted_) {
            fit();
            since_ = 0;
        } else if (++since_ == refit_every_) {
            fit();
            since_ = 0;
        } else {
            filter_step(x, prev);
        }
        return outputs(x);
    }

private:
    struct Params {
        double pi[2] = {0.5, 0.5};
        double A[2][2] = {{0.95, 0.05}, {0.05, 0.95}};
        double mu[2] = {0.0, 0.0};      // mean ("gaussian") or intercept c ("ar1")
        double phi[2] = {0.0, 0.0};     // AR coefficient ("ar1"); 0 for "gaussian"
        double var[2] = {1.0, 1.0};
        double logv[2] = {0.0, 0.0};    // log(var), from set_logv()
        void set_logv() {
            logv[0] = std::log(var[0]);
            logv[1] = std::log(var[1]);
        }
    };

    static constexpr double kLog2Pi = 1.8378770664093453;   // log(2 pi)

    static std::size_t check_window(int w) {
        if (w < 10) throw std::invalid_argument("window_size must be 10 or more.");
        return static_cast<std::size_t>(w);
    }

    std::size_t observations() const { return ar1_ ? W_ - 1 : W_; }
    const double* ys() const { return win_.data() + (ar1_ ? 1 : 0); }
    const double* zs() const { return win_.data(); }    // ys()[i]'s predecessor ("ar1")

    double log_b(const Params& th, int k, double y, double z) const {
        const double m = ar1_ ? th.mu[k] + th.phi[k] * z : th.mu[k];
        const double d = y - m;
        return -0.5 * (kLog2Pi + th.logv[k]) - d * d / (2.0 * th.var[k]);
    }

    // The window, oldest first.
    void load_window() {
        for (std::size_t i = 0; i < W_; ++i) {
            std::size_t j = head_ + i;
            if (j >= W_) j -= W_;
            win_[i] = buf_[j];
        }
    }

    // Scaled forward pass: alpha_ (filtered probabilities), scale_, and eb_
    // = exp(log b - max_k log b). The log-likelihood, or NaN when a step's
    // mass vanishes.
    double forward(const Params& th, std::size_t N) {
        const double* y = ys();
        const double* z = zs();
        double ll = 0.0;
        for (std::size_t i = 0; i < N; ++i) {
            const double l0 = log_b(th, 0, y[i], z[i]);
            const double l1 = log_b(th, 1, y[i], z[i]);
            const double mx = l0 > l1 ? l0 : l1;
            const double e0 = std::exp(l0 - mx), e1 = std::exp(l1 - mx);
            double q0 = th.pi[0], q1 = th.pi[1];
            if (i > 0) {
                const double* a = &alpha_[2 * (i - 1)];
                q0 = a[0] * th.A[0][0] + a[1] * th.A[1][0];
                q1 = a[0] * th.A[0][1] + a[1] * th.A[1][1];
            }
            const double a0 = q0 * e0, a1 = q1 * e1;
            const double s = a0 + a1;
            if (!(s > 0.0) || !std::isfinite(s)) return std::numeric_limits<double>::quiet_NaN();
            alpha_[2 * i] = a0 / s;
            alpha_[2 * i + 1] = a1 / s;
            scale_[i] = s;
            eb_[2 * i] = e0;
            eb_[2 * i + 1] = e1;
            ll += std::log(s) + mx;
        }
        return ll;
    }

    void backward(const Params& th, std::size_t N) {
        beta_[2 * (N - 1)] = 1.0;
        beta_[2 * (N - 1) + 1] = 1.0;
        for (std::size_t i = N - 1; i-- > 0;) {
            const double n0 = eb_[2 * (i + 1)] * beta_[2 * (i + 1)];
            const double n1 = eb_[2 * (i + 1) + 1] * beta_[2 * (i + 1) + 1];
            const double s = scale_[i + 1];
            beta_[2 * i] = (th.A[0][0] * n0 + th.A[0][1] * n1) / s;
            beta_[2 * i + 1] = (th.A[1][0] * n0 + th.A[1][1] * n1) / s;
        }
    }

    double gamma(std::size_t i, int k) const {
        const double g0 = alpha_[2 * i] * beta_[2 * i];
        const double g1 = alpha_[2 * i + 1] * beta_[2 * i + 1];
        return (k == 0 ? g0 : g1) / (g0 + g1);
    }

    Params m_step(const Params& th, std::size_t N, double center, double floor) const {
        Params nw = th;
        const double* y = ys();
        const double* z = zs();
        const double g00 = gamma(0, 0), g01 = gamma(0, 1);
        nw.pi[0] = g00 / (g00 + g01);
        nw.pi[1] = g01 / (g00 + g01);
        double X[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
        for (std::size_t i = 0; i + 1 < N; ++i) {
            const double s = scale_[i + 1];
            const double n0 = eb_[2 * (i + 1)] * beta_[2 * (i + 1)];
            const double n1 = eb_[2 * (i + 1) + 1] * beta_[2 * (i + 1) + 1];
            for (int j = 0; j < 2; ++j) {
                X[j][0] += alpha_[2 * i + static_cast<std::size_t>(j)] * th.A[j][0] * n0 / s;
                X[j][1] += alpha_[2 * i + static_cast<std::size_t>(j)] * th.A[j][1] * n1 / s;
            }
        }
        for (int j = 0; j < 2; ++j) {
            const double den = X[j][0] + X[j][1];
            if (den > 0.0) {
                nw.A[j][0] = X[j][0] / den;
                nw.A[j][1] = X[j][1] / den;
            }
        }
        for (int k = 0; k < 2; ++k) {
            double S0 = 0.0, Sy = 0.0, Sz = 0.0, Szz = 0.0, Szy = 0.0;
            for (std::size_t i = 0; i < N; ++i) {
                const double w = gamma(i, k);
                const double yc = y[i] - center;
                S0 += w;
                Sy += w * yc;
                if (ar1_) {
                    const double zc = z[i] - center;
                    Sz += w * zc;
                    Szz += w * zc * zc;
                    Szy += w * zc * yc;
                }
            }
            if (!(S0 > 0.0)) continue;
            double c = 0.0, phi = 0.0;
            if (ar1_) {
                const double den = S0 * Szz - Sz * Sz;
                if (den > 1e-12 * S0 * Szz) phi = (S0 * Szy - Sz * Sy) / den;
                c = (Sy - phi * Sz) / S0 + center * (1.0 - phi);
            } else {
                c = center + Sy / S0;
            }
            double ss = 0.0;
            for (std::size_t i = 0; i < N; ++i) {
                const double r = ar1_ ? y[i] - c - phi * z[i] : y[i] - c;
                ss += gamma(i, k) * r * r;
            }
            nw.mu[k] = c;
            nw.phi[k] = phi;
            nw.var[k] = std::max(ss / S0, floor);
        }
        nw.set_logv();
        return nw;
    }

    Params initial(double floor) {
        Params th;
        const double Wd = static_cast<double>(W_);
        if (ar1_) {
            double xm = 0.0;
            for (double v : win_) xm += v;
            xm /= Wd;
            double c0 = 0.0, c1 = 0.0;
            for (std::size_t i = 0; i < W_; ++i) {
                const double d = win_[i] - xm;
                c0 += d * d;
                if (i > 0) c1 += d * (win_[i - 1] - xm);
            }
            const double rho = c0 > 0.0 ? c1 / c0 : 0.0;
            const double v = std::max(c0 / Wd * (1.0 - rho * rho), floor);
            th.phi[0] = std::clamp(rho - 0.5, -0.99, 0.99);
            th.phi[1] = std::clamp(rho + 0.5, -0.99, 0.99);
            for (int k = 0; k < 2; ++k) {
                th.mu[k] = xm * (1.0 - th.phi[k]);
                th.var[k] = v;
            }
        } else {
            // median of |x|, in scale_ (free until the first E-step)
            for (std::size_t i = 0; i < W_; ++i) scale_[i] = std::fabs(win_[i]);
            const std::size_t mid = W_ / 2;
            std::nth_element(scale_.begin(), scale_.begin() + static_cast<std::ptrdiff_t>(mid),
                             scale_.begin() + static_cast<std::ptrdiff_t>(W_));
            double med = scale_[mid];
            if (W_ % 2 == 0)
                med = 0.5 * (*std::max_element(scale_.begin(),
                                               scale_.begin() + static_cast<std::ptrdiff_t>(mid)) + med);
            bool split = false;                      // some |x| above the median
            for (double v : win_) split = split || std::fabs(v) > med;
            for (int k = 0; k < 2; ++k) {
                const auto member = [&](double v) {
                    return !split || ((std::fabs(v) <= med) == (k == 0));
                };
                double cnt = 0.0, mean = 0.0;
                for (double v : win_)
                    if (member(v)) { cnt += 1.0; mean += v; }
                mean /= cnt;
                double ss = 0.0;
                for (double v : win_)
                    if (member(v)) ss += (v - mean) * (v - mean);
                th.mu[k] = mean;
                th.var[k] = std::max(ss / cnt, floor);
            }
        }
        th.set_logv();
        return th;
    }

    void order(Params& th) const {
        const bool swap = ar1_ ? th.phi[0] > th.phi[1] : th.var[0] > th.var[1];
        if (!swap) return;
        std::swap(th.pi[0], th.pi[1]);
        std::swap(th.mu[0], th.mu[1]);
        std::swap(th.phi[0], th.phi[1]);
        std::swap(th.var[0], th.var[1]);
        std::swap(th.logv[0], th.logv[1]);
        std::swap(th.A[0][0], th.A[1][1]);
        std::swap(th.A[0][1], th.A[1][0]);
    }

    void fit() {
        load_window();
        const std::size_t N = observations();
        const double Wd = static_cast<double>(W_);
        double center = 0.0;
        for (double v : win_) center += v;
        center /= Wd;
        double ss = 0.0;
        for (double v : win_) ss += (v - center) * (v - center);
        const double floor = std::max(1e-6 * (ss / Wd), 1e-300);
        Params th;
        if (fitted_) {
            th = th_;
            th.pi[0] = 0.5;
            th.pi[1] = 0.5;
        } else {
            th = initial(floor);
        }
        double ll_prev = -std::numeric_limits<double>::infinity();
        for (int it = 0; it < max_iter_; ++it) {
            const double ll = forward(th, N);
            if (std::isnan(ll) || (it > 0 && !(ll - ll_prev >= tol_))) break;
            backward(th, N);
            th = m_step(th, N, center, floor);
            ll_prev = ll;
        }
        order(th);
        th_ = th;
        fitted_ = true;
        if (std::isnan(forward(th_, N))) {
            p_[0] = 0.5;
            p_[1] = 0.5;
        } else {
            p_[0] = alpha_[2 * (N - 1)];
            p_[1] = alpha_[2 * (N - 1) + 1];
        }
    }

    void filter_step(double x, double prev) {
        const double l0 = log_b(th_, 0, x, prev), l1 = log_b(th_, 1, x, prev);
        const double mx = l0 > l1 ? l0 : l1;
        const double q0 = p_[0] * th_.A[0][0] + p_[1] * th_.A[1][0];
        const double q1 = p_[0] * th_.A[0][1] + p_[1] * th_.A[1][1];
        const double a0 = q0 * std::exp(l0 - mx), a1 = q1 * std::exp(l1 - mx);
        const double s = a0 + a1;
        if (s > 0.0 && std::isfinite(s)) {
            p_[0] = a0 / s;
            p_[1] = a1 / s;
        } else {
            const double t = q0 + q1;
            p_[0] = q0 / t;
            p_[1] = q1 / t;
        }
    }

    ResultTuple outputs(double x) const {
        const double q0 = p_[0] * th_.A[0][0] + p_[1] * th_.A[1][0];
        const double q1 = p_[0] * th_.A[0][1] + p_[1] * th_.A[1][1];
        const double m0 = ar1_ ? th_.mu[0] + th_.phi[0] * x : th_.mu[0];
        const double m1 = ar1_ ? th_.mu[1] + th_.phi[1] * x : th_.mu[1];
        const double mean = q0 * m0 + q1 * m1;
        const double dm = m0 - m1;
        const double var = q0 * th_.var[0] + q1 * th_.var[1] + q0 * q1 * dm * dm;
        return std::make_tuple(p_[1], mean, std::sqrt(var));
    }

    const std::size_t W_;
    const int refit_every_;
    const bool ar1_;
    const int max_iter_;
    const double tol_;
    std::vector<double> buf_, win_, eb_, alpha_, beta_, scale_;
    std::size_t head_ = 0, count_ = 0;
    int since_ = 0;
    bool fitted_ = false;
    double prev_ = 0.0;
    double p_[2] = {0.5, 0.5};
    Params th_;
};

}  // namespace screamer

#endif  // SCREAMER_ROLLING_HMM_H
```

`bindings/bindings_rolling.cpp`: `#include "screamer/rolling_hmm.h"` and

```cpp
    // 1 -> 3 two-state hidden Markov model, refit on a trailing window:
    // (prob, pred_mean, pred_std).
    nb::class_<screamer::RollingHMM, screamer::EvalOp>(m, "RollingHMM")
        .def(nb::init<screamer::IntArg<int>, screamer::IntArg<int>, const std::string&,
                      screamer::IntArg<int>, screamer::FloatArg>(),
             "window_size"_a = 720, "refit_every"_a = 24, "emission"_a = "gaussian",
             "max_iter"_a = 50, "tol"_a = 1e-6)
        .def("__call__", &screamer::functor_call<screamer::RollingHMM>)
        .def("reset", &screamer::RollingHMM::reset, "Reset.");
```

`tests/param_cases.py`:
- add `'RollingHMM'` to `_ROLLING_AUTO_EXCLUDE` (comment `# 1 -> 3; baseline entry in multi_input_definitions`);
- add two generators and their dispatch in `generate_array` (`if array_type == 'regime': return generate_regime_array(array_length)` and `if array_type == 'ar_regime': return generate_ar_regime_array(array_length)`):

```python
def generate_regime_array(array_length):
    """Returns whose scale switches between 0.5 and 2.0 every 50 steps, so a
    two-state model has two separated states to find (on i.i.d. input the
    two fitted variances can tie, and the state order would be arbitrary)."""
    scale = np.where((np.arange(array_length) // 50) % 2 == 0, 0.5, 2.0)
    return np.random.randn(array_length) * scale


def generate_ar_regime_array(array_length):
    """An AR(1) series whose coefficient switches between 0.7 and -0.5 every
    50 steps, for the "ar1" emission of RollingHMM."""
    e = np.random.randn(array_length)
    x = np.zeros(array_length)
    for t in range(1, array_length):
        x[t] = (0.7 if (t // 50) % 2 == 0 else -0.5) * x[t - 1] + e[t]
    return x
```

- in `multi_input_definitions`:

```python
    ( ('RollingHMM',)               , {"window_size": [60], "refit_every": [15], "emission": ["gaussian"],
                                       "max_iter": [30], "tol": [1e-8], "array_type": ["regime"],
                                       "array_length": [300]}),
    ( ('RollingHMM',)               , {"window_size": [60], "refit_every": [15], "emission": ["ar1"],
                                       "max_iter": [30], "tol": [1e-8], "array_type": ["ar_regime"],
                                       "array_length": [300]}),
```

`tests/test_nan_input_compliance.py`: the default `window_size` of `RollingHMM` (720) is longer than the 500-sample input, so its last output would be NaN by design. Add after `_input_for`:

```python
def _n_samples(entry: dict) -> int:
    """500, or long enough for two windows of the default window_size, so an
    operator with a long default window has finite output to check."""
    w = next((p["default"] for p in (entry.get("parameters") or [])
              if p["name"] == "window_size" and isinstance(p["default"], int)), 0)
    return max(500, 2 * w + 100)
```

and use `n_samples=_n_samples(entry)` in place of `n_samples=500` in `test_no_sticky_nan`, `test_nan_at_nan_index`, `test_ignore_nan_is_state_transparent` and `test_ignore_nan_is_state_transparent_per_input` (the NaN index 250 stays).

`tests/test_batch_stream_bitwise.py`: `CHOICES` gains `"emission": ("ar1",),`.

`devtools/wasm/gen_research_oracle.py`: `tol=0.0` runs every iteration, so the oracle does not depend on where the gain crosses the tolerance on one platform:

```python
    ("RollingHMM", {"window_size": 150, "refit_every": 40, "max_iter": 20, "tol": 0.0}, "returns", 1e-6),
    ("RollingHMM", {"window_size": 150, "refit_every": 40, "emission": "ar1", "max_iter": 20, "tol": 0.0},
     "returns", 1e-6),
```

`js/test/research.test.ts`:

```typescript
test("research: RollingHMM is NaN until its first fit", () => {
  const x = Float64Array.from({ length: 30 }, (_, i) => Math.sin(i) * (i % 7 === 0 ? 3 : 1));
  const out = (api as any).RollingHMM({ windowSize: 20, refitEvery: 5 })(x) as { data: Float64Array; shape: number[] };
  assert.deepEqual(out.shape, [30, 3]);
  assert.ok(Array.from(out.data.subarray(0, 19 * 3)).every(Number.isNaN));
  assert.ok(Array.from(out.data.subarray(19 * 3)).every(Number.isFinite));
});
```

`docs/functions_rolling/RollingHMM.md`:

````markdown
---
name: RollingHMM
title: Rolling two-state hidden Markov model
implementation_family: rolling
topics:
- regime
tags:
- hidden markov model
- hmm
- regime switching
- markov switching
- baum-welch
- volatility regime
- hamilton
short: Two-state hidden Markov model of a series, refit on a trailing window and filtered forward
  each step; the probability of the high-variance (or more persistent) state and a one-step predictive
  mean and standard deviation.
inputs: 1
outputs: 3
parameters:
- name: window_size
  type: int
  default: 720
  min: 10
  description: Number of most recent values the model is fit on.
- name: refit_every
  type: int
  default: 24
  min: 1
  description: Steps between fits. Between fits the model filters each new value with fixed
    parameters.
- name: emission
  type: str
  default: gaussian
  enum:
  - gaussian
  - ar1
  description: '"gaussian": each state emits a normal value with its own mean and variance;
    "ar1": each state emits an AR(1) step x_t = c + phi x_{t-1} + e with its own c, phi, and
    variance (Hamilton''s Markov-switching AR(1)).'
- name: max_iter
  type: int
  default: 50
  min: 1
  description: Largest number of EM iterations per fit.
- name: tol
  type: float
  default: 1.0e-06
  min: 0.0
  description: A fit stops when an EM iteration raises the log-likelihood by less than this.
nan_policy: ignore
see_also:
- BayesianChangepoint
- RollingVarianceRatio
- BayesianRegression
- EwStd
output_names:
- prob
- pred_mean
- pred_std
references:
- 'Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series
  and the business cycle. *Econometrica* 57(2), 357-384.'
- 'Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech
  recognition. *Proceedings of the IEEE* 77(2), 257-286.'
---

# `RollingHMM`

## Description

`RollingHMM` is a two-state hidden Markov model of a series, normally
returns. The series is in one of two states, which persist and switch with
the probabilities of a 2 x 2 transition matrix, and each state emits values
from its own distribution:

- `emission="gaussian"`: state `k` emits `N(mu_k, sigma_k^2)`, for example a
  calm and a volatile regime of returns;
- `emission="ar1"`: state `k` emits `x_t = c_k + phi_k x_{t-1} + sigma_k e_t`,
  Hamilton's (1989) Markov-switching AR(1), for example a trending and a
  reverting regime.

The model is fit by Baum-Welch EM on the last `window_size` values, first on
the step the window fills and then every `refit_every` steps, each fit
starting from the previous one. Between fits each new value updates the state
probability with the fitted parameters (the forward filter). After each fit
the states are ordered so that state 1 is the high-variance state
(`"gaussian"`) or the state with the larger `phi` (`"ar1"`), so the
meaning of the output does not change between fits. Each output at `t` uses
values up to `t`.

It returns three columns:

0. `prob`: the filtered probability of state 1 at `t`.
1. `pred_mean`: the mean of the one-step-ahead predictive distribution, a
   mixture of the two states weighted by their probability at `t + 1`.
2. `pred_std`: the standard deviation of that mixture.

The first `window_size - 1` rows are `NaN`. The first fit starts from a
deterministic split (values below and above the median absolute value for
`"gaussian"`; `phi` at the window's lag-1 autocorrelation minus and plus 0.5
for `"ar1"`), so the result is reproducible. Variances are floored at
`1e-6` times the window variance, which keeps a fit finite on a constant
window. A fit costs `O(max_iter * window_size)`; a step between fits is
`O(1)`.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input gives a `NaN` output at that step. The step is skipped: it does not enter the window or update the internal state. See [NaN and warmup](../nan_and_warmup.md).
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import RollingHMM

    rng = np.random.default_rng(8)
    T = 3000
    state = np.zeros(T, dtype=int)
    for t in range(1, T):
        stay = 0.99 if state[t - 1] == 0 else 0.98
        state[t] = state[t - 1] if rng.random() < stay else 1 - state[t - 1]
    r = rng.standard_normal(T) * np.where(state == 1, 2.0, 0.5)
    out = RollingHMM(window_size=1000, refit_every=100)(r)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06)
    fig.add_trace(go.Scatter(y=r, name='returns', line=dict(color='lightgray')), row=1, col=1)
    fig.add_trace(go.Scatter(y=out[:, 0], name='P(volatile state)', line=dict(color='crimson')), row=2, col=1)
    fig.add_trace(go.Scatter(y=state, name='true state', line=dict(color='gray', dash='dot')), row=2, col=1)
    fig.update_layout(title='RollingHMM: the filtered probability of the volatile regime',
                      xaxis2_title='step', margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```

<!-- HELP_END -->
````

- [ ] **Step 5: Regenerate and run the tests**

Run: `REGEN`, `.venv/bin/python -m pytest tests/test_rolling_hmm.py -q 2>&1 | tail -5`, `REGISTRY RollingHMM`, `.venv/bin/python -m pytest tests/test_nan_input_compliance.py -q 2>&1 | tail -3`, `FRESH`.
Expected: all pass. If `test_matches_baseline` differs by more than `rtol=1e-6`, compare the two fits step by step on the first window (initial parameters, then `loglik` per iteration); the two implementations must agree on every formula above, and a difference is a transcription error in one of them, not a tolerance to loosen.

- [ ] **Step 6: Tidy, full suite, JavaScript** (`make tidy`, full suite, `JSTEST`). Expected: clean, all pass.

- [ ] **Step 7: Commit**

```bash
git add include/screamer/rolling_hmm.h bindings/bindings_rolling.cpp include/screamer/common/int_settings.h \
  docs/functions_rolling/RollingHMM.md devtools/baselines/RollingHMM.py tests/test_rolling_hmm.py \
  tests/param_cases.py tests/test_nan_input_compliance.py tests/test_batch_stream_bitwise.py \
  devtools/wasm/gen_research_oracle.py js/test/research.test.ts screamer/__init__.py screamer/data/help.json \
  docs/by_group/machine-learning.rst docs/function_index.txt docs/gallery.rst docs/_static/gallery/manifest.json \
  docs/_static/gallery/functions_rolling__RollingHMM.webp devtools/wasm/wasm_manifest.json \
  wasm/generated/bindings_wasm.cpp js/src/generated/ops.ts js/src/generated/ops.d.ts \
  js/src/generated/docs_data.ts js/test/docs_oracle.json wasm/smoke/null_oracle.json \
  wasm/smoke/field_axis_oracle.json wasm/smoke/oracle_research.json
git status --short   # add any other path the regeneration changed, by name
git commit -m "feat(regime): RollingHMM, a two-state hidden Markov model refit on a trailing window

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PZMDN2ijvnSZ7ZnrJUxnd8"
```

---

### Task 11: `backtest_trades` (A1)

**Files:**
- Create: `include/screamer/detail/round_trips.h` (the trade state machine), `include/screamer/backtest_trade_list.h` (`backtest_trades`, `check_trade_rows`), `docs/functions_fin/backtest_trades.md`, `tests/test_backtest_trades.py`
- Modify: `include/screamer/backtest_stats.h` (use the state machine; no change in numbers), `bindings/bindings_fin.cpp` (`backtest_trades_py` after `int64_array` ~line 1114, `m.def("_backtest_trades", ...)` next to `_backtest_report`), `screamer/backtest.py` (`backtest_trades`, `__all__`), `wasm/backtest_bindings.cpp` (`backtestTrades`), `js/src/backtest.ts` (`backtestTrades`; `engineRows` takes the caller's name), `js/src/index.ts` (export), `docs/functions_fin/backtest_report.md` (`see_also`), `devtools/wasm/gen_research_oracle.py` (`_helpers`), `js/test/research.test.ts`

**Interfaces:**
- Produces: C++ `detail::RoundTrip { entry, exit, side, pnl, cost, held, closed }`, `detail::RoundTripTracker::step(row, pnl, cost, position, on_close)`, `screamer::backtest_trades(const double* eng, size_t T, ptrdiff_t row_stride, ptrdiff_t col_stride) -> std::vector<detail::RoundTrip>`, `screamer::check_trade_rows(ndim, ncols)`.
- Produces: Python `backtest_trades(values, index=None) -> dict` with keys `entry, exit, side, bars, pnl, cost, closed` (int64, int64, int64, int64, float64, float64, bool arrays; `entry`/`exit` take the index's values when `index` is given). JavaScript `backtestTrades(values, { index }) -> { entry, exit, side, bars, pnl, cost: Float64Array; closed: Uint8Array }`.
- Consumes (Task 12): the oracle helper `_engine_output()` defined here.

**Behaviour (spec A1):** rows with a NaN field are skipped. A trade opens when the position leaves zero and closes when it returns to zero; a sign flip closes one trade and opens the next on the same row; resizing without crossing zero stays one trade. `pnl` and `cost` sum the trade's rows from its opening row to its closing row; on a flip row they go to the closing trade. `exit` is the closing row, or `T - 1` for a trade open on the last row (`closed` False). `bars = exit - entry`. A `(T, K, 4)` portfolio output raises.

- [ ] **Step 1: Write the failing tests** (`tests/test_backtest_trades.py`)

```python
import numpy as np
import pytest

import screamer as s
from screamer import backtest_report, backtest_trades

POS = [0, 1, 1, -1, -1, 0, 2, 2]
PNL = [0.0, -0.1, 0.5, 0.3, -0.2, 0.4, -0.1, 0.6]
COST = [0.0, 0.1, 0.0, 0.2, 0.0, 0.1, 0.1, 0.0]


def _hand():
    return np.column_stack([np.cumsum(PNL), PNL, POS, COST]).astype(float)


def test_hand_example():
    tr = backtest_trades(_hand())
    np.testing.assert_array_equal(tr["entry"], [1, 3, 6])
    np.testing.assert_array_equal(tr["exit"], [3, 5, 7])
    np.testing.assert_array_equal(tr["side"], [1, -1, 1])
    np.testing.assert_array_equal(tr["bars"], [2, 2, 1])
    np.testing.assert_allclose(tr["pnl"], [0.7, 0.2, 0.5], rtol=1e-12)    # the flip row's 0.3 closes trade 1
    np.testing.assert_allclose(tr["cost"], [0.3, 0.1, 0.1], rtol=1e-12)
    np.testing.assert_array_equal(tr["closed"], [True, True, False])
    assert tr["closed"].dtype == bool and tr["entry"].dtype == np.int64


def test_index_labels_entry_and_exit():
    tr = backtest_trades(_hand(), index=np.arange(8) * 10)
    np.testing.assert_array_equal(tr["entry"], [10, 30, 60])
    np.testing.assert_array_equal(tr["exit"], [30, 50, 70])
    np.testing.assert_array_equal(tr["bars"], [2, 2, 1])


def test_nan_rows_are_skipped_and_counted_in_bars():
    v = _hand()
    v[2] = np.nan
    tr = backtest_trades(v)
    np.testing.assert_array_equal(tr["entry"], [1, 3, 6])
    np.testing.assert_allclose(tr["pnl"], [0.2, 0.2, 0.5], rtol=1e-12)
    np.testing.assert_array_equal(tr["bars"], [2, 2, 1])


def test_resizing_stays_one_trade_and_no_position_gives_no_trades():
    v = np.column_stack([np.zeros(4), [0.0, 1, 1, 1], [0.0, 1, 3, 0], np.zeros(4)])
    tr = backtest_trades(v)
    np.testing.assert_array_equal(tr["entry"], [1])
    np.testing.assert_array_equal(tr["exit"], [3])
    empty = backtest_trades(np.zeros((5, 4)))
    assert all(len(a) == 0 for a in empty.values())


def _engine_outputs():
    rng = np.random.default_rng(0)
    T = 600
    price = 100 * np.exp(np.cumsum(rng.standard_normal(T) * 0.01))
    target = np.repeat(rng.integers(-2, 3, T // 20), 20).astype(float)
    o = np.r_[price[0], price[:-1]]
    h = np.maximum(o, price) * 1.002
    lo = np.minimum(o, price) * 0.998
    yield "price", s.BacktestPriceTarget(fee=1e-3)(target, price)
    yield "ohlc", s.BacktestOHLCTarget(taker_fee=5e-4)(target, o, h, lo, price)
    yield "ohlcv", s.BacktestOHLCVTarget()(target, o, h, lo, price, rng.uniform(1e3, 1e4, T))
    yield "trades", s.BacktestTradesTarget()(target, price, rng.uniform(1.0, 10.0, T))
    yield "l1", s.BacktestL1Target()(target, price * 0.9995, price * 1.0005, np.full(T, 5.0), np.full(T, 5.0))


@pytest.mark.parametrize("name,out", list(_engine_outputs()))
def test_closed_trades_agree_with_backtest_stats(name, out):
    out = np.asarray(out, dtype=float)
    tr = backtest_trades(out)
    _, summary = backtest_report(out)
    pnl, bars = tr["pnl"][tr["closed"]], tr["bars"][tr["closed"]]
    assert len(pnl) == summary["round_trips"] > 0
    wins = pnl > 0
    np.testing.assert_allclose(wins.mean(), summary["win_rate"], rtol=1e-12)
    np.testing.assert_allclose(pnl[wins].mean(), summary["avg_win"], rtol=1e-9)
    if (~wins).any():
        np.testing.assert_allclose(pnl[~wins].mean(), summary["avg_loss"], rtol=1e-9)
    np.testing.assert_allclose(bars.mean(), summary["avg_holding"], rtol=1e-12)


def test_portfolio_output_raises():
    with pytest.raises(ValueError, match="once per asset"):
        backtest_trades(np.zeros((10, 3, 4)))
    with pytest.raises(ValueError, match=r"\(T, 4\)"):
        backtest_trades(np.zeros((10, 3)))
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_backtest_trades.py -q 2>&1 | tail -5`
Expected: FAIL (`ImportError: cannot import name 'backtest_trades'`).

- [ ] **Step 3: Implement the C++**

`include/screamer/detail/round_trips.h`:

```cpp
#ifndef SCREAMER_DETAIL_ROUND_TRIPS_H
#define SCREAMER_DETAIL_ROUND_TRIPS_H

// The round trips of a backtest engine's position column, shared by
// BacktestStats (its trade statistics) and backtest_trades (the trade list),
// so the two count the same trades. A trade opens when the position leaves
// zero and closes when it returns to zero; a sign flip closes one trade and
// opens the next on the same row. Resizing without crossing zero stays one
// trade. A trade's pnl and cost are the sums from its opening row to its
// closing row; on a flip row they go to the closing trade, because the
// engine reports one pnl per row. `held` counts the rows that held a
// position: every row from the opening row up to the row before the close.

#include <cstddef>

namespace screamer { namespace detail {

struct RoundTrip {
    std::size_t entry = 0, exit = 0;
    double side = 0.0;              // +1 long, -1 short (the opening position's sign)
    double pnl = 0.0, cost = 0.0;
    long long held = 0;
    bool closed = false;
};

class RoundTripTracker {
public:
    void reset() {
        open_ = false;
        prev_pos_ = 0.0;
        cur_ = RoundTrip{};
    }

    // One row without NaN; on_close(trade) for a trade that closes on it.
    template <class OnClose>
    void step(std::size_t row, double pnl, double cost, double position, OnClose&& on_close) {
        const bool was_open = open_;
        const bool flipped = prev_pos_ != 0.0 && position != 0.0 &&
                             ((prev_pos_ > 0.0) != (position > 0.0));
        if (was_open) {
            cur_.pnl += pnl;
            cur_.cost += cost;
            if (position != 0.0 && !flipped) ++cur_.held;   // still held
        }
        if (was_open && (position == 0.0 || flipped)) {
            cur_.exit = row;
            cur_.closed = true;
            on_close(cur_);
            open_ = false;
        }
        if (position != 0.0 && !open_) {
            open_ = true;
            cur_ = RoundTrip{};
            cur_.entry = row;
            cur_.exit = row;
            cur_.side = position > 0.0 ? 1.0 : -1.0;
            cur_.pnl = was_open ? 0.0 : pnl;    // a flip's pnl went to the old trade
            cur_.cost = was_open ? 0.0 : cost;
            cur_.held = 1;
        }
        prev_pos_ = position;
    }

    bool open() const { return open_; }
    const RoundTrip& current() const { return cur_; }

private:
    bool open_ = false;
    double prev_pos_ = 0.0;
    RoundTrip cur_;
};

}} // namespace screamer::detail

#endif // SCREAMER_DETAIL_ROUND_TRIPS_H
```

`include/screamer/backtest_stats.h`: `#include "screamer/detail/round_trips.h"`; delete the members `open_`, `trade_pnl_`, `trade_len_`, `prev_pos_` and `close_trade()`; add `detail::RoundTripTracker tracker_;`; in `reset()` replace `open_ = false; trade_pnl_ = 0.0; trade_len_ = 0; prev_pos_ = 0.0;` with `tracker_.reset();`; replace the block from `// Trade bookkeeping.` to `prev_pos_ = position;` with

```cpp
            // Trade bookkeeping (detail::RoundTripTracker, which
            // backtest_trades shares).
            tracker_.step(0, pnl, cost, position, [this](const detail::RoundTrip& t) {
                ++trades_;
                if (t.pnl > 0.0) { ++wins_; gross_win_ += t.pnl; }
                else gross_loss_ += -t.pnl;
                held_total_ += t.held;
            });
```

(the arithmetic is the old `close_trade` and the old accumulation, in the same order, so every output keeps its bits).

`include/screamer/backtest_trade_list.h`:

```cpp
#ifndef SCREAMER_BACKTEST_TRADE_LIST_H
#define SCREAMER_BACKTEST_TRADE_LIST_H

// backtest_trades: the round trips of a backtest engine's output, one per
// trade, as BacktestStats counts them (detail::RoundTripTracker). Shared by
// the Python backtest_trades and the JavaScript backtestTrades, which only
// label the result. Rows with a NaN field are skipped. A trade still open on
// the last row is listed with exit = T - 1 and closed = false.

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include "screamer/backtest_summary.h"
#include "screamer/detail/round_trips.h"

namespace screamer {

    // One asset's (T, 4) output; a portfolio's (T, K, 4) has one trade list
    // per asset.
    inline void check_trade_rows(std::size_t ndim, std::size_t ncols) {
        if (ndim == 3)
            throw std::invalid_argument(
                "backtest_trades takes one asset's (T, 4) output; call it once per asset "
                "of a (T, K, 4) portfolio output");
        check_engine_rows(ndim, ncols);
    }

    inline std::vector<detail::RoundTrip> backtest_trades(const double* eng, std::size_t T,
                                                          std::ptrdiff_t row_stride,
                                                          std::ptrdiff_t col_stride) {
        std::vector<detail::RoundTrip> out;
        detail::RoundTripTracker tracker;
        for (std::size_t t = 0; t < T; ++t) {
            const double* r = eng + static_cast<std::ptrdiff_t>(t) * row_stride;
            const double equity = r[0], pnl = r[col_stride], position = r[2 * col_stride],
                         cost = r[3 * col_stride];
            if (std::isnan(equity) || std::isnan(pnl) || std::isnan(position) || std::isnan(cost))
                continue;
            tracker.step(t, pnl, cost, position,
                         [&out](const detail::RoundTrip& tr) { out.push_back(tr); });
        }
        if (tracker.open()) {
            detail::RoundTrip tr = tracker.current();
            tr.exit = T - 1;
            tr.closed = false;
            out.push_back(tr);
        }
        return out;
    }

} // namespace screamer

#endif // SCREAMER_BACKTEST_TRADE_LIST_H
```

`bindings/bindings_fin.cpp`: `#include "screamer/backtest_trade_list.h"`; after `int64_array`:

```cpp
// backtest_trades (screamer/backtest.py): the round trips of an engine's
// (T, 4) output, any strides, as (entry, exit, side, bars, pnl, cost,
// closed): int64 arrays except pnl and cost, closed as 0 or 1.
nb::tuple backtest_trades_py(EngineRows v) {
    screamer::check_trade_rows(v.ndim(), v.ndim() == 2 ? v.shape(1) : 0);
    const auto trades = screamer::backtest_trades(
        v.data(), v.shape(0), static_cast<std::ptrdiff_t>(v.stride(0)),
        static_cast<std::ptrdiff_t>(v.stride(1)));
    const std::size_t n = trades.size();
    std::vector<std::int64_t> entry(n), exit(n), side(n), bars(n), closed(n);
    auto pnl = screamer::detail::new_output(n);
    auto cost = screamer::detail::new_output(n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto& tr = trades[i];
        entry[i] = static_cast<std::int64_t>(tr.entry);
        exit[i] = static_cast<std::int64_t>(tr.exit);
        side[i] = tr.side > 0.0 ? 1 : -1;
        bars[i] = static_cast<std::int64_t>(tr.exit - tr.entry);
        closed[i] = tr.closed ? 1 : 0;
        pnl[i] = tr.pnl;
        cost[i] = tr.cost;
    }
    using screamer::detail::make_owned_array;
    return nb::make_tuple(int64_array(entry), int64_array(exit), int64_array(side), int64_array(bars),
                          make_owned_array(std::move(pnl), {n}), make_owned_array(std::move(cost), {n}),
                          int64_array(closed));
}
```

and in `init_bindings_fin`, next to `_backtest_report`: `m.def("_backtest_trades", &backtest_trades_py, "values"_a);`

- [ ] **Step 4: Implement the Python wrapper** (`screamer/backtest.py`)

Import `_backtest_trades` from `.screamer_bindings`, set `__all__ = ["backtest_report", "backtest_trades", "MARKET"]`, and add:

```python
def backtest_trades(values, index=None):
    """One row per round trip of a backtest engine's output.

    ``values`` is the ``(T, 4)`` array a backtest engine emits, with columns
    ``[equity, pnl, position, cost]``. Returns a dict of equal-length numpy
    arrays, one entry per trade:

    - ``entry``: the row of the trade's first nonzero position (the index
      value when ``index`` is given);
    - ``exit``: the row where the position returns to zero or flips sign, or
      the last row for a trade still open (the index value when given);
    - ``side``: +1 long, -1 short;
    - ``bars``: ``exit - entry`` in rows;
    - ``pnl``: the pnl over the trade's rows (the pnl column is after cost);
    - ``cost``: the cost over the trade's rows;
    - ``closed``: False for a trade still open on the last row.

    Trades are counted as ``BacktestStats`` counts them: flat to flat, a sign
    flip closes one trade and opens the next, and the flip row's pnl goes to
    the closing trade. Rows with a NaN field are skipped. A portfolio output
    ``(T, K, 4)`` raises; call this once per asset. The trades come from C++;
    this wrapper only labels them.
    """
    values = _as_float64(values)
    entry, exit_, side, bars, pnl, cost, closed = _backtest_trades(values)
    if index is not None:
        index = np.asarray(index)
        _check_feed_length("backtest_trades", -1, len(values), index.shape[0] if index.ndim else 1)
        entry, exit_ = index[entry], index[exit_]
    return {"entry": entry, "exit": exit_, "side": side, "bars": bars, "pnl": pnl,
            "cost": cost, "closed": closed.astype(bool)}
```

- [ ] **Step 5: Implement JavaScript**

`wasm/backtest_bindings.cpp`: `#include "screamer/backtest_trade_list.h"`; in the anonymous namespace:

```cpp
// The round trips of the engine output `eng` (T rows of `ncols` values,
// row-major) as one Float64Array of 7 n values: entry, exit, side, bars, pnl,
// cost, and closed (1 or 0), n each. The shape is checked first.
emscripten::val backtestTrades(std::uintptr_t eng, std::size_t T, std::size_t ndim, std::size_t ncols) {
    screamer::check_trade_rows(ndim, ncols);
    const auto trades = screamer::backtest_trades(reinterpret_cast<const double*>(eng), T,
                                                  static_cast<std::ptrdiff_t>(ncols), 1);
    const std::size_t n = trades.size();
    std::vector<double> cols(7 * n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto& tr = trades[i];
        cols[i] = static_cast<double>(tr.entry);
        cols[n + i] = static_cast<double>(tr.exit);
        cols[2 * n + i] = tr.side;
        cols[3 * n + i] = static_cast<double>(tr.exit - tr.entry);
        cols[4 * n + i] = tr.pnl;
        cols[5 * n + i] = tr.cost;
        cols[6 * n + i] = tr.closed ? 1.0 : 0.0;
    }
    return emscripten::val(emscripten::typed_memory_view(cols.size(), cols.data()))
        .call<emscripten::val>("slice");
}
```

and `emscripten::function("backtestTrades", &backtestTrades);` in `EMSCRIPTEN_BINDINGS(screamer_backtest)`.

`js/src/backtest.ts`: change `function engineRows(values: unknown)` to `function engineRows(values: unknown, who: string = WHO)` and use `who` in place of `WHO` inside it; add:

```typescript
export interface BacktestTradesOptions {
  /**
   * An index for the rows: `entry` and `exit` hold its values instead of row
   * numbers. Its values must be numbers.
   */
  index?: ArrayLike<number>;
}

export interface BacktestTradesResult {
  entry: Float64Array;
  exit: Float64Array;
  /** +1 long, -1 short. */
  side: Float64Array;
  /** exit - entry, in rows. */
  bars: Float64Array;
  pnl: Float64Array;
  cost: Float64Array;
  /** 0 for a trade still open on the last row, else 1. */
  closed: Uint8Array;
}

/**
 * One row per round trip of a backtest engine's output, as
 * `BacktestStats` counts trades: flat to flat, and a sign flip closes one
 * trade and opens the next. `values` is the `[T, 4]` engine output. Rows with
 * a NaN field are skipped; a trade open on the last row has `exit = T - 1`
 * and `closed = 0`. A `[T, K, 4]` portfolio output throws; call it once per
 * asset. Mirrors Python's `backtest_trades`.
 */
export function backtestTrades(
  values: NdArray | ArrayLike<ArrayLike<number | null>>,
  opts: BacktestTradesOptions | null = {},
): BacktestTradesResult {
  const who = "backtestTrades";
  const M = current();
  const { data, ndim, T, ncols } = engineRows(values, who);
  let inPtr = 0;
  try {
    inPtr = M.allocF64(Math.max(1, data.length));
    M.viewF64(inPtr, data.length).set(data);
    const cols = wasmCall(() => M.backtestTrades(inPtr, T, ndim, ncols)) as Float64Array;
    const n = cols.length / 7;
    const col = (k: number) => cols.slice(k * n, (k + 1) * n);
    const entry = col(0), exit = col(1);
    const o = opts ?? {};
    if (o.index != null) {
      const index = toF64Checked(o.index as ArrayLike<unknown>, who, (i) => `index item ${i}`);
      throwMessage(M.feedLengthMessage(who, -1, T, index.length));
      for (let i = 0; i < n; i++) { entry[i] = index[entry[i]]; exit[i] = index[exit[i]]; }
    }
    return { entry, exit, side: col(2), bars: col(3), pnl: col(4), cost: col(5),
             closed: Uint8Array.from(col(6)) };
  } finally {
    M.freeBuf(inPtr);
  }
}
```

`js/src/index.ts:50`: `export { backtestReport, backtestTrades, MARKET } from "./backtest.js";` (and the matching `type` exports if the file exports option types for `backtestReport`).

- [ ] **Step 6: Docs page and oracle**

`docs/functions_fin/backtest_trades.md`:

````markdown
---
name: backtest_trades
title: backtest_trades
kind: function
short: One row per round trip of a backtest engine's output, with its entry and exit, side,
  length, pnl, and cost.
tags:
- trades
- trade list
- round trip
- holding period
- trade analysis
see_also:
- backtest_report
- BacktestStats
topics:
- backtesting
---

# `backtest_trades`

The list of round trips in a backtest engine's `[equity, pnl, position, cost]`
output, one row per trade. It returns a dict of equal-length numpy arrays:

- `entry`: the row of the trade's first nonzero position;
- `exit`: the row where the position returns to zero or flips sign, or the
  last row for a trade still open;
- `side`: `+1` long, `-1` short;
- `bars`: `exit - entry`, in rows;
- `pnl`: the pnl over the trade's rows, which is after cost;
- `cost`: the cost over the trade's rows;
- `closed`: `False` for a trade still open on the last row.

Trades are the round trips [`BacktestStats`](BacktestStats.md) counts: a
trade opens when the position leaves zero and closes when it returns to zero,
a sign flip closes one trade and opens the next, and resizing without
crossing zero stays one trade. The pnl of a flip row goes to the closing
trade. On the same output, the closed trades give `BacktestStats`'s count,
win rate, average win and loss, and average holding (rows with a `NaN` field
are skipped by both, and count in `bars` but not in `avg_holding`). Group the
rows by their `entry` to see results per month or per year; trade statistics
are not part of [`backtest_report`](backtest_report.md)'s `by=`, because a
trade can span groups.

A portfolio output `(T, K, 4)` raises; call `backtest_trades(values[:, k])`
for each asset. In JavaScript, `backtestTrades(values, { index })` returns the
same keys as typed arrays.

<!-- HELP_END -->

## Signature

`backtest_trades(values, index=None)`

`values` is the `(T, 4)` array a backtest engine emits. With `index`, `entry`
and `exit` hold its values (dates, for example) instead of row numbers.

## Examples

```{eval-rst}
.. exec_code::

    import numpy as np
    from screamer import BacktestPriceTarget, backtest_trades

    rng = np.random.default_rng(0)
    price = 100 + np.cumsum(rng.standard_normal(300))
    target = np.repeat(rng.integers(-1, 2, 15), 20).astype(float)

    trades = backtest_trades(BacktestPriceTarget(fee=1e-3)(target, price))
    for k in range(len(trades["entry"])):
        print(f"{trades['entry'][k]:4d} {trades['exit'][k]:4d} {trades['side'][k]:+d} "
              f"{trades['pnl'][k]:8.3f} {trades['closed'][k]}")
```
````

`docs/functions_fin/backtest_report.md`: add `- backtest_trades` to `see_also`.

`devtools/wasm/gen_research_oracle.py`: replace `_helpers` with

```python
def _engine_output():
    """A BacktestPriceTarget output with long, short, flat, flipping, and
    resized stretches, two NaN rows, and a trade open on the last row."""
    rng = np.random.default_rng(7)
    target = np.repeat([0, 1, 1, -2, 0, 3, -1, 0, 2, 3, 0, -1], 10).astype(float)
    price = 100 + np.cumsum(rng.standard_normal(len(target)))
    out = np.asarray(s.BacktestPriceTarget(fee=1e-3)(target, price), dtype=float)
    out[[33, 34]] = np.nan
    return out


def _helpers():
    """Inputs and results of the backtest helpers; Tasks 12 to 14 add entries."""
    out = _engine_output()
    trades = s.backtest_trades(out)
    return {
        "backtest_trades": {
            "values": _flat(out), "T": len(out),
            "result": {k: _flat(np.asarray(v, dtype=float)) for k, v in trades.items()},
        },
    }
```

`js/test/research.test.ts`:

```typescript
test("research: backtestTrades matches Python", () => {
  const h = oracle.helpers.backtest_trades;
  const values = { data: Float64Array.from(h.values, num), shape: [h.T, 4] };
  const tr = (api as any).backtestTrades(values);
  for (const k of ["entry", "exit", "side", "bars", "pnl", "cost"]) close(tr[k], h.result[k], `backtestTrades ${k}`);
  close(Float64Array.from(tr.closed), h.result.closed, "backtestTrades closed");
});

test("research: backtestTrades rejects a portfolio output", () => {
  assert.throws(() => (api as any).backtestTrades({ data: new Float64Array(24), shape: [2, 3, 4] }),
                /once per asset/);
});
```

- [ ] **Step 7: Regenerate and run the tests**

Run: `REGEN`, then `.venv/bin/python -m pytest tests/test_backtest_trades.py tests/test_backtest.py tests/test_validation.py tests/test_walk_forward.py tests/test_doc_coverage.py -q 2>&1 | tail -5`, then `FRESH`.
Expected: all pass (the existing `BacktestStats` tests unchanged).

- [ ] **Step 8: Tidy, full suite, JavaScript** (`make tidy`, full suite, `JSTEST`). Expected: clean, all pass.

- [ ] **Step 9: Commit**

```bash
git add include/screamer/detail/round_trips.h include/screamer/backtest_trade_list.h include/screamer/backtest_stats.h \
  bindings/bindings_fin.cpp screamer/backtest.py wasm/backtest_bindings.cpp js/src/backtest.ts js/src/index.ts \
  docs/functions_fin/backtest_trades.md docs/functions_fin/backtest_report.md tests/test_backtest_trades.py \
  devtools/wasm/gen_research_oracle.py js/test/research.test.ts screamer/__init__.py screamer/data/help.json \
  docs/by_group/backtesting-risk.rst docs/function_index.txt js/src/generated/docs_data.ts js/test/docs_oracle.json \
  wasm/smoke/oracle_research.json
git status --short   # add any other path the regeneration changed, by name
git commit -m "feat(backtest): backtest_trades, the round trips of an engine output

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PZMDN2ijvnSZ7ZnrJUxnd8"
```

---

### Task 12: `backtest_report(by=)` (A2)

**Files:**
- Modify: `include/screamer/backtest_summary.h` (`kGroupStats`, `check_group_codes`, `backtest_groups`), `bindings/bindings_fin.cpp` (`backtest_groups_py`, `_backtest_groups`, a fifth tuple in `backtest_report_names`), `screamer/backtest.py` (`by=`), `wasm/backtest_bindings.cpp` (`backtestGroupsInto`, a fifth array in `backtestReportNames`), `js/src/backtest.ts` (`by` option, `byGroup` result), `docs/functions_fin/backtest_report.md`, `devtools/wasm/gen_research_oracle.py` (`_helpers`), `js/test/research.test.ts`
- Test: `tests/test_backtest_report_by.py`

**Interfaces:**
- Consumes: `_engine_output()` of `devtools/wasm/gen_research_oracle.py` (Task 11).
- Produces: C++ `screamer::kGroupStats` (`total_pnl, total_cost, turnover, sharpe, max_drawdown, steps`), `screamer::check_group_codes(size_t n_codes, size_t T)`, `screamer::backtest_groups(const double* eng, size_t T, ptrdiff_t row_stride, ptrdiff_t col_stride, const int64_t* codes, size_t G, double periods_per_year, double* out /* G x 6 */)`.
- Produces: Python `backtest_report(values, index=None, periods_per_year=1.0, by=None)`; with `by`, `summary["by_group"]` is a dict from label (order of first appearance) to a dict of the six statistics (`steps` an int). JavaScript `backtestReport(values, { by })` returns `byGroup: Map<number | string, Record<string, number>>` on the result object (the JavaScript summary stays a flat record of numbers).

**Behaviour (spec A2):** rows with a NaN field are skipped. Per group: `total_pnl` and `total_cost` the sums of its rows' pnl and cost; `turnover` the sum of `|position change|` over its rows, each change measured from the previous finite row of the whole output (the first from flat), as `BacktestReport` measures it; `sharpe` the mean over the sample standard deviation of its rows' pnl, times `sqrt(periods_per_year)` (NaN below two rows or without dispersion); `max_drawdown` the worst drop of the cumulative pnl over its rows, in order (`<= 0`); `steps` its row count. Without `by` the output is unchanged.

- [ ] **Step 1: Write the failing tests** (`tests/test_backtest_report_by.py`)

```python
import numpy as np
import pytest

from screamer import BacktestPriceTarget, backtest_report


def _out(T=500, seed=0):
    rng = np.random.default_rng(seed)
    price = 100 + np.cumsum(rng.standard_normal(T))
    target = np.resize(np.repeat(rng.integers(-2, 3, T // 25 + 1), 25), T).astype(float)
    out = np.asarray(BacktestPriceTarget(fee=1e-3)(target, price), dtype=float)
    out[[40, 41]] = np.nan
    return out


def test_hand_example():
    pnl = np.array([1.0, -2.0, 3.0, 0.5, -1.0])
    v = np.column_stack([np.cumsum(pnl), pnl, [1.0, 1, 0, -1, -1], [0.1, 0, 0.1, 0.1, 0]])
    _, summary = backtest_report(v, by=np.array(["b", "b", "a", "b", "a"]))
    g = summary["by_group"]
    assert list(g) == ["b", "a"]
    b, a = g["b"], g["a"]
    np.testing.assert_allclose([b["total_pnl"], b["total_cost"], b["turnover"], b["max_drawdown"]],
                               [-0.5, 0.2, 2.0, -2.0], rtol=1e-12)
    np.testing.assert_allclose(b["sharpe"], np.mean([1, -2, 0.5]) / np.std([1, -2, 0.5], ddof=1), rtol=1e-12)
    np.testing.assert_allclose([a["total_pnl"], a["total_cost"], a["turnover"], a["max_drawdown"]],
                               [2.0, 0.1, 1.0, -1.0], rtol=1e-12)
    np.testing.assert_allclose(a["sharpe"], 1.0 / np.sqrt(8.0), rtol=1e-12)
    assert (b["steps"], a["steps"]) == (3, 2) and isinstance(b["steps"], int)


def test_group_totals_add_up_to_the_ungrouped_totals():
    out = _out()
    _, summary = backtest_report(out, by=np.arange(len(out)) // 100)
    g = summary["by_group"]
    assert list(g) == [0, 1, 2, 3, 4]
    for key in ("total_pnl", "total_cost", "turnover"):
        np.testing.assert_allclose(sum(v[key] for v in g.values()), summary[key], rtol=1e-9, atol=1e-9)
    assert sum(v["steps"] for v in g.values()) == int(np.sum(~np.isnan(out).any(axis=1)))


def test_a_single_label_equals_the_ungrouped_summary():
    out = _out(seed=1)
    _, summary = backtest_report(out, periods_per_year=252, by=np.full(len(out), "all"))
    g = summary["by_group"]["all"]
    for key in ("total_pnl", "total_cost", "turnover", "sharpe", "max_drawdown"):
        np.testing.assert_allclose(g[key], summary[key], rtol=1e-9, atol=1e-9)


def test_month_labels():
    out = _out(T=90)
    days = np.datetime64("2024-01-01") + np.arange(90)
    _, summary = backtest_report(out, by=days.astype("datetime64[M]"))
    assert list(summary["by_group"]) == [np.datetime64("2024-01"), np.datetime64("2024-02"),
                                         np.datetime64("2024-03")]
    assert [v["steps"] for v in summary["by_group"].values()] == [31, 29 - 2, 30]   # rows 40 and 41 are in February


def test_without_by_nothing_changes_and_bad_labels_raise():
    out = _out()
    running, summary = backtest_report(out)
    assert "by_group" not in summary
    running_by, summary_by = backtest_report(out, by=np.zeros(len(out)))
    np.testing.assert_equal({k: v for k, v in summary_by.items() if k != "by_group"}, summary)
    with pytest.raises(ValueError, match="one label per row"):
        backtest_report(out, by=np.zeros(len(out) - 1))
    with pytest.raises(ValueError, match="1-D"):
        backtest_report(out, by=np.zeros((len(out), 2)))
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_backtest_report_by.py -q 2>&1 | tail -5`
Expected: FAIL (`unexpected keyword argument 'by'`).

- [ ] **Step 3: Implement the C++** (append to `include/screamer/backtest_summary.h`, inside `namespace screamer`, with `#include <cstdint>`, `#include <string>`, `#include <vector>`)

```cpp
    // backtest_report(by=): the statistics of each group of rows, in this
    // order.
    inline constexpr std::array<const char*, 6> kGroupStats{{
        "total_pnl", "total_cost", "turnover", "sharpe", "max_drawdown", "steps"}};

    // One group label per engine row.
    inline void check_group_codes(std::size_t n_codes, std::size_t T) {
        if (n_codes != T)
            throw std::invalid_argument("by must hold one label per row: " + std::to_string(T) +
                                        " rows, " + std::to_string(n_codes) + " labels");
    }

    // The statistics of G groups of T engine rows (`row_stride` and
    // `col_stride` apart), row t in group codes[t], written to out[g * 6 + j]
    // in kGroupStats order. Rows with a NaN field are skipped, as
    // BacktestReport skips them. turnover measures each position change from
    // the previous finite row of the whole output (the first from flat), and
    // sharpe is BacktestReport's running Sharpe ratio over the group's rows,
    // so a single group reproduces the ungrouped turnover and sharpe.
    inline void backtest_groups(const double* eng, std::size_t T, std::ptrdiff_t row_stride,
                                std::ptrdiff_t col_stride, const std::int64_t* codes,
                                std::size_t G, double periods_per_year, double* out) {
        if (!(periods_per_year > 0.0) || !std::isfinite(periods_per_year))
            throw std::invalid_argument("periods_per_year must be finite and positive.");
        for (std::size_t t = 0; t < T; ++t)
            if (codes[t] < 0 || static_cast<std::size_t>(codes[t]) >= G)
                throw std::invalid_argument("group codes must lie in [0, groups)");
        struct Acc {
            double pnl = 0.0, cost = 0.0, turnover = 0.0, mean = 0.0, m2 = 0.0, cum = 0.0;
            double peak = -std::numeric_limits<double>::infinity(), worst = 0.0;
            long long n = 0;
        };
        std::vector<Acc> acc(G);
        bool has_prev = false;
        double prev_pos = 0.0;
        for (std::size_t t = 0; t < T; ++t) {
            const double* r = eng + static_cast<std::ptrdiff_t>(t) * row_stride;
            const double equity = r[0], pnl = r[col_stride], pos = r[2 * col_stride],
                         cost = r[3 * col_stride];
            if (std::isnan(equity) || std::isnan(pnl) || std::isnan(pos) || std::isnan(cost)) continue;
            Acc& g = acc[static_cast<std::size_t>(codes[t])];
            g.pnl += pnl;
            g.cost += cost;
            g.turnover += std::abs(has_prev ? pos - prev_pos : pos);
            prev_pos = pos;
            has_prev = true;
            g.n += 1;
            const double delta = pnl - g.mean;
            g.mean += delta / static_cast<double>(g.n);
            g.m2 += delta * (pnl - g.mean);
            g.cum += pnl;
            if (g.cum > g.peak) g.peak = g.cum;
            if (g.cum - g.peak < g.worst) g.worst = g.cum - g.peak;
        }
        const double sqrt_ppy = std::sqrt(periods_per_year);
        for (std::size_t k = 0; k < G; ++k) {
            const Acc& g = acc[k];
            double sharpe = std::numeric_limits<double>::quiet_NaN();
            if (g.n > 1) {
                const double sd = std::sqrt(g.m2 / static_cast<double>(g.n - 1));
                if (sd > 0.0) sharpe = g.mean / sd * sqrt_ppy;
            }
            double* o = out + k * kGroupStats.size();
            o[0] = g.pnl;
            o[1] = g.cost;
            o[2] = g.turnover;
            o[3] = sharpe;
            o[4] = g.worst;
            o[5] = static_cast<double>(g.n);
        }
    }
```

`bindings/bindings_fin.cpp`, after `backtest_report_py`:

```cpp
// backtest_report(by=): the per-group statistics of an engine's (T, 4)
// output for int64 group codes in [0, groups), as a (groups, 6) array.
using GroupCodes = nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

nb::object backtest_groups_py(EngineRows v, GroupCodes codes, std::size_t groups, double periods_per_year) {
    screamer::check_engine_rows(v.ndim(), v.ndim() == 2 ? v.shape(1) : 0);
    const std::size_t T = v.shape(0);
    screamer::check_group_codes(codes.shape(0), T);
    constexpr std::size_t ng = screamer::kGroupStats.size();
    auto out = screamer::detail::new_output(groups * ng);
    screamer::backtest_groups(v.data(), T, static_cast<std::ptrdiff_t>(v.stride(0)),
                              static_cast<std::ptrdiff_t>(v.stride(1)), codes.data(), groups,
                              periods_per_year, out.get());
    return screamer::detail::make_owned_array(std::move(out), {groups, ng});
}
```

In `backtest_report_names` return a fifth element, `names(screamer::kGroupStats)`; in `init_bindings_fin`: `m.def("_backtest_groups", &backtest_groups_py, "values"_a, "codes"_a, "groups"_a, "periods_per_year"_a);`

- [ ] **Step 4: Implement the Python argument** (`screamer/backtest.py`)

Import `_backtest_groups`; unpack five names: `_ENGINE_COLUMNS, _REPORT_COLUMNS, _STATS_COLUMNS, _SUMMARY, _GROUP_STATS = _backtest_report_names()`; add

```python
def _group_codes(by):
    """The labels of ``by`` as int64 codes numbered in order of first
    appearance, and the labels in that order."""
    by = np.asarray(by)
    if by.ndim != 1:
        raise ValueError("backtest_report: by must be a 1-D array of labels, one per row")
    labels, first, inverse = np.unique(by, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order), dtype=np.int64)
    return np.ascontiguousarray(rank[inverse.reshape(-1)]), [labels[j] for j in order]
```

change the signature to `def backtest_report(values, index=None, periods_per_year=1.0, by=None):`, add to the docstring

```
    ``by`` is one label per row (any NumPy dtype: ``datetime64[M]`` months,
    integer years, strings), as pandas ``groupby(by=)``. With it, ``summary``
    gains ``by_group``: a dict from label, in order of first appearance, to
    ``total_pnl``, ``total_cost``, ``turnover``, ``sharpe`` (annualized with
    ``periods_per_year``), ``max_drawdown`` (of the cumulative pnl over the
    group's rows), and ``steps`` (its rows without NaN). Trade statistics are
    not grouped, because a trade can span groups; group the rows of
    ``backtest_trades`` by their entry instead.
```

and replace the `return` with

```python
    summary = dict(zip(_SUMMARY, summary))
    if by is not None:
        codes, labels = _group_codes(by)
        table = _backtest_groups(values, codes, len(labels), periods_per_year)
        summary["by_group"] = {
            label: {**dict(zip(_GROUP_STATS, map(float, row))), "steps": int(row[-1])}
            for label, row in zip(labels, table)}
    return running, summary
```

- [ ] **Step 5: Implement JavaScript**

`wasm/backtest_bindings.cpp`:

```cpp
constexpr std::size_t kGroups = screamer::kGroupStats.size();

// backtest_report(by=): the statistics of `groups` groups of the engine
// output `eng` (T rows of `ncols` values, row-major) for the `n_codes` group
// codes at `codes` (doubles holding integers in [0, groups)), written row by
// row to `out` (groups x kGroupStats). The shape and the code count are
// checked first.
void backtestGroupsInto(std::uintptr_t eng, std::size_t T, std::size_t ndim, std::size_t ncols,
                        std::uintptr_t codes, std::size_t n_codes, std::size_t groups,
                        double periods_per_year, std::uintptr_t out) {
    screamer::check_engine_rows(ndim, ncols);
    screamer::check_group_codes(n_codes, T);
    const double* c = reinterpret_cast<const double*>(codes);
    std::vector<std::int64_t> ic(T);
    for (std::size_t t = 0; t < T; ++t) ic[t] = static_cast<std::int64_t>(c[t]);
    screamer::backtest_groups(reinterpret_cast<const double*>(eng), T,
                              static_cast<std::ptrdiff_t>(ncols), 1, ic.data(), groups,
                              periods_per_year, reinterpret_cast<double*>(out));
}
```

register `emscripten::function("backtestGroupsInto", &backtestGroupsInto);`, and in `backtestReportNames` add `all.set(4, names(screamer::kGroupStats));`.

`js/src/backtest.ts`: `Names` gains `groups: string[]` (read as the fifth array in `reportNames`); `BacktestReportOptions` gains

```typescript
  /**
   * One label per row, numbers or strings (Python's `by=`). The result then
   * holds `byGroup`.
   */
  by?: ArrayLike<number | string>;
```

`BacktestReportResult` gains

```typescript
  /**
   * With `by`: for each label, in order of first appearance, `total_pnl`,
   * `total_cost`, `turnover`, `sharpe`, `max_drawdown`, and `steps`, as
   * Python's `summary["by_group"]`.
   */
  byGroup?: Map<number | string, Record<string, number>>;
```

add

```typescript
// The labels of `by` as codes numbered in order of first appearance.
function groupCodes(by: ArrayLike<number | string>): { codes: Float64Array; labels: (number | string)[] } {
  const codes = new Float64Array(by.length);
  const labels: (number | string)[] = [];
  const seen = new Map<number | string, number>();
  for (let t = 0; t < by.length; t++) {
    const v = by[t];
    if (typeof v !== "number" && typeof v !== "string")
      throw new TypeError(`${WHO}: by item ${t} must be a number or a string, got ${typeof v}`);
    let c = seen.get(v);
    if (c === undefined) { c = labels.length; seen.set(v, c); labels.push(v); }
    codes[t] = c;
  }
  return { codes, labels };
}
```

and in `backtestReport`, before `return { running, summary }` (inside the `try`, with `codesPtr` and `groupPtr` declared next to the other pointers and freed in the `finally`):

```typescript
    let byGroup: Map<number | string, Record<string, number>> | undefined;
    if (o.by != null) {
      const { codes, labels } = groupCodes(o.by);
      const { groups } = reportNames();
      codesPtr = M.allocF64(Math.max(1, codes.length));
      M.viewF64(codesPtr, codes.length).set(codes);
      groupPtr = M.allocF64(Math.max(1, labels.length * groups.length));
      wasmCall(() => M.backtestGroupsInto(inPtr, T, ndim, ncols, codesPtr, codes.length,
                                          labels.length, ppy, groupPtr));
      const g = M.viewF64(groupPtr, labels.length * groups.length);
      byGroup = new Map(labels.map((label, k) =>
        [label, Object.fromEntries(groups.map((name, j) => [name, g[k * groups.length + j]]))]));
    }
    return byGroup === undefined ? { running, summary } : { running, summary, byGroup };
```

- [ ] **Step 6: Docs and oracle**

`docs/functions_fin/backtest_report.md`: the signature becomes `backtest_report(values, index=None, periods_per_year=1.0, by=None)`; add after the `periods_per_year` paragraph:

```markdown
`by` is one label per row, of any NumPy dtype, as in pandas `groupby(by=)`:
`datetime64[M]` months, integer years, or strings. With it, `summary` gains
`by_group`, a dict from label (in order of first appearance) to that group's
`total_pnl`, `total_cost`, `turnover`, `sharpe` (annualized with
`periods_per_year`), `max_drawdown` (of the cumulative pnl over the group's
rows), and `steps`. Rows with a `NaN` field are skipped. Trade statistics are
not grouped, because a trade can span two groups; group the rows of
[`backtest_trades`](backtest_trades.md) by their `entry` instead. In
JavaScript, `backtestReport(values, { by })` returns the groups as `byGroup`,
a `Map` from label to the same six statistics.
```

and a second `exec_code` example:

````markdown
```{eval-rst}
.. exec_code::

    import numpy as np
    from screamer import BacktestPriceTarget, backtest_report

    rng = np.random.default_rng(0)
    days = np.datetime64("2024-01-01") + np.arange(366)
    price = 100 + np.cumsum(rng.standard_normal(366))
    signal = np.sign(rng.standard_normal(366))

    _, summary = backtest_report(BacktestPriceTarget(fee=1e-3)(signal, price),
                                 periods_per_year=252, by=days.astype("datetime64[M]"))
    for month, s in summary["by_group"].items():
        print(month, f"{s['total_pnl']:8.2f} {s['sharpe']:6.2f} {s['steps']:4d}")
```
````

`devtools/wasm/gen_research_oracle.py`: in `_helpers`, before the `return`, compute and add an entry `"by"`:

```python
    labels = (np.arange(len(out)) // 30) * 10
    _, summary = s.backtest_report(out, periods_per_year=252.0, by=labels)
    by = {"labels": labels.tolist(), "periods_per_year": 252.0,
          "groups": [[int(k), {n: _num(v) for n, v in g.items()}]
                     for k, g in summary["by_group"].items()]}
```

(and `"by": by` in the returned dict).

`js/test/research.test.ts` (import `backtestReport`):

```typescript
test("research: backtestReport by matches Python", () => {
  const hb = oracle.helpers.by, ht = oracle.helpers.backtest_trades;
  const values = { data: Float64Array.from(ht.values, num), shape: [ht.T, 4] };
  const { byGroup } = backtestReport(values, { periodsPerYear: hb.periods_per_year, by: hb.labels });
  assert.deepEqual([...byGroup!.keys()], hb.groups.map((g: any) => g[0]));
  for (const [label, stats] of hb.groups) {
    const got = byGroup!.get(label)!;
    for (const [k, v] of Object.entries(stats as Record<string, Num>)) close([got[k]], [v], `by ${label} ${k}`);
  }
});

test("research: backtestReport by with one string label equals the summary", () => {
  const ht = oracle.helpers.backtest_trades;
  const values = { data: Float64Array.from(ht.values, num), shape: [ht.T, 4] };
  const { summary, byGroup } = backtestReport(values, { by: Array(ht.T).fill("all") });
  const g = byGroup!.get("all")!;
  for (const k of ["total_cost", "turnover", "sharpe"]) assert.ok(Math.abs(g[k] - summary[k]) <= 1e-9 * Math.max(1, Math.abs(summary[k])), k);
});
```

- [ ] **Step 7: Regenerate and run the tests**

Run: `REGEN`, `.venv/bin/python -m pytest tests/test_backtest_report_by.py tests/test_backtest.py tests/test_validation.py -q 2>&1 | tail -5`, `FRESH`.
Expected: all pass.

- [ ] **Step 8: Tidy, full suite, JavaScript** (`make tidy`, full suite, `JSTEST`; `js/test/backtest_report.test.ts` must pass unchanged). Expected: clean, all pass.

- [ ] **Step 9: Commit**

```bash
git add include/screamer/backtest_summary.h bindings/bindings_fin.cpp screamer/backtest.py wasm/backtest_bindings.cpp \
  js/src/backtest.ts docs/functions_fin/backtest_report.md tests/test_backtest_report_by.py \
  devtools/wasm/gen_research_oracle.py js/test/research.test.ts screamer/data/help.json \
  js/src/generated/docs_data.ts js/test/docs_oracle.json wasm/smoke/oracle_research.json
git status --short   # add any other path the regeneration changed, by name
git commit -m "feat(backtest): backtest_report(by=) per-group statistics

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PZMDN2ijvnSZ7ZnrJUxnd8"
```

---

### Task 13: `walk_forward(carry=, carry_multiplier=)` (A3)

**Files:**
- Modify: `include/screamer/validation/position_test.h` (`WalkForwardOptions`, `WalkForwardResult`, `detail::carry_price_input`, `detail::apply_carry`, `detail::carry_input`, `walk_forward`), `bindings/bindings_fin.cpp` (`walk_forward_py` and its `m.def`), `screamer/validation.py` (`walk_forward`), `wasm/validation_bindings.cpp` (`walkForwardRun`), `js/src/runtime.ts` (`opName` on every op), `js/src/validation.ts` (`walkForward`, `heapSpecs`), `docs/functions_fin/walk_forward.md`, `devtools/wasm/gen_research_oracle.py`, `js/test/research.test.ts`
- Test: `tests/test_walk_forward.py` (append)

**Interfaces:**
- Produces: `walk_forward(engine, targets, *prices, train, test, metric="sharpe", anchored=False, periods_per_year=1.0, larger_is_better=None, carry=None, carry_multiplier=1.0)`; JavaScript `walkForward(engine, targets, market, { ..., carry, carryMultiplier })`.
- Produces (Task 14 extends): `WalkForwardOptions` fields `std::optional<ArrayShape> carry`, `double carry_multiplier = 1.0`, `std::string engine`; `WalkForwardResult` fields `std::vector<double> output`, `std::vector<std::size_t> output_shape` (the output after carry; empty without carry); `detail::apply_carry(const double* out, size_t T, size_t cols, const double* price, const double* rate, bool shared, double multiplier, double* dst)`; `detail::carry_input(opt, prices, T, assets) -> std::optional<CarryInput>`.

**Behaviour (spec A3):** with `carry` (a per-step rate, `(T,)` for one asset), every candidate's engine output in training and the stitched output in testing pass through `BacktestCarry(multiplier=carry_multiplier)` before scoring. Its price is the engine's last price input: input 0 (`price`) of `BacktestPriceTarget` and `BacktestTradesTarget`, input 3 (`close`) of `BacktestOHLCTarget` and `BacktestOHLCVTarget`, input 1 (`ask`) of `BacktestL1Target`; any other engine with `carry` raises. `output` and `stats` are after carry. `carry=None` changes nothing.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_walk_forward.py`; extend the import line with `BacktestCarry, BacktestL1Target, BacktestOHLCVTarget, BacktestTradesTarget`)

```python
def _rate(T):
    return 1e-3 * np.sin(np.arange(T) / 7.0)


def test_zero_carry_equals_no_carry():
    T = 300
    o, h, l, c = _market(T)
    X = _targets(T)
    a = walk_forward(BacktestOHLCTarget(taker_fee=2e-4), X, o, h, l, c, train=100, test=50)
    b = walk_forward(BacktestOHLCTarget(taker_fee=2e-4), X, o, h, l, c, train=100, test=50,
                     carry=np.zeros(T))
    np.testing.assert_array_equal(a.train_scores, b.train_scores)
    np.testing.assert_array_equal(a.choice, b.choice)
    np.testing.assert_array_equal(a.output, b.output)
    np.testing.assert_equal(a.stats, b.stats)


def test_carry_is_backtest_carry_on_the_close():
    T = 300
    o, h, l, c = _market(T)
    X = _targets(T)
    rate = _rate(T)
    res = walk_forward(BacktestOHLCTarget(taker_fee=2e-4), X, o, h, l, c, train=100, test=50,
                       metric="total_pnl", carry=rate, carry_multiplier=2.0)
    eng = BacktestOHLCTarget(taker_fee=2e-4)
    raw = np.asarray(eng(res.target, o, h, l, c))
    want = np.asarray(BacktestCarry(multiplier=2.0)(*raw.T, c, rate))
    np.testing.assert_allclose(res.output, want, rtol=1e-12, atol=1e-12)
    for k in range(X.shape[1]):
        eng.reset()
        run = np.asarray(eng(np.ascontiguousarray(X[:, k]), o, h, l, c))
        carried = np.asarray(BacktestCarry(multiplier=2.0)(*run.T, c, rate))
        for i, t0 in enumerate(res.fold_start):
            np.testing.assert_allclose(res.train_scores[i, k], np.nansum(carried[t0 - 100:t0, 1]),
                                       rtol=1e-9, atol=1e-12)
    assert res.stats["total_pnl"] == pytest.approx(np.nansum(res.output[100:, 1]), rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("kind", ["ohlcv", "trades", "l1"])
def test_carry_marks_the_last_price_input(kind):
    T = 300
    o, h, l, c = _market(T)
    X = _targets(T)
    rate = _rate(T)
    if kind == "ohlcv":
        make, market, mark = BacktestOHLCVTarget, (o, h, l, c, np.full(T, 1e6)), c
    elif kind == "trades":
        make, market, mark = BacktestTradesTarget, (c, np.full(T, 100.0)), c
    else:
        bid, ask = c * 0.999, c * 1.001
        make, market, mark = BacktestL1Target, (bid, ask, np.full(T, 10.0), np.full(T, 10.0)), ask
    res = walk_forward(make(), X, *market, train=100, test=50, carry=rate)
    raw = np.asarray(make()(res.target, *market))
    np.testing.assert_allclose(res.output, np.asarray(BacktestCarry()(*raw.T, mark, rate)),
                               rtol=1e-12, atol=1e-12)


def test_carry_argument_checks():
    T = 200
    o, h, l, c = _market(T)
    X = _targets(T)
    with pytest.raises(ValueError, match=r"carry must be a \(T,\) series"):
        walk_forward(BacktestOHLCTarget(), X, o, h, l, c, train=50, test=50, carry=np.zeros(T - 1))
    with pytest.raises(ValueError, match="carry_multiplier"):
        walk_forward(BacktestOHLCTarget(), X, o, h, l, c, train=50, test=50, carry=np.zeros(T),
                     carry_multiplier=0.0)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_walk_forward.py -q -k carry 2>&1 | tail -5`
Expected: FAIL (`unexpected keyword argument 'carry'`).

- [ ] **Step 3: Implement the driver** (`include/screamer/validation/position_test.h`; add `#include "screamer/backtest_carry.h"` and `#include <tuple>`)

`WalkForwardOptions` gains:

```cpp
        std::optional<ArrayShape> carry;         // per-step carry rate, the shape of a price input
        double carry_multiplier = 1.0;
        std::string engine;                      // the engine's class name, for the carry's price input
```

`WalkForwardResult` gains:

```cpp
        std::vector<double> output;              // the stitched output after carry; empty without carry
        std::vector<std::size_t> output_shape;
```

In `namespace detail`, after `pick`:

```cpp
        // The market input BacktestCarry marks with: the last price input of
        // each target engine (the close of the OHLC engines, the trade price,
        // the ask of BacktestL1Target).
        inline std::size_t carry_price_input(const std::string& engine, std::size_t n_inputs) {
            struct Mark { const char* engine; std::size_t input; };
            static constexpr Mark kMarks[] = {
                {"BacktestPriceTarget", 0}, {"BacktestOHLCTarget", 3},
                {"BacktestOHLCVTarget", 3}, {"BacktestTradesTarget", 0},
                {"BacktestL1Target", 1}, {"BacktestPortfolioTarget", 0},
                {"BacktestPortfolioOHLCTarget", 3}};
            for (const Mark& m : kMarks) {
                if (engine != m.engine) continue;
                if (m.input >= n_inputs)
                    throw std::invalid_argument("carry needs the engine's price inputs as separate arrays");
                return m.input;
            }
            throw std::invalid_argument(
                "carry needs a target engine whose price input is known (BacktestPriceTarget, "
                "BacktestOHLCTarget, BacktestOHLCVTarget, BacktestTradesTarget, BacktestL1Target, "
                "BacktestPortfolioTarget, or BacktestPortfolioOHLCTarget), got " +
                (engine.empty() ? std::string("another engine") : engine));
        }

        // `out` (T rows of `cols` strategies or assets of 4 values) through
        // BacktestCarry(multiplier) into `dst`, column by column. Column k is
        // marked with price[at] and charged rate[at], at = t for one (T,)
        // series shared by the columns (`shared`), else t * cols + k.
        inline void apply_carry(const double* out, std::size_t T, std::size_t cols,
                                const double* price, const double* rate, bool shared,
                                double multiplier, double* dst) {
            for (std::size_t k = 0; k < cols; ++k) {
                BacktestCarry carry(multiplier);
                for (std::size_t t = 0; t < T; ++t) {
                    const std::size_t at = shared ? t : t * cols + k;
                    const double* row = out + (t * cols + k) * 4;
                    const BacktestCarry::InputArray in{{row[0], row[1], row[2], row[3], price[at], rate[at]}};
                    const auto r = carry.call(in);
                    double* d = dst + (t * cols + k) * 4;
                    d[0] = std::get<0>(r);
                    d[1] = std::get<1>(r);
                    d[2] = std::get<2>(r);
                    d[3] = std::get<3>(r);
                }
            }
        }

        struct CarryInput { const double* price; const double* rate; double multiplier; };

        // The carry of a walk_forward call, checked: a (T,) rate for one asset,
        // (T, assets) for a portfolio, and a finite positive multiplier.
        inline std::optional<CarryInput> carry_input(const WalkForwardOptions& opt,
                                                     const std::vector<ArrayShape>& prices,
                                                     std::size_t T, std::size_t assets) {
            if (!opt.carry) return std::nullopt;
            if (!(opt.carry_multiplier > 0.0) || !std::isfinite(opt.carry_multiplier))
                throw std::invalid_argument("carry_multiplier must be finite and positive");
            const ArrayShape& c = *opt.carry;
            if (opt.portfolio ? !(c.ndim == 2 && c.T == T && c.K == assets)
                              : !(c.ndim == 1 && c.T == T))
                throw std::invalid_argument(opt.portfolio
                    ? "carry must be a (T, K) array, the shape of the price inputs"
                    : "carry must be a (T,) series, the shape of the price inputs");
            const std::size_t m = carry_price_input(opt.engine, prices.size());
            return CarryInput{prices[m].data, c.data, opt.carry_multiplier};
        }
```

In `walk_forward` (single-asset path), after `detail::check_engine_output(full, T, K);`:

```cpp
        const auto carry = detail::carry_input(opt, prices, T, 0);
        std::vector<double> carried;
        const double* fd = full.data;
        if (carry) {
            carried.resize(T * K * 4);
            detail::apply_carry(full.data, T, K, carry->price, carry->rate, true, carry->multiplier,
                                carried.data());
            fd = carried.data();
        }
```

and use `fd` in place of `full.data` in both `detail::window_rows(full.data, ...)` calls of the fold loop. At the end replace

```cpp
        const EngineOutput out = host.run(false, T, 0);
        detail::check_engine_output(out, T, 0);
        host.reset();
        const std::size_t a = train;   // the first test window
        rows.resize((T - a) * 4);
        detail::window_rows(out.data, T, 1, 0, a, T, rows.data(), 4);
```

with

```cpp
        const EngineOutput out = host.run(false, T, 0);
        detail::check_engine_output(out, T, 0);
        host.reset();
        const double* od = out.data;
        if (carry) {
            res.output.resize(T * 4);
            res.output_shape = {T, 4};
            detail::apply_carry(out.data, T, 1, carry->price, carry->rate, true, carry->multiplier,
                                res.output.data());
            od = res.output.data();
        }
        const std::size_t a = train;   // the first test window
        rows.resize((T - a) * 4);
        detail::window_rows(od, T, 1, 0, a, T, rows.data(), 4);
```

`prices[m].data` is read for the mark: the Python binding fills `ArrayShape::data` for every price input (`array_shape`); the JavaScript side copies the market inputs to the heap when `carry` is given (Step 5).

- [ ] **Step 4: Implement the Python side**

`bindings/bindings_fin.cpp` `walk_forward_py`: add parameters `std::optional<BlockIn> carry, double carry_multiplier, const std::string& engine` after `portfolio`, set

```cpp
    if (carry) opt.carry = array_shape(*carry);
    opt.carry_multiplier = carry_multiplier;
    opt.engine = engine;
```

and return the carried output when there is one:

```cpp
    nb::object output = host.last_output();
    if (!r.output.empty()) {
        auto p = screamer::detail::new_output(r.output.size());
        std::copy(r.output.begin(), r.output.end(), p.get());
        output = screamer::detail::make_owned_array(
            std::move(p), std::vector<size_t>(r.output_shape.begin(), r.output_shape.end()));
    }
    return nb::make_tuple(host.last_target(), int64_array(r.choice), int64_array(r.fold_start),
                          screamer::detail::make_owned_array(std::move(r.scores), {r.folds, r.K}),
                          output, stats);
```

and extend its `m.def` with `"carry"_a.none(), "carry_multiplier"_a, "engine"_a`.

`screamer/validation.py` `walk_forward`: add the keyword arguments `carry=None, carry_multiplier: float = 1.0`; before the `_walk_forward` call

```python
    carry = None if carry is None else _as_float64(carry, contiguous=True)
```

and append to the call `carry, _float(carry_multiplier, "walk_forward: carry_multiplier"), type(engine).__name__`. Add to the docstring, before the last paragraph:

```
    ``carry`` is a per-step carry rate (funding, borrow), a ``(T,)`` series,
    with the sign and timing ``BacktestCarry`` documents. With it, every
    candidate's run and the stitched run pass through
    ``BacktestCarry(multiplier=carry_multiplier)`` before they are scored,
    marked with the engine's last price input (the close for the OHLC
    engines, the trade price for ``BacktestTradesTarget``, the ask for
    ``BacktestL1Target``), so ``output`` and ``stats`` are after carry.
```

- [ ] **Step 5: Implement JavaScript**

`js/src/runtime.ts`: next to each `(op as any).numInputs = ...` (three places, ~lines 717, 1079, 1287) add `(op as any).opName = opName(raw);`.

`wasm/validation_bindings.cpp` `walkForwardRun`: add parameters `std::uintptr_t carry, double carry_multiplier, emscripten::val engine` after `portfolio` (`carry` is the address of an `[address, ndim, T, K]` spec, or 0), set

```cpp
    if (carry != 0) opt.carry = array_shapes(carry, 1)[0];
    opt.carry_multiplier = carry_multiplier;
    opt.engine = engine.as<std::string>();
```

and pass the carried output to `done`:

```cpp
    host.js().call("done", JsCallbacks::addr(r.stats.data()), JsCallbacks::addr(folds.data()),
                   JsCallbacks::addr(choice.data()), JsCallbacks::addr(r.scores.get()), r.folds, r.K,
                   JsCallbacks::addr(r.output.data()), r.output.size());
```

`js/src/validation.ts`:
- `WalkForwardOptions` gains

```typescript
  /**
   * Per-step carry rate (funding, borrow), one value per row, with the sign
   * and timing `BacktestCarry` documents. Every run is scored after
   * `BacktestCarry`, marked with the engine's last price input.
   */
  carry?: ArrayLike<number>;
  /** The carry's contract multiplier (default 1). */
  carryMultiplier?: number;
```

- a helper and a `heapSpecs` flag that copies the market data (the driver reads the mark price):

```typescript
// Row-major data of a market input: a series, an NdArray, or T rows of K values.
function marketData(x: any): Float64Array {
  const nd = asNd(x);
  if (nd !== null) return nd.data;
  if (dims(x)[0] === 2) return flattenRows(x, "walkForward", "market").data;
  return Float64Array.from(x as ArrayLike<number>);
}

// Heap copies of the target and of the [address, ndim, T, K] specs of the
// target and the market inputs. The market's data stays in JS unless
// `withData` (the carry reads the mark price in C++).
function heapSpecs(target: Target, market: any[], bufs: number[], withData = false) {
  const data = heapCopy(target.data);
  bufs.push(data);
  const t = heapCopy([data, target.ndim, target.T, target.K]);
  bufs.push(t);
  const m = heapCopy(market.flatMap((x) => {
    let addr = 0;
    if (withData) { addr = heapCopy(marketData(x)); bufs.push(addr); }
    return [addr, ...dims(x)];
  }));
  bufs.push(m);
  return { target: t, market: m };
}
```

- in `walkForward`: add `carryMultiplier: [opts.carryMultiplier, false]` to `noBools`; before the call

```typescript
    const carry = opts.carry == null ? null : Float64Array.from(opts.carry as ArrayLike<number>);
    const spec = heapSpecs(tgt, market, bufs, carry !== null);
    let carrySpec = 0;
    if (carry !== null) {
      const cp = heapCopy(carry);
      bufs.push(cp);
      carrySpec = heapCopy([cp, 1, carry.length, 1]);
      bufs.push(carrySpec);
    }
```

pass `carrySpec, opts.carryMultiplier ?? 1, (engine as any).opName ?? ""` after the `rowOp` argument of `M.walkForwardRun`, and in `done` take `(…, F, K, outPtr: number, outLen: number)` and set `output: outLen > 0 ? { data: M.viewF64(outPtr, outLen).slice(), shape: (asNd(state.out) as NdArray).shape } : state.out`. Update the TSDoc of `walkForward` with the carry sentence of the Python docstring.

- [ ] **Step 6: Docs and oracle**

`docs/functions_fin/walk_forward.md`: the signature gains `carry=None, carry_multiplier=1.0`; add a bullet:

```markdown
- `carry`: a per-step carry rate (perpetual funding, borrow fee), a `(T,)`
  series with the sign and timing [`BacktestCarry`](BacktestCarry.md)
  documents. With it, every candidate's run and the stitched run pass through
  `BacktestCarry(multiplier=carry_multiplier)` before they are scored, marked
  with the engine's last price input: the close for the OHLC engines, the
  trade price for `BacktestTradesTarget`, and the ask for `BacktestL1Target`.
  `output` and `stats` are after carry. `carry_multiplier` is the contract
  multiplier, as in `BacktestCarry`.
```

`devtools/wasm/gen_research_oracle.py`: add, above `_helpers`,

```python
def _market(T):
    """The deterministic OHLC market of tests/test_walk_forward.py."""
    o, h, l, c = [], [], [], []
    p = 100.0
    for t in range(T):
        prev = p
        p *= np.exp(0.01 * np.sin(1.7 * t) * np.cos(0.31 * t * t))
        o.append(prev)
        c.append(p)
        h.append(max(prev, p) * 1.001)
        l.append(min(prev, p) * 0.999)
    return [np.array(x) for x in (o, h, l, c)]


def _candidates(T):
    t = np.arange(T)
    return np.column_stack([np.sign(np.sin(t / 6.0)), np.sign(np.cos(t / 17.0)),
                            ((t // 5) % 3) - 1.0]).astype(float)
```

and in `_helpers`, an entry `"walk_forward_carry"`:

```python
    T = 260
    o, h, l, c = _market(T)
    X = _candidates(T)
    rate = 1e-3 * np.sin(np.arange(T) / 7.0)
    wf = s.walk_forward(s.BacktestOHLCTarget(taker_fee=2e-4), X, o, h, l, c, train=100, test=40,
                        carry=rate, carry_multiplier=2.0)
    walk_forward_carry = {
        "market": [_flat(x) for x in (o, h, l, c)], "targets": _flat(X), "K": X.shape[1],
        "carry": _flat(rate), "carry_multiplier": 2.0, "train": 100, "test": 40,
        "train_scores": _flat(wf.train_scores), "choice": wf.choice.tolist(),
        "output": _flat(wf.output), "stats": {k: _num(v) for k, v in wf.stats.items()}}
```

`js/test/research.test.ts` (import `walkForward`, `BacktestOHLCTarget`):

```typescript
test("research: walkForward with carry matches Python", () => {
  const h = oracle.helpers.walk_forward_carry;
  const T = h.carry.length;
  const market = h.market.map((x: Num[]) => Float64Array.from(x, num));
  const targets = { data: Float64Array.from(h.targets, num), shape: [T, h.K] };
  const res = walkForward(BacktestOHLCTarget({ takerFee: 2e-4 }), targets, market,
    { train: h.train, test: h.test, carry: Float64Array.from(h.carry, num), carryMultiplier: h.carry_multiplier });
  close(res.trainScores.data, h.train_scores, "trainScores");
  assert.deepEqual(Array.from(res.choice), h.choice);
  close(res.output.data, h.output, "output");
  for (const [k, v] of Object.entries(h.stats as Record<string, Num>)) close([(res.stats as any)[k]], [v], `stats ${k}`);
});
```

- [ ] **Step 7: Regenerate and run the tests**

Run: `make install-dev && .venv/bin/python devtools/wasm/gen_research_oracle.py && .venv/bin/python -m pytest tests/test_walk_forward.py tests/test_validation.py -q 2>&1 | tail -5`, then `FRESH`.
Expected: all pass (every existing walk-forward test unchanged).

- [ ] **Step 8: Tidy, full suite, JavaScript** (`make tidy`, full suite, `JSTEST`; `js/test/walk_forward.test.ts` must pass unchanged). Expected: clean, all pass.

- [ ] **Step 9: Commit**

```bash
git add include/screamer/validation/position_test.h bindings/bindings_fin.cpp screamer/validation.py \
  wasm/validation_bindings.cpp js/src/runtime.ts js/src/validation.ts docs/functions_fin/walk_forward.md \
  tests/test_walk_forward.py devtools/wasm/gen_research_oracle.py js/test/research.test.ts \
  screamer/data/help.json js/src/generated/docs_data.ts js/test/docs_oracle.json wasm/smoke/oracle_research.json
git status --short   # add any other path the regeneration changed, by name
git commit -m "feat(validation): walk_forward carry= scores every run after BacktestCarry

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PZMDN2ijvnSZ7ZnrJUxnd8"
```

---

### Task 14: `walk_forward` for portfolio engines (A4)

**Files:**
- Modify: `include/screamer/validation/position_test.h` (`MetricOf::BookWindow`, `EngineHost::metric` gains `width`, `detail::custom_metric`, `WalkForwardOptions::candidates`, `walk_forward_portfolio`, `walk_forward` no longer rejects portfolios), `bindings/bindings_fin.cpp` (`PyEngineHost::metric`, `walk_forward_py` `candidates`), `screamer/validation.py` (`walk_forward`), `wasm/validation_bindings.cpp` (`JsEngineHost::metric`, `walkForwardRun`), `js/src/validation.ts` (`candidates`, `engineCallbacks.metric`, `walkForward`, carry as an NdArray), `docs/functions_fin/walk_forward.md`, `docs/portfolio_backtesting.md`, `devtools/wasm/gen_research_oracle.py`, `js/test/research.test.ts`
- Test: `tests/test_walk_forward.py` (append; replace the portfolio assertion in `test_input_errors`)

**Interfaces:**
- Consumes: Task 13's carry fields and helpers.
- Produces: `walk_forward(portfolio_engine, targets, *prices, ...)` with `targets` `(T, K, C)` (or `(T, K)` for one candidate), prices `(T, K)`, `carry` `(T, K)`; result `target` `(T, K)`, `output` `(T, K, 4)`, `train_scores` `(n_folds, C)`. A callable metric gets one candidate's `(L, K, 4)` window and returns one number. JavaScript `walkForward` takes a `[T, K, C]` or `[T, K]` NdArray target with a portfolio engine.
- Produces: C++ `EngineHost::metric(MetricOf what, double* values, size_t count, size_t rows, size_t width)`; `MetricOf::BookWindow` (one candidate's `(rows, width, 4)` window, one value).

**Behaviour (spec A4):** each candidate `c` runs once over the whole history as a `(T, K)` target; each training window is the `(L, K, 4)` rows of that run, each asset's equity shifted as `detail::window_rows` does, scored as one book (`RunScorer::score`: equity, pnl and cost summed over assets, the gross position, `PortfolioReport` turnover). The stitched `(T, K)` target holds the chosen candidate's `(K,)` targets in each test window and 0 before the first. With K = 1 the scores, choices, and the non-trade statistics equal the single-asset result.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_walk_forward.py`)

```python
def _book(T, K, seed):
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(rng.standard_normal((T, K)) * 0.01, axis=0))


def _book_targets(T, K, C):
    t = np.arange(T)[:, None, None]
    return 0.3 * np.sign(np.sin(t / (5.0 + 3.0 * np.arange(C))[None, None, :]
                                + np.arange(K)[None, :, None]))


def _book_sharpe(out, a, b):
    pnl = np.nansum(out[a:b, :, 1], axis=1)
    return pnl.mean() / pnl.std(ddof=1)


def test_portfolio_scores_selection_and_stitching():
    T, K, C = 300, 3, 4
    prices, X = _book(T, K, 0), _book_targets(T, K, C)
    eng = BacktestPortfolioTarget(fee=1e-4)
    res = walk_forward(eng, X, prices, train=100, test=50)
    assert res.train_scores.shape == (4, C)
    assert res.target.shape == (T, K) and res.output.shape == (T, K, 4)
    for c in range(C):
        eng.reset()
        out = np.asarray(eng(np.ascontiguousarray(X[:, :, c]), prices))
        for i, t0 in enumerate(res.fold_start):
            np.testing.assert_allclose(res.train_scores[i, c], _book_sharpe(out, t0 - 100, t0), rtol=1e-9)
    expect = np.zeros((T, K))
    for t0, c in zip(res.fold_start, res.choice):
        expect[t0:t0 + 50] = X[t0:t0 + 50, :, c]
    np.testing.assert_array_equal(res.target, expect)
    eng.reset()
    np.testing.assert_array_equal(res.output, np.asarray(eng(expect, prices)))


def test_portfolio_with_one_asset_equals_the_single_asset_result():
    T = 300
    _, _, _, c = _market(T)
    X = _targets(T)
    book = lambda: BacktestPortfolioTarget(fee=1e-4, size="units", initial_capital=0.0)  # noqa: E731
    # the two engines agree on one asset
    np.testing.assert_allclose(np.asarray(book()(X[:, :1], c[:, None]))[:, 0, :],
                               np.asarray(BacktestPriceTarget(fee=1e-4)(X[:, 0], c)), rtol=1e-12, atol=1e-12)
    a = walk_forward(BacktestPriceTarget(fee=1e-4), X, c, train=100, test=50)
    b = walk_forward(book(), X[:, None, :], c[:, None], train=100, test=50)
    np.testing.assert_allclose(b.train_scores, a.train_scores, rtol=1e-9, atol=1e-12)
    np.testing.assert_array_equal(b.choice, a.choice)
    np.testing.assert_array_equal(b.target[:, 0], a.target)
    np.testing.assert_allclose(b.output[:, 0, :], a.output, rtol=1e-12, atol=1e-12)
    for key in ("total_pnl", "total_cost", "turnover", "num_trades", "sharpe", "max_drawdown"):
        np.testing.assert_allclose(b.stats[key], a.stats[key], rtol=1e-9, atol=1e-12)


def test_portfolio_callable_metric_gets_one_candidate_book():
    T, K, C = 240, 2, 3
    prices, X = _book(T, K, 1), _book_targets(T, K, C)
    shapes = []

    def book_pnl(out):
        shapes.append(out.shape)
        return float(np.nansum(out[:, :, 1]))

    res = walk_forward(BacktestPortfolioTarget(), X, prices, train=80, test=40, metric=book_pnl)
    assert shapes and all(sh[1:] == (K, 4) for sh in shapes)
    eng = BacktestPortfolioTarget()
    for c in range(C):
        eng.reset()
        out = np.asarray(eng(np.ascontiguousarray(X[:, :, c]), prices))
        for i, t0 in enumerate(res.fold_start):
            np.testing.assert_allclose(res.train_scores[i, c], np.nansum(out[t0 - 80:t0, :, 1]),
                                       rtol=1e-9, atol=1e-12)


def test_portfolio_carry_per_asset():
    T, K, C = 240, 3, 2
    prices, X = _book(T, K, 2), _book_targets(T, K, C)
    rate = 1e-3 * np.cos(np.arange(T)[:, None] / 5.0 + np.arange(K))
    res = walk_forward(BacktestPortfolioTarget(), X, prices, train=80, test=40, carry=rate)
    raw = np.asarray(BacktestPortfolioTarget()(res.target, prices))
    want = np.stack([np.asarray(BacktestCarry()(*raw[:, k, :].T, prices[:, k], rate[:, k]))
                     for k in range(K)], axis=1)
    np.testing.assert_allclose(res.output, want, rtol=1e-12, atol=1e-12)


def test_portfolio_argument_checks():
    T = 120
    prices = _book(T, 3, 3)
    with pytest.raises(ValueError, match=r"\(T, K\) or \(T, K, C\)"):
        walk_forward(BacktestPortfolioTarget(), np.ones(T), prices, train=40, test=40)
    with pytest.raises(ValueError, match="price inputs"):
        walk_forward(BacktestPortfolioTarget(), np.ones((T, 3, 2)), prices[:, :2], train=40, test=40)
    with pytest.raises(ValueError, match=r"carry must be a \(T, K\) array"):
        walk_forward(BacktestPortfolioTarget(), np.ones((T, 3, 2)), prices, train=40, test=40,
                     carry=np.zeros(T))
```

In `test_input_errors`, replace

```python
    with pytest.raises(ValueError, match="portfolio"):
        walk_forward(BacktestPortfolioTarget(), X, np.ones((T, 3)), train=10, test=10)
```

with

```python
    with pytest.raises(ValueError, match=r"\(T, K\) or \(T, K, C\)"):
        walk_forward(BacktestPortfolioTarget(), X[:, 0], np.ones((T, 3)), train=10, test=10)
```

If the first assertion of `test_portfolio_with_one_asset_equals_the_single_asset_result` fails, the two engines fill differently at their defaults: read `docs/functions_fin/BacktestPortfolioTarget.md` and `docs/functions_fin/BacktestPriceTarget.md` and set the portfolio engine so that one asset matches (the spec requires this equivalence; keep the test).

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_walk_forward.py -q -k "portfolio or input_errors" 2>&1 | tail -5`
Expected: FAIL (`walk_forward needs a single-asset engine`).

- [ ] **Step 3: Implement the driver** (`include/screamer/validation/position_test.h`)

`MetricOf` gains `BookWindow,   // one candidate's (rows, width, 4) window of a portfolio, one value`. `EngineHost::metric` becomes

```cpp
        // The custom metric of `what`: `count` values, one per strategy (one
        // for a book). ... `rows` and `width` are the window's shape,
        // (rows, width, 4), for Window and BookWindow.
        virtual std::size_t metric(MetricOf what, double* values, std::size_t count,
                                   std::size_t rows, std::size_t width) = 0;
```

`detail::custom_metric` gains `std::size_t width = 0` after `rows` and passes it on; the `MetricOf::Window` call in `walk_forward` passes `K, len, K`. `WalkForwardOptions`: replace the `portfolio` comment with `// a portfolio engine: targets (T, K * candidates)` and add `std::size_t candidates = 1;   // C, for a portfolio's (T, K, C) targets flattened to (T, K * C)`.

Add before `walk_forward`:

```cpp
    // walk_forward for a portfolio engine. `targets` is (T, A * C), the
    // row-major (T, A, C) block of C candidates for A assets; the prices are
    // (T, A). Each candidate runs once over the whole history, and each
    // training window, (L, A, 4) with each asset's equity shifted as
    // window_rows does, is scored as one book (RunScorer::score). The
    // stitched (T, A) target runs once at the end.
    inline WalkForwardResult walk_forward_portfolio(EngineHost& host, const ArrayShape& targets,
                                                    const std::vector<ArrayShape>& prices,
                                                    const WalkForwardOptions& opt,
                                                    const Metric& metric, std::size_t train,
                                                    std::size_t test) {
        const std::size_t T = targets.T, C = opt.candidates;
        if (C == 0 || targets.K == 0 || targets.K % C != 0)
            throw std::invalid_argument("targets must be a (T, K) or (T, K, C) array");
        const std::size_t A = targets.K / C;
        for (const ArrayShape& p : prices)
            if (p.ndim != 2 || p.T != T || p.K != A)
                throw std::invalid_argument(
                    "a portfolio engine needs (T, K) price inputs, the shape of each candidate's target");
        const auto carry = detail::carry_input(opt, prices, T, A);

        WalkForwardResult res;
        res.K = C;
        res.folds = (T - train + test - 1) / test;
        const std::size_t F = res.folds;
        res.fold_start.resize(F);
        res.choice.resize(F);
        res.scores.reset(new double[F * C ? F * C : 1]);
        for (std::size_t i = 0; i < F; ++i) res.fold_start[i] = static_cast<std::int64_t>(train + i * test);

        RunScorer scorer(opt.periods_per_year);
        std::vector<double> carried, rows, custom;
        // Rows [a, b) of every asset of a (T, A, 4) output, as (b - a, A, 4).
        const auto book = [&](const double* o, std::size_t a, std::size_t b, double* dst) {
            for (std::size_t k = 0; k < A; ++k) detail::window_rows(o, T, A, k, a, b, dst + k * 4, A * 4);
        };
        const auto score = [&](const double* w, std::size_t len) {
            EngineOutput win;
            win.data = w;
            win.ndim = 3;
            win.shape[0] = len;
            win.shape[1] = A;
            win.shape[2] = 4;
            return scorer.score(win)[metric.stat];
        };
        for (std::size_t c = 0; c < C; ++c) {
            double* buf = host.target_buffer(T * A);
            for (std::size_t t = 0; t < T; ++t)
                for (std::size_t k = 0; k < A; ++k) buf[t * A + k] = targets.data[(t * A + k) * C + c];
            const EngineOutput out = host.run(false, T, A);
            detail::check_book_output(out, T, A);
            const double* o = out.data;
            if (carry) {
                carried.resize(T * A * 4);
                detail::apply_carry(o, T, A, carry->price, carry->rate, false, carry->multiplier,
                                    carried.data());
                o = carried.data();
            }
            for (std::size_t i = 0; i < F; ++i) {
                const std::size_t t0 = train + i * test, a = opt.anchored ? 0 : t0 - train, len = t0 - a;
                double v;
                if (metric.custom()) {
                    double* w = host.window_buffer(len * A * 4);
                    book(o, a, t0, w);
                    v = detail::custom_metric(host, MetricOf::BookWindow, custom, 1, len, A)[0];
                } else {
                    rows.resize(len * A * 4);
                    book(o, a, t0, rows.data());
                    v = score(rows.data(), len);
                }
                res.scores[i * C + c] = v;
            }
        }
        for (std::size_t i = 0; i < F; ++i) res.choice[i] = detail::pick(res.scores.get() + i * C, C, metric.larger);

        double* target = host.target_buffer(T * A);
        std::fill(target, target + T * A, 0.0);
        for (std::size_t i = 0; i < F; ++i) {
            if (res.choice[i] < 0) continue;
            const std::size_t c = static_cast<std::size_t>(res.choice[i]), t0 = train + i * test;
            for (std::size_t t = t0; t < std::min(t0 + test, T); ++t)
                for (std::size_t k = 0; k < A; ++k) target[t * A + k] = targets.data[(t * A + k) * C + c];
        }
        const EngineOutput out = host.run(false, T, A);
        detail::check_book_output(out, T, A);
        host.reset();
        const double* o = out.data;
        if (carry) {
            res.output.resize(T * A * 4);
            res.output_shape = {T, A, 4};
            detail::apply_carry(out.data, T, A, carry->price, carry->rate, false, carry->multiplier,
                                res.output.data());
            o = res.output.data();
        }
        rows.resize((T - train) * A * 4);
        book(o, train, T, rows.data());
        EngineOutput win;
        win.data = rows.data();
        win.ndim = 3;
        win.shape[0] = T - train;
        win.shape[1] = A;
        win.shape[2] = 4;
        res.stats = scorer.score(win);
        return res;
    }
```

In `walk_forward`: delete the `if (opt.portfolio) throw ...;` block; keep the metric, `targets.ndim`, `K == 0`, `train` and `test` checks where they are, move the two single-asset price loops (`p.ndim != 1`, `p.T != T`) below the `train`/`test` computation, and insert before them

```cpp
        if (opt.portfolio) return walk_forward_portfolio(host, targets, prices, opt, metric, train, test);
```

Update the comment above `walk_forward` with one paragraph on portfolio engines.

- [ ] **Step 4: Implement the bindings and the Python wrapper**

`bindings/bindings_fin.cpp` `PyEngineHost::metric`:

```cpp
    std::size_t metric(sv::MetricOf what, double* values, std::size_t count,
                       std::size_t rows, std::size_t width) override {
        const bool window = what == sv::MetricOf::Window || what == sv::MetricOf::BookWindow;
        nb::object arg = window
            ? screamer::detail::make_owned_array(std::move(window_buf_), {rows, width, 4})
            : last_obj_;
        return metric_values(metric_(arg), values, count);
    }
```

`walk_forward_py`: the parameter list becomes `(..., double periods_per_year, bool portfolio, std::size_t candidates, std::optional<BlockIn> carry, double carry_multiplier, const std::string& engine)`, with `opt.candidates = candidates;`; its `m.def` lists `"portfolio"_a, "candidates"_a, "carry"_a.none(), "carry_multiplier"_a, "engine"_a`. The other callers of `metric` in this file (if any) pass `width` 0.

`screamer/validation.py` `walk_forward`, replacing the target preparation:

```python
    name, custom = _metric_label(metric)
    portfolio = isinstance(engine, RowOp)
    targets = _as_float64(targets)
    candidates = 1
    if portfolio:
        if targets.ndim == 2:
            targets = targets[:, :, None]
        if targets.ndim != 3:
            raise ValueError("walk_forward: a portfolio engine needs (T, K) or (T, K, C) targets")
        candidates = targets.shape[2]
        targets = np.ascontiguousarray(targets).reshape(targets.shape[0], -1)
    elif targets.ndim == 1:
        targets = targets[:, None]
    targets = np.ascontiguousarray(targets)
```

and the call passes `portfolio, candidates, carry, ...` in that order (replacing `isinstance(engine, RowOp)`). Docstring: replace "Portfolio engines are not supported." with

```
    Portfolio engines (``BacktestPortfolioTarget``,
    ``BacktestPortfolioOHLCTarget``) take ``(T, K, C)`` targets, C candidates
    for K assets (``(T, K)`` is one candidate), and ``(T, K)`` prices. Each
    window is scored on the book, the pnl summed across assets; ``target`` is
    ``(T, K)``, ``output`` ``(T, K, 4)``, and a callable metric gets one
    candidate's ``(L, K, 4)`` window and returns one number. ``carry`` is
    then ``(T, K)``.
```

- [ ] **Step 5: Implement JavaScript**

`wasm/validation_bindings.cpp` `JsEngineHost::metric`:

```cpp
    std::size_t metric(sv::MetricOf what, double* values, std::size_t count,
                       std::size_t rows, std::size_t width) override {
        res_.resize(3 * count);
        js_.call("metric", static_cast<int>(what), count, rows, width,
                 JsCallbacks::addr(window_.data()), JsCallbacks::addr(res_.data()));
        JsCallbacks::metric_values(res_.data(), values, count);
        return count;
    }
```

(update the comment block above `JsEngineHost` to `metric(what, count, rows, width, windowPtr, resPtr)` with `3: one candidate's (rows, width, 4) book window`). `walkForwardRun` gains `double candidates` after `portfolio`: `opt.candidates = static_cast<std::size_t>(candidates);`.

`js/src/validation.ts`:
- `engineCallbacks.metric` takes `(what, count, rows, width, windowPtr, resPtr)` and builds

```typescript
        const arg = what === 3
          ? { data: M.viewF64(windowPtr, rows * width * 4).slice(), shape: [rows, width, 4] }
          : what === 2     // a fresh view each time: the metric may grow the heap
          ? strategyOf({ data: M.viewF64(windowPtr, rows * count * 4), shape: [rows, count, 4] }, j)
          : what === 1 || last.shape.length === 2 ? last : strategyOf(last, j);
```

- `candidates(targets, portfolio)` returns `C` as well:

```typescript
// walkForward: K candidate columns, a (T,) or (T, K) NdArray, or one series;
// always (T, K) for the driver. A portfolio engine takes a [T, K, C] or
// [T, K] NdArray, or T rows of K values, flattened to (T, K * C).
function candidates(targets: any, portfolio: boolean): Target & { C: number } {
  const nd = asNd(targets);
  if (portfolio) {
    if (nd !== null && (nd.shape.length === 2 || nd.shape.length === 3)) {
      const [T, K] = nd.shape, C = nd.shape.length === 3 ? nd.shape[2] : 1;
      return { data: nd.data, ndim: 2, T, K: K * C, original: nd, C };
    }
    if (isColumns(targets)) {
      const { data, T, K } = flattenRows(targets, "walkForward", "targets");
      return { data, ndim: 2, T, K, original: { data, shape: [T, K] }, C: 1 };
    }
    throw new RangeError("walkForward: a portfolio engine needs (T, K) or (T, K, C) targets");
  }
  ...the existing body, each return with C: 1...
}
```

- `walkForward`: `const portfolio = !!(engine as any).rowOp; const tgt = candidates(targets, portfolio);`; the carry may be an NdArray (`[T, K]` for a portfolio):

```typescript
    let carrySpec = 0;
    if (opts.carry != null) {
      const cnd = asNd(opts.carry);
      const cdata = cnd !== null ? cnd.data : Float64Array.from(opts.carry as ArrayLike<number>);
      const cp = heapCopy(cdata);
      bufs.push(cp);
      carrySpec = cnd !== null
        ? heapCopy([cp, cnd.shape.length, cnd.shape[0] ?? 0, cnd.shape[1] ?? 1])
        : heapCopy([cp, 1, cdata.length, 1]);
      bufs.push(carrySpec);
    }
```

(replacing Task 13's carry block; `heapSpecs(tgt, market, bufs, opts.carry != null)`), and the call passes `portfolio, tgt.C, carrySpec, opts.carryMultiplier ?? 1, (engine as any).opName ?? ""`. `WalkForwardOptions.carry` becomes `ArrayLike<number> | NdArray`; `WalkForwardResult.target` becomes `Float64Array | NdArray` with the TSDoc "`[T, K]` for a portfolio engine", `output` "`(T, 4)`, or `(T, K, 4)` for a portfolio engine", `trainScores` "`(nFolds, C)`". Update the `walkForward` TSDoc with the portfolio paragraph of the Python docstring.

- [ ] **Step 6: Docs and oracle**

`docs/functions_fin/walk_forward.md`: the `engine` bullet becomes "a target-position engine, such as `BacktestOHLCTarget` or `BacktestPriceTarget`, or a portfolio engine, `BacktestPortfolioTarget` or `BacktestPortfolioOHLCTarget`"; the `targets` bullet adds "For a portfolio engine, `(T, K, C)`: C candidates for K assets (`(T, K)` is one candidate), with `(T, K)` prices."; add after the fold paragraphs:

```markdown
With a portfolio engine, each candidate is a `(T, K)` target for the whole
book. Every window is scored on the book: equity, pnl and cost summed across
the assets, as [`position_test`](position_test.md) scores a portfolio. The
result's `target` is `(T, K)` and `output` `(T, K, 4)`, a callable metric gets
one candidate's `(L, K, 4)` window and returns one number, and `carry` is a
`(T, K)` rate, one column per asset.
```

and remove "Portfolio engines are not supported." `docs/portfolio_backtesting.md`: in the section on validation (`git grep -n "position_test\|walk_forward" docs/portfolio_backtesting.md` finds it; if there is none, at the end before the references), add one paragraph with an `exec_code` example:

````markdown
`walk_forward` selects between candidate weightings of the book. Pass the
candidates on a last axis, `(T, K, C)`; each training window is scored on the
summed pnl of the book.

```{eval-rst}
.. exec_code::

    import numpy as np
    from screamer import BacktestPortfolioTarget, CrossRank, RollingMean, walk_forward

    rng = np.random.default_rng(0)
    prices = 100 * np.exp(np.cumsum(rng.standard_normal((750, 5)) * 0.01, axis=0))
    returns = np.diff(np.log(prices), axis=0, prepend=np.nan)
    candidates = np.stack([np.asarray(CrossRank()(np.asarray(RollingMean(w)(returns)))) - 0.5
                           for w in (20, 60, 120)], axis=2) * 0.4
    res = walk_forward(BacktestPortfolioTarget(fee=5e-4), np.nan_to_num(candidates), prices,
                       train=250, test=125, periods_per_year=252)
    print("choices:", res.choice, " out-of-sample sharpe:", round(res.stats["sharpe"], 3))
```
````

(Check that `CrossRank()` returns ranks in `[0, 1]` by reading `docs/functions_cross/CrossRank.md` before writing the example; adjust the centring so the weights are centred on 0.)

`devtools/wasm/gen_research_oracle.py` `_helpers`: add an entry `"walk_forward_portfolio"`:

```python
    Tp, Kp, Cp = 240, 3, 2
    rng = np.random.default_rng(11)
    prices = 100 * np.exp(np.cumsum(rng.standard_normal((Tp, Kp)) * 0.01, axis=0))
    t = np.arange(Tp)[:, None, None]
    Xp = 0.3 * np.sign(np.sin(t / (5.0 + 3.0 * np.arange(Cp))[None, None, :] + np.arange(Kp)[None, :, None]))
    ratep = 1e-3 * np.cos(np.arange(Tp)[:, None] / 5.0 + np.arange(Kp))
    wfp = s.walk_forward(s.BacktestPortfolioTarget(fee=1e-4), Xp, prices, train=80, test=40, carry=ratep)
    walk_forward_portfolio = {
        "T": Tp, "K": Kp, "C": Cp, "prices": _flat(prices), "targets": _flat(Xp), "carry": _flat(ratep),
        "train": 80, "test": 40, "train_scores": _flat(wfp.train_scores), "choice": wfp.choice.tolist(),
        "target": _flat(wfp.target), "output": _flat(wfp.output),
        "stats": {k: _num(v) for k, v in wfp.stats.items()}}
```

`js/test/research.test.ts` (import `BacktestPortfolioTarget`):

```typescript
test("research: walkForward with a portfolio engine matches Python", () => {
  const h = oracle.helpers.walk_forward_portfolio;
  const prices = { data: Float64Array.from(h.prices, num), shape: [h.T, h.K] };
  const targets = { data: Float64Array.from(h.targets, num), shape: [h.T, h.K, h.C] };
  const carry = { data: Float64Array.from(h.carry, num), shape: [h.T, h.K] };
  const res = walkForward(BacktestPortfolioTarget({ fee: 1e-4 }), targets, [prices],
                          { train: h.train, test: h.test, carry });
  close(res.trainScores.data, h.train_scores, "trainScores");
  assert.deepEqual(Array.from(res.choice), h.choice);
  close((res.target as any).data, h.target, "target");
  close(res.output.data, h.output, "output");
  for (const [k, v] of Object.entries(h.stats as Record<string, Num>)) close([(res.stats as any)[k]], [v], `stats ${k}`);
});
```

- [ ] **Step 7: Regenerate and run the tests**

Run: `make install-dev && .venv/bin/python devtools/build_help_registry.py && .venv/bin/python devtools/wasm/gen_js_docs.py && .venv/bin/python devtools/wasm/gen_research_oracle.py && .venv/bin/python -m pytest tests/test_walk_forward.py tests/test_validation.py -q 2>&1 | tail -5`, then `FRESH`.
Expected: all pass.

- [ ] **Step 8: Tidy, full suite, JavaScript** (`make tidy`, full suite, `JSTEST`; `js/test/walk_forward.test.ts` and `js/test/validation.test.ts` pass unchanged). Expected: clean, all pass.

- [ ] **Step 9: Commit**

```bash
git add include/screamer/validation/position_test.h bindings/bindings_fin.cpp screamer/validation.py \
  wasm/validation_bindings.cpp js/src/validation.ts docs/functions_fin/walk_forward.md docs/portfolio_backtesting.md \
  tests/test_walk_forward.py devtools/wasm/gen_research_oracle.py js/test/research.test.ts \
  screamer/data/help.json js/src/generated/docs_data.ts js/test/docs_oracle.json wasm/smoke/oracle_research.json
git status --short   # add any other path the regeneration changed, by name
git commit -m "feat(validation): walk_forward with portfolio engines, scored on the book

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PZMDN2ijvnSZ7ZnrJUxnd8"
```

---

### Task 15: Guides, CHANGELOG, and the final checks

**Files:**
- Modify: `docs/choosing_an_operator.md` (a regime section and rows for the other new operators), `docs/backtesting_best_practices.md` ("Costs and carry", "Selection", "Measuring results"), `docs/recipes/walk-forward.md` (a carry section), `CHANGELOG.md` (`Unreleased`), `screamer/data/agent_guide.md` (only if it lists operator families by name; `git grep -n "BayesianRegression\|RollingHurst" screamer/data/agent_guide.md` shows whether it does)
- Regenerate: every file of `REGEN`

**Interfaces:**
- Consumes: Tasks 1 to 14, names and behaviour exactly as built (read the code before describing it).

- [ ] **Step 1: `docs/choosing_an_operator.md`**

After "## Detect cycles and regimes" (and rename that heading to "## Detect cycles"), add:

```markdown
## Detect regimes and change points

| Intent | Operator(s) | Note |
|:-------|:------------|:-----|
| Probability of a volatile, or of a trending, regime | `RollingHMM` | Two-state hidden Markov model refit on a trailing window; `prob` is the filtered probability of the high-variance state, or with `emission="ar1"` of the more persistent state. |
| Detect that the mean or spread of a series changed | `BayesianChangepoint` | Expected run length since the last change, and a predictive mean and standard deviation; feed a standardized series. |
| Sample events at cumulative moves | `CusumFilter` | +1 or -1 when the move since the last event passes `threshold`; label or trade at the events. |
| Test whether returns trend or revert | `RollingVarianceRatio`, `RollingHurst` | A variance ratio above 1 trends, below 1 reverts; `output="zscore"` gives the robust test statistic. |
| Measure how random a series is | `RollingPermutationEntropy` | 1 for noise, lower when some orderings of consecutive values are more common. |
| Share of the variance from jumps | `RollingJumpShare`, `RollingBipowerVar` | Bipower variation estimates the variance without the jumps. |
```

Add rows to the existing tables: "Measure dispersion or volatility": `| Typical value or spread of each hour of the day | RollingSeasonalMean, RollingSeasonalStd, RollingSeasonalZscore | The same slot of the last window_size cycles; the input must be on a regular clock. |` and `| Variance without the jumps | RollingBipowerVar | Bipower variation of returns, on the scale of the realized variance. |`; "Turn a signal into a position": `| Hold a signal, then ignore new ones for a while | Hold(cooldown=) | After a hold expires, triggers in the next cooldown steps are ignored. |`; "Measure risk and performance": `| Spread of the gains above a threshold | RollingUpsideDeviation | The mirror of RollingDownsideDeviation. |`; "Backtest a strategy": `| A list of the trades, or results per month | backtest_trades, backtest_report(by=) | One row per round trip; per-group totals, Sharpe, and drawdown. |`; "Is it more than luck?": `| Select between variants with funding, or between portfolio weightings | walk_forward(carry=), walk_forward with a portfolio engine | Every run is scored after BacktestCarry; a (T, K, C) target is scored on the book. |`. Use backticks around every name, as the existing rows do.

- [ ] **Step 2: `docs/backtesting_best_practices.md`**

In "Costs and carry", after the `BacktestCarry` sentence:

```markdown
[`walk_forward`](functions_fin/walk_forward.md) takes the same rate as
`carry=`, so every candidate is chosen and scored after its carry.
```

In "Selection", after the `deflated_sharpe` bullet:

```markdown
- [`walk_forward`](functions_fin/walk_forward.md) chooses a variant on each
  training window and trades it on the next, so the reported result never
  uses the data the choice was made on. With a portfolio engine the
  candidates are whole books, `(T, K, C)`.
```

In "Measuring results", after the first paragraph:

```markdown
[`backtest_trades`](functions_fin/backtest_trades.md) lists the round trips
behind those statistics, one row per trade with its entry, exit, side,
length, pnl, and cost. `backtest_report(..., by=labels)` adds the totals,
Sharpe ratio, and drawdown of each group of rows, for example per month with
`by=dates.astype("datetime64[M]")`. A result that comes from one month, or
from a few trades, shows there before it shows in the summary.
```

- [ ] **Step 3: `docs/recipes/walk-forward.md`**

Add at the end:

````markdown
## With funding

A perpetual pays funding on the position held. `carry=` takes the per-step
rate and scores every candidate after [`BacktestCarry`](../functions_fin/BacktestCarry.md),
marked with the close. A flat rate of 0.03% per day, about 11% a year, charges
the longs and pays the shorts.

```{code-cell} ipython3
funding = np.full(len(c), 3e-4)
wf_carry = walk_forward(engine, X, o, h, l, c, train=365, test=91, metric="sharpe",
                        periods_per_year=365, carry=funding)
print("span chosen per fold:", [spans[k] for k in wf_carry.choice])
print(f"out-of-sample Sharpe {wf_carry.stats['sharpe']:.2f}, "
      f"pnl {wf_carry.stats['total_pnl']:,.0f} USD after funding")
```
````

- [ ] **Step 4: `CHANGELOG.md`**, under `Unreleased`, in `### Added` (after the existing bullets):

```markdown
* Regime and change-point operators, in Python and JavaScript: `RollingHMM`
  (a two-state hidden Markov model refit on a trailing window, Gaussian or
  Markov-switching AR(1) emissions; `prob`, `pred_mean`, `pred_std`),
  `BayesianChangepoint` (Adams-MacKay online changepoint detection;
  `run_length`, `pred_mean`, `pred_std`), and `CusumFilter` (the symmetric
  CUSUM event filter). A "Regimes & change points" topic lists them.
* `RollingVarianceRatio` (the Lo-MacKinlay variance ratio, or its robust
  z-statistic with `output="zscore"`), `RollingPermutationEntropy`,
  `RollingBipowerVar`, `RollingJumpShare`, `RollingUpsideDeviation`, and
  `RollingSeasonalMean`, `RollingSeasonalStd`, `RollingSeasonalZscore` (the
  same slot of the last `window_size` cycles).
* `Hold(cooldown=)` and `Hold({ cooldown })`: after a hold expires, triggers
  in the next `cooldown` steps are ignored. The default 0 keeps the previous
  behaviour.
* `backtest_trades(values, index=None)` and `backtestTrades(values, { index })`:
  one row per round trip of an engine's output, counted as `BacktestStats`
  counts trades.
* `backtest_report(..., by=)` and `backtestReport(values, { by })`: totals,
  turnover, Sharpe ratio, drawdown, and steps per group of rows, as
  `summary["by_group"]` in Python and `byGroup` in JavaScript.
* `walk_forward(..., carry=, carry_multiplier=)` and
  `walkForward(..., { carry, carryMultiplier })`: every run is scored after
  `BacktestCarry`, marked with the engine's last price input.
* `walk_forward` and `walkForward` accept the portfolio engines
  (`BacktestPortfolioTarget`, `BacktestPortfolioOHLCTarget`) with `(T, K, C)`
  targets, scored on the book.
```

and in `### Changed`:

```markdown
* `permute_bars` and `permute_returns` with `mode="log"` name the first value
  that is not positive, its row (and asset), and how many there are, in the
  error message.
```

- [ ] **Step 5: Regenerate everything and run every check**

Run, in order:

```bash
make install-dev
.venv/bin/python devtools/inject_nan_footnote.py
.venv/bin/python devtools/build_help_registry.py
.venv/bin/python devtools/build_topic_pages.py
.venv/bin/python devtools/build_function_index.py
.venv/bin/python devtools/build_gallery.py
.venv/bin/python devtools/wasm/gen_wasm_manifest.py
.venv/bin/python devtools/wasm/gen_wasm_bindings.py
.venv/bin/python devtools/wasm/gen_ts_api.py
.venv/bin/python devtools/wasm/gen_js_docs.py
.venv/bin/python devtools/wasm/gen_null_oracle.py
.venv/bin/python devtools/wasm/gen_field_axis_oracle.py
.venv/bin/python devtools/wasm/gen_research_oracle.py
.venv/bin/python -m pytest -q -n auto 2>&1 | tail -5
make tidy
bash devtools/check_pure_headers.sh
.venv/bin/python devtools/compat_probe.py --json build/compat_after.json
.venv/bin/python devtools/compat_probe.py --diff build/compat_before.json build/compat_after.json
.venv/bin/python -m pytest tests/test_wasm_codegen_fresh.py tests/test_help_json_fresh.py tests/test_js_docs_fresh.py tests/test_ts_api_fresh.py tests/test_null_oracle_fresh.py tests/test_research_oracle_fresh.py -q 2>&1 | tail -3
source /Users/thijs/Projects/emsdk/emsdk_env.sh && cd js && npm run build:wasm && npm run build && npm test 2>&1 | tail -15; cd ..
make docs 2>&1 | grep -i -E "warning|error" | head -30
```

Expected: the full suite passes with no skips; tidy clean; the pure-header check passes; the compat diff prints no changed rows (no existing operator or helper changed its numbers or errors, the permute message aside if the probe records it: then the only `ERR CHANGED` rows are the two `mode="log"` permute errors); the freshness tests pass; `npm test` passes; the docs build shows no new warnings (fix each one, then rebuild). Also `git grep -n -P "\x{2014}" -- docs CHANGELOG.md screamer js/src` prints nothing new (no em dashes).

- [ ] **Step 6: Commit**

```bash
git add docs/choosing_an_operator.md docs/backtesting_best_practices.md docs/recipes/walk-forward.md CHANGELOG.md \
  screamer/data/help.json
git status --short   # add any other path the regeneration changed, by name (agent_guide.md if edited)
git commit -m "docs: regime operators, trade lists, per-group reports, and walk-forward carry in the guides

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PZMDN2ijvnSZ7ZnrJUxnd8"
```

The release (`make minor`) is the user's step, after their approval; do not run it.
