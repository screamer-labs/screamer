# Filter steady-state start, portfolio backtest_report, time_splits

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Three library improvements found while writing the Guide: IIR filters start from a steady state instead of zero, `backtest_report` accepts portfolio engine output, and a new `time_splits` gives train/test index pairs that purge overlapping labels.

**Architecture:** Each item lands in C++ where it computes (screamer is a C++-core library: Python and JavaScript are thin bindings), with Python and JavaScript bindings kept in parity, a function page (frontmatter feeds `screamer/data/help.json` and the JS docs), tests, and the Guide pages that worked around the gap updated.

**Tech Stack:** C++17 core, nanobind Python bindings, Embind/WASM JavaScript binding (`js/`), pytest, node test runner, Sphinx docs.

**Spec:** the designs approved in chat on 2026-10-27/28, recorded here as the binding requirements (no separate spec file):

1. **Filter start.** All filters on the shared `IIRFilter` core (`include/screamer/signal/signal.h`): `Butter`, `ButterHighpass`, `ButterBandpass`, `ButterBandstop`, `SuperSmoother`, `Decycler`, `RoofingFilter`. New setting `start_policy` with two values: `"steady"` (new default): on the first finite input `x0`, set the internal state to the steady state for a constant input `x0` (scipy `lfilter_zi(b, a) * x0` for the transposed direct form II state), so a low-pass output starts at `x0` and a high-pass or band output starts at 0; `"zero"`: today's zero state, which matches `scipy.signal.lfilter` exactly. `reset()` returns to the not-yet-started state. NaN inputs keep their current policy (`ignore`); a leading NaN does not start the filter.
2. **Portfolio report.** `backtest_report(values, ...)` also accepts the `(T, K, 4)` output of the portfolio engines. `summary` then describes the book: equity, pnl and cost summed over assets; Sharpe, drawdown and turnover from the summed book; trade counts and exposure counted over all assets. `summary["per_asset"]` is a list of K per-asset summaries with the usual keys. `running` holds the book series. `by=` grouping works on the book. Computation in C++.
3. **time_splits.** `time_splits(n, n_splits=5, *, test_size=None, label_length=0, max_train_size=None)` returns a list of `(train_indices, test_indices)` integer-array pairs, usable as scikit-learn `cv=`. Folds follow scikit-learn `TimeSeriesSplit` (expanding train window before each test block; `max_train_size` caps it; `test_size` defaults to `n // (n_splits + 1)`). Training rows only come before the test block. `label_length` is an int (same for every row) or a length-`n` array (e.g. `TripleBarrier`'s `holding` output): a training row `t` is dropped when `t + label_length[t] >= first test row` (its label reaches into the test block). A NaN or negative label length raises ValueError naming the function (ambiguity raises). No purged K-fold, no embargo (rejected by the user).

## Global Constraints

- All computation in C++; Python and JS are thin bindings. Check for an existing node or helper before adding one.
- Python and JS parity is a hard default: every new parameter, value and function exists in both, with the same names (camelCase in JS where the binding already does so) and the same results.
- Batch and streaming give identical results; all functions causal.
- Compatibility rules: `"steady"` changes default results of seven filters for their first outputs; record it in CHANGELOG under a new top `## Unreleased` section (`### Changed`), naming the filters and `start_policy="zero"` for the old behavior. Never edit version numbers (only `make patch/minor/major` with approval). Ambiguous input raises with a message naming the function.
- After any C++ change run `make install-dev` (not just `make build`), or Python imports a stale binding.
- After changing any function page frontmatter, regenerate `screamer/data/help.json` (`devtools/build_help_registry.py`) and the JS docs (`devtools/wasm/gen_js_docs.py`); the freshness tests fail otherwise.
- Efficiency matters: no extra allocations or passes in the hot loops.
- Docs follow the user's style guide (/Users/thijs/.claude/projects/-Users-thijs-screamer/memory/feedback_writing_style.md), no em-dashes, and the Guide conventions (memory project_docs_guide.md).
- Commits with explicit paths; never `git add -A` or `git stash`.

## Review Focus

1. A series that starts with NaN: the filter must start at the first finite value, in batch and streaming alike.
2. `reset()` followed by new data: the filter starts again from the new first value.
3. A portfolio with an asset that is NaN for its first rows (listed later): book sums must skip NaN per asset, not turn the whole book NaN.
4. `time_splits` with `label_length` longer than the training block: an empty training set must raise, not return an empty fold silently.
5. JS `time_splits` and portfolio `backtestReport` results equal Python's on the same input (parity test).

---

### Task 1: Steady-state filter start

**Files:** `include/screamer/signal/signal.h` (IIRFilter), the seven filter headers, their Python bindings, the JS binding/codegen inputs, the seven function pages under `docs/functions_signal/` (or wherever they live), C++/Python/JS tests, CHANGELOG.md, and the Guide pages that describe the start-up swing (`docs/guide/denoise/low-lag.md`, `docs/guide/denoise/lowpass.md`, `docs/guide/cycles/dominant-cycle.md`; grep docs/guide for "start-up", "starts from zero", "swing").

- [ ] Tests first (Python): for each of the seven filters, `start_policy="zero"` equals `scipy.signal.lfilter(b, a, x)` (Butter family: `scipy.signal.butter`); `"steady"` equals `lfilter(b, a, x, zi=lfilter_zi(b, a) * x[first finite])` on the finite part; a constant series gives a constant output (low-pass) or zeros (high-pass/band) from the first sample; leading NaNs; batch equals value-by-value; `reset()` restarts. Default is `"steady"`. An unknown value raises ValueError naming the filter.
- [ ] Implement in the IIR core (one place), expose `start_policy` on the seven filters in C++, Python and JS.
- [ ] Function pages: document `start_policy` (frontmatter parameter and one sentence in Description); regenerate help.json and JS docs.
- [ ] Guide pages: remove or rewrite text and plot windows that worked around the start-up swing.
- [ ] CHANGELOG `## Unreleased` / `### Changed` entry.
- [ ] Verify: `make install-dev`; the new tests; `tests/` subsets for signal and docs freshness; JS `npm test` in `js/`; Sphinx build (only the known warning).

Commit: `feat(signal): IIR filters start from a steady state (start_policy)`.

### Task 2: backtest_report on portfolio output

**Files:** the C++ report code behind `_backtest_report` (find via `bindings/bindings_fin.cpp`), `screamer/backtest.py`, `js/src/backtest.ts`, `docs/functions_fin/backtest_report.md`, tests (Python, C++ if the report has C++ tests, JS), CHANGELOG (`### Added`), Guide pages `docs/guide/strategies/pairs.md`, `cross-sectional-momentum.md`, `carry.md`, `docs/guide/portfolios/multi-asset.md` (replace the `PortfolioReport` x sqrt(365) workaround with `backtest_report(..., periods_per_year=365)` and keep the printed numbers consistent with the prose).

- [ ] Tests first: a `(T, K, 4)` input gives a book summary equal to `backtest_report` on the hand-summed book for equity, pnl, cost, Sharpe, drawdown; `per_asset[k]` equals `backtest_report(values[:, k, :])`; NaN-leading asset; K=1 equals the `(T, 4)` result; `by=` on the book; wrong rank raises naming the function; JS result equals Python's.
- [ ] Implement in C++; Python and JS wrappers label the result.
- [ ] Function page, help.json, JS docs regenerated.
- [ ] Guide pages updated; their numbers re-checked against the build.

Commit: `feat(backtest): backtest_report reports portfolio books`.

### Task 3: time_splits

**Files:** C++ for the split computation (place next to `walk_forward`'s C++ helpers if any, else a small new header under `include/screamer/`), `screamer/validation.py` (export from `screamer/__init__` as the generator does for other validation functions), `js/src/validation.ts`, a new function page `docs/functions_fin/time_splits.md` (topics: significance or the validation topic `walk_forward` uses), help.json and JS docs regenerated, tests (Python incl. a scikit-learn `TimeSeriesSplit` equivalence check when `label_length=0` and sklearn is installed, purge cases with int and array `label_length`, empty-train raises, NaN label length raises, `n_splits` larger than possible raises; JS parity), CHANGELOG (`### Added`), and `docs/guide/machine-learning/training-set.md` (use `time_splits(len(X), label_length=holding)` with the triple-barrier `holding` output instead of the hand-made gap; keep the page shape).

Commit: `feat(validation): time_splits with label purging`.

### Task 4: Final pass

- [ ] `make install-dev`; full `tests/`; `npm test` in `js/`; Sphinx build with `-W --keep-going` (only the known `sphinx_exec_code` warning); `tests/test_guide.py`.
- [ ] CHANGELOG `## Unreleased` reads as one coherent section (Changed: filter start; Added: portfolio report, time_splits).
- [ ] Screenshot the three changed Guide areas (denoise/lowpass, a portfolio page, training-set) and one filter function page.

Commit (if anything changed): `chore: final pass for filter start, portfolio report, time_splits`.
