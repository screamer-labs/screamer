# Shape Contract, Phase 1 Addendum: Compatibility and Streaming

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Close the decisions the user made after the final review, so the library has one small set of behavior rules. Nothing that worked in 2.8.0 changes without an explicit decision, and streaming (lazy and async) is a tested contract for every operator in Python and JavaScript.

**Spec:** `docs/superpowers/specs/2026-10-11-shape-contract-design.md` (unchanged), plus the user decisions below.

**Base plan:** `docs/superpowers/plans/2026-10-11-shape-contract-phase1.md` (Tasks 1 to 10 done on `feat/shape-contract`).

## User decisions (binding)

- **D1. Tuples.** A tuple of arrays passed to a one-input operator raises `TypeError`. The message must say that a `(values, index)` stream goes to stream operators or a `Pipeline`, and that a compute operator takes the values while the index stays as it was. A list is a sequence of events. A tuple of numbers stays a sequence of scalar events (2.8.0 behavior). A tuple of arrays passed as the one argument of an N-input operator stays the N inputs (2.8.0 behavior).
- **D2. `from_pandas`.** Without `layout`, the result is exactly what 2.8.0 returned for every frame, including complete two-level columns (flat `(T, A*F)`). `from_pandas(df, layout="TAF")` gives `(T, A, F)` for a complete two-level product.
- **D3. Microstructure keywords.** `SignedVolume`, `RollingKyleLambda`, `EwKyleLambda`, `QueueImbalance` and `RollingOrderImbalance` accept their 2.8.0 keyword arguments again, as well as the new forms (separate arrays, one packed array, iterators, asset events).
- **D4. Lazy one-in/one-out operators.** Keep the fix: a generator into the nine `FunctorBase<_, 1, 1>` operators returns a lazy iterator. The changelog lists it under "Changed".
- **D5. Streaming is a contract.** For every operator, in both languages:
  - A synchronous iterator returns an iterator that reads nothing before the first `next()`.
  - An async generator returns an async iterator.
  - Both give exactly the batch result.
  - This holds for several inputs and for asset events.

  Python gains async generators for every operator. JavaScript gains sync and async iterators for multi-input operators.
- **D6. Rule 3 stays.** `(T, K, 1)` gives `(T, K)`. The next release is 3.0.0, but never edit version files: the user runs `make major` later.
- **D7. Six rules, stated once.** The rules are written on `docs/polymorphic_api.md` and enforced by contract tests:
  1. Shape `(T, *A, F?)`.
  2. The container never changes the numbers.
  3. One meaning per container.
  4. Inputs: separate arguments, or one packed array, with assets broadcast.
  5. Ambiguity raises.
  6. A change to the result of code that worked before needs an explicit decision and a "Changed" entry.

## Global Constraints

Same as the base plan: `.superpowers/sdd/2026-10-11-shape-contract-phase1/constraints.md`. All operator logic stays in C++. Never edit version files. Run `make install-dev` after C++ changes. Python tests run with `.venv/bin/python -m pytest`. JavaScript tests run with `cd js && npm run build:wasm && npm test`. No em-dashes. Commit on `feat/shape-contract`.

## Compatibility gate (applies to every task)

`devtools/compat_probe.py` (Task 11 creates it) runs a matrix of input forms on the installed build and on a released wheel, then prints every row whose result changed. After each task, run it against 2.8.0. The only "value changed" rows allowed are the decided ones:
- `(T, K, 1)` shapes (D6);
- the nine lazy operators (D4);
- the last-bit `EwMean`, `Return` and `ROCP` fixes.

---

### Task 11: Compatibility decisions D1 to D3, and the probe

**Files:** `include/screamer/common/dispatch.h`, `src/screamer/common/dispatch.cpp` (the one-input `is_event_list` routing), `screamer/streams.py` (`from_pandas`), `screamer/shape.py`, `screamer/microstructure.py`, `devtools/compat_probe.py` (new), and tests in `tests/test_compat.py` (new).

- [ ] **D1.** On the one-input paths, only a `list` whose first item is an array is an event list (pass `tuples=false`). A tuple whose first element is an ndarray raises `TypeError` with the D1 message. A tuple of numbers and the N-input tuple form are unchanged. Tests:
  - `RollingMean(2)((v, idx))` and `BollingerBands(3)((v, idx))` raise `TypeError`, and the message matches `"(values, index)"`;
  - `RollingMean(3)((1.0, 2.0, 3.0))` returns a list;
  - `Add()((v, w))` equals `v + w`;
  - `RollingMean(2)([np.ones(3), np.ones(3)])` returns a list of two `(3,)` arrays.
- [ ] **D2.** With `layout=None`, `from_pandas` returns `np.asarray(obj.to_numpy()), np.asarray(obj.index)` for every frame. With a `layout`, it reads through `frame_axes` (a complete two-level product is three axes) and reshapes. Tests:
  - flat, int, string, datetime, ragged and complete two-level frames without a layout each equal the 2.8.0 result, written as literals in the test;
  - `from_pandas(df2level, layout="TAF").shape == (T, A, F)`.

  Update the docs pages that claim `(T, A, F)` by default (`docs/functions_streams/from_pandas.md`, `docs/functions_streams/reshape.md`, `docs/polymorphic_api.md`, CHANGELOG).
- [ ] **D3.** Each microstructure class's `__call__` accepts its 2.8.0 keyword names (read them from `git show 33baa22:screamer/microstructure.py`) as well as the positional and packed forms. Mixing keywords and positionals for the same field raises `TypeError`. Tests check each class with keywords, positional, packed and iterator input, and expect identical results.
- [ ] **Probe.** Create `devtools/compat_probe.py`. It runs the matrix from `/private/tmp/claude-501/-Users-thijs-screamer/0e59f489-97b5-4001-ae0f-0e9e21f0976c/scratchpad/probe.py` (copy it and extend the cases to cover D1 to D3). Usage:
  - `python devtools/compat_probe.py --json out.json` writes this build's results;
  - `python devtools/compat_probe.py --diff old.json new.json` prints the rows that changed, tagged `same`, `ERR->VALUE`, `VALUE->ERR` or `VALUE CHANGED`.

  The module docstring says how to get a released wheel's JSON: create a venv, `pip install screamer==X`, and run the probe with `-I`. Record the diff against 2.8.0 in the task report. Only the decided rows may change.

### Task 12: Async generators for every Python operator, and the laziness contract

**Files:** `include/screamer/common/lazy_eval_iterator.h`, `include/screamer/common/async_generator.h`, `src/screamer/common/async_generator.cpp`, `screamer/_async.py`, `include/screamer/common/dispatch.h` (async routing on every `functor_*` path), `bindings/bindings_core.cpp`, and tests in `tests/test_shape_contract.py`.

- [ ] **Refactor.** Separate the per-event work from the pulling. Move what `LazyEvalIterator` does with one event into a C++ `EventStepper` class (scalar mode, asset mode, per-asset `_fresh()` copies, shape checks, packed or tuple unpacking). The stepper takes the event's items and returns the output.
  - `LazyEvalIterator` pulls synchronous items and calls the stepper.
  - `LazyAsyncIterator` awaits one item from each async source in a coroutine in `screamer/_async.py`, then calls the same stepper.

  The semantics are identical by construction; there is no second copy of the logic.
- [ ] **Routing.** Every `functor_*` path, as well as `screamer_call`, routes async sources to the new `LazyAsyncIterator`:
  - one async source;
  - N async sources;
  - one async source of tuples or packed events.

  Mixing sync and async sources raises `TypeError`.
- [ ] **Contract tests** in `tests/test_shape_contract.py`, parametrized over every registry operator:
  - (a) `test_iterator_is_lazy`: a generator that counts its pulls is consumed 0 times before the first `next()` and once per step afterwards.
  - (b) `test_async_equals_batch`: N async generators give exactly the batch result.
  - (c) `test_async_asset_events_equal_batch`: async `(K,)` events, and N async sources of `(K,)` events, equal the `(T, K)` batch.
  - (d) Async with a packed event source equals batch for multi-input operators.

  Use exact equality, with NaN treated as equal.
- [ ] **Docs.** Update the `docs/polymorphic_api.md` tables so that async is listed for every arity.

### Task 13: JavaScript iterators for multi-input operators, and JS streaming contract tests

**Files:** `js/src/runtime.ts`, `js/test/shape_contract.test.ts` (or a new `js/test/streaming_contract.test.ts`), `docs/javascript.md`.

- [ ] **Multi-input iterators.** `wrapOp` with `nIn > 1` accepts:
  - N sync iterators, which return a generator;
  - N async iterables, which return an async generator;
  - one iterator or async iterable of N-tuples or packed rows.

  Events may be numbers, or `(K,)` asset rows that broadcast like Python, with one op copy per asset from the factory. Python's behavior is the reference. Reuse the existing single-input asset-event code; do not duplicate it.
- [ ] **JS contract tests over every generated operator** with a default-constructible factory, mirroring Task 12:
  - laziness (nothing pulled before the first `next()`);
  - sync iterator equals batch;
  - async equals batch, for scalar events and `(K,)` asset events, in single-input and multi-input forms.
- [ ] **Docs.** Remove the closed gaps from the parity list in `docs/javascript.md`.

### Task 14: The six rules and the changelog

**Files:** `docs/polymorphic_api.md`, `docs/javascript.md`, `CHANGELOG.md`.

- [ ] State the six rules from D7 once, near the top of `docs/polymorphic_api.md`, each with a one-line example. Link the existing detailed sections to them instead of repeating them. Follow the writing style guide (`/Users/thijs/.claude/projects/-Users-thijs-screamer/memory/feedback_writing_style.md`), with no em-dashes. Add the short JS form of the same rules to `docs/javascript.md`.
- [ ] CHANGELOG, under Unreleased:
  - "Changed" holds exactly the decided changes of results: rule 3 shapes, the nine lazy operators, and the last-bit `EwMean`, `Return` and `ROCP` fixes. Each entry gives the one-line way to get the old result.
  - Add a note at the top of Unreleased that it is a major release (3.0.0) because of these changes.
  - "Added" lists async for every operator and JS multi-input iterators.
  - Remove the entries made obsolete by D1 to D3: the `from_pandas` default and the microstructure keyword break.
- [ ] Run `make docs` (clean) and both full test suites.
