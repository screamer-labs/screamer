# Contributing to screamer

Thanks for your interest in improving screamer. This guide covers the development
setup, the build/test loop, and the conventions that keep the library fast, correct,
and well documented. By participating you agree to the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Design principles

The library serves four goals at once. Every design decision serves them, and the
rules below fall out of them. Read this before adding or changing an operator.

1. **Correct, clean, well-tested code.** One clear implementation per behavior. No
   parallel or duplicate implementations of the same operator.
2. **State-of-the-art speed and memory, in C++.** The compute is C++. No Python loops
   over the data, and no needless data crossing the Python/C++ boundary.
3. **A friendly, uniform API.** Polymorphic argument handling, and batch and streaming
   modes that produce identical results from the same engine.
4. **C++ is a first-class consumer, and more bindings are coming (JS, R).** A user of
   the C++ core, or a future binding, must get the whole library. So all logic must
   live in the C++ core, not in any one binding.

From these, the hard rules:

- **All operator logic lives in the C++ core.** A new operator is a C++ node plus a
  thin binding (see [Adding a function](#adding-a-function)). Never implement an
  operator's computation, or a numerical primitive, in Python.
- **The Python layer is bindings and structure only.** The only logic that may
  originate in Python is (a) a name or argument **synonym** of exactly one C++
  operator, or (b) a **special instance** that binds the parameters of exactly one C++
  operator. Combining two or more operators, adding arithmetic, or holding new state
  must be a C++ node. If you are chaining ops or writing a data loop in Python, stop:
  it belongs in C++.
- **Gluing the outputs of C++ ops in Python is still a Python operator.** Stacking
  columns, concatenating, masking or selecting rows, or doing arithmetic on the
  results of one or more C++ operators is operator logic and belongs in the C++
  node. "The reducers are already in C++" does not make a Python-orchestrated
  combination compliant: the combination *is* the operator. The one sanctioned way
  to combine operators is a `Pipeline`/DAG, which is graph *structure* (built once)
  and compiles to C++ - the data never crosses into Python per event.
- **One implementation per operator.** Never ship a second implementation of the same
  behavior (for example a C++ eager path plus a separate Python lazy path). Batch and
  stream are the same compiled engine; a divergent second path is a maintenance and a
  correctness-parity defect.
- **Every operator works in every regime.** An operator must run in all three call
  regimes - eager (arrays), graph (`Node`/`Pipeline`), and lazy (event iterator) -
  and produce identical results across them (goal 3). No operator may be eager-only,
  batch-only, or otherwise regime-restricted: an operator that raises in the graph or
  lazy path is an *incomplete* operator, not a finished one. This holds for
  training-time and dataset-assembly helpers too (someone will feed them a live
  production stream). If a capability is awkward to express in the streaming engine,
  design the C++ node (or a `Pipeline` of C++ nodes) correctly - do not ship an
  eager-only shortcut.
- **No Python orchestration of the data path.** The Pipeline/DAG compiles to and runs
  in C++ so data does not cross the boundary per event. Do not orchestrate windowing,
  joins, aggregation, or event scheduling in Python.
- **Applications reuse the core, never re-implement it.** Downstream projects must call
  screamer operators, not hand-roll a rolling mean, regression, lag, or forward-fill in
  numpy or pandas. A missing capability is a request for a new core operator, not a
  local reimplementation.

## Development setup

You need a C++17 compiler, CMake, and Python 3.11 or newer.

```bash
git clone https://github.com/screamer-labs/screamer.git
cd screamer

# Poetry is the reference workflow (the Makefile auto-detects it):
poetry install

# Or with plain pip:
pip install -e ".[dev]"
```

The C++ extension is built with CMake via scikit-build-core and pybind11.

## Build and test loop

The `Makefile` wraps the common tasks (run `make help` for the full list):

| Command | What it does |
|---|---|
| `make build` | Compile the C++ extension, copy the `.so` into `screamer/`, regenerate `screamer/__init__.py` |
| `make test` | Build, install editable, run the full pytest suite |
| `make tidy` | Run clang-tidy (catches uninitialised class members and similar) |
| `make docs` | Build the Sphinx HTML docs into `docs/_build/html/` |
| `make benchmark` | Run the benchmark suite and generate plots |

After any C++ change, run `make build` (or `make install-dev`) before testing, or
Python will import a stale binding. Before pushing a C++ change, also run
`make tidy` (the clang-tidy `cppcoreguidelines-pro-type-member-init` check that CI
enforces as an error): it catches uninitialized class members, the bug class that
once broke `RollingZscore` on one platform. A new operator is not done until
`make tidy` is clean.

Run the tests directly with `poetry run pytest -q`. The suite has 3600+ tests and
should be green with zero skips before you open a PR.

## Core rules

- **Logic lives in C++.** See [Design principles](#design-principles): operator
  computation and numerical primitives belong in the C++ core, the Python layer is thin
  bindings plus single-operator synonyms/instances, and there is one implementation per
  behavior. A reviewer rejects Python that chains operators, loops over data, or
  duplicates a C++ path.
- **Causality is non-negotiable.** Every function must depend only on current and past
  inputs, never future ones (no backfilling, no look-ahead). Batch and streaming calls
  must produce identical results; this is enforced by the stream-vs-batch tests.
- **Match the surrounding code.** Follow the naming, structure, and comment style of the
  files you touch.
- **Keep it efficient.** Screamer competes on speed. Avoid dead allocations and redundant
  work in hot paths.

## Adding a function

1. **Implement** the C++ functor and its pybind11 binding under `bindings/`.
2. **Document it.** Add a page at `docs/functions_<family>/<Name>.md` with YAML
   frontmatter. The docs build validates every field by instantiating and calling the
   functor, and refuses to publish an undocumented or mis-tagged function.
3. **Assign topics.** Give the function one or more `topics:` slugs from
   `docs/topics.yml` (the single source of truth for the left-nav taxonomy).
   `tests/test_doc_coverage.py` fails if any public function has no page or no topic.
4. **Declare a `nan_policy`** in the frontmatter (see below).
5. **Optionally add a baseline.** For parity testing, add an independent reference
   implementation in `devtools/baselines/<Name>.py` as a class named `<Name>_<lib>`
   (for example `RollingMean_pandas`). `tests/test_baselines.py` compares screamer
   against every baseline it finds; `python -m devtools.report_baselines` shows coverage.
6. **Regenerate.** `make build` refreshes `screamer/__init__.py`. If you changed docs
   frontmatter, regenerate the help registry and topic pages:
   ```bash
   poetry run python devtools/build_help_registry.py
   poetry run python devtools/build_topic_pages.py
   ```
7. **Check the contract.** Run `pytest -q`. The registry-driven suites pick your
   operator up from its docs page and assert every rule in
   [The operator contract](#the-operator-contract) against it: causality, batch ==
   stream, all three regimes, `reset()`, the `nan_policy` and `start_policy` you
   declared, and the shape surface. You do not write those tests. Write the tests
   that prove your operator computes the right numbers.

### The operator contract

An operator must satisfy every rule below. None of them are checked by a test you
write yourself: each is asserted against every operator in the registry, and the
registry is built from your docs page. Add the page and your operator is enrolled;
there is no separate registration step, and no way to forget one.

Read this list as the spec you are writing against.

| Rule | What it means | Enforced by |
|---|---|---|
| **Causal** | Output at index `t` depends only on input up to `t`. Changing the future must not change the past. No backfill, no lookahead. | `test_contract_compliance.py::test_causal` |
| **Batch == stream** | An array call and a sample-at-a-time call produce identical output. | `test_stream_vs_batch.py` |
| **Every regime** | Eager (arrays), graph (`Pipeline`), and lazy (event iterator) produce identical output. | `test_contract_compliance.py::test_regimes_agree` |
| **`reset()` is complete** | A reset operator behaves exactly like a fresh one; no state survives. | `test_contract_compliance.py::test_reset_restores_initial_state` |
| **Declared `nan_policy` is the real one** | See below. Four properties, including that an ignored `NaN` costs exactly one output slot and never reaches internal state. | `test_nan_input_compliance.py` |
| **Declared `start_policy` is the real one** | Warmup behaves as `strict` / `expanding` / `zero` claim, including on `NaN`-bearing input. | `test_nan_start_policy_compliance.py` |
| **Shape surface** | Correct results through tensors, strided views, matrices, and every input/output size. | `test_tensor.py`, `test_view.py`, `test_matrix.py`, `test_io_size.py` |
| **Documented and tagged** | A docs page with frontmatter and at least one topic from `docs/topics.yml`. | `test_doc_coverage.py` |
| **Reachable by the harness** | Driven by `tests/param_cases.py`, or named in `PARITY_EXEMPT` with a reason. | `test_parity_registration.py` |

A 1-input/1-output operator whose constructor arguments all have defaults is adopted
by the parity harness automatically. Anything else needs a `test_definitions` entry
in `tests/param_cases.py`, and `test_parity_registration.py` fails until it has one
or an explicit exemption.

**Writing a new contract test.** If a rule applies to every operator, it belongs in
`tests/test_contract_compliance.py` (or the `nan` / `start_policy` compliance files)
and must enumerate the registry. A contract test with a hardcoded list of function
names is the smell: the next operator will not be covered by it. Per-operator
numerical behaviour (does `ATR` match TA-Lib) is the opposite case and belongs in
that operator's own file.

**When a property finds pre-existing failures.** Add the offenders to the
`KNOWN_*` set beside the property with `xfail(strict=True)` and a stated reason,
rather than weakening the property or expanding the change. `strict=True` means
fixing one fails the suite until the name is removed, so the list can only shrink.

### NaN policy

Every function with a docs page must declare a `nan_policy` in its frontmatter. Pick one:

- `ignore` - summary statistics, smoothers, filters. Skips `NaN` in internal state,
  emits `NaN` at the same index, recovers at the next finite sample. This is almost
  always the right answer.
- `propagate` - positional functions only (`Lag`, `Diff`, `Diff2`, `Momentum`, `ROC`,
  `ROCP`, `ROCR`, `LogReturn`, `Return`). `NaN` flows through the lookback and recovers
  once it clears.
- `nan-aware` - only for functions whose purpose is to consume `NaN` (`FillNa`, `Ffill`).

The full contract and rationale live in `docs/nan_and_warmup.md`; compliance is verified
by `tests/test_nan_input_compliance.py`.

## Writing documentation

This governs all prose in screamer: notebook cells, help pages, function docs,
docstrings, and code comments. Aim for documentation that is correct, clear, and serves
a purpose, in the register of the NumPy, PyTorch, and Boost references. Every sentence
teaches a detail the reader needs. If a sentence only adds energy, cut it.

### Content and purpose (what to write)

- **Lead with what the thing is.** State its identity, not a property, a benefit, or a
  side effect. `forecast_pairs` is a helper that aligns past features with a future
  target; it is not "a guard against lookahead bias" (that is a consequence, not the
  thing).
- **Ground every claim in the actual behavior.** If you cannot point to the code or the
  output that makes a sentence true, cut it. Do not write what sounds plausible; read
  the implementation and describe what it does.
- **One checkable point per paragraph or cell, and it maps to the code it describes.**
  The reader should be able to hold the sentence against the code and confirm it.
- **State the mechanism, not a vibe.** Give the sentence that explains how it works.
  "`y[t]` is the sum of the last `h` returns; paired with features from `h` steps
  earlier, it is the sum of the next `h` returns" is the mechanism. "A trailing sum is
  what makes it work" is not.
- **Write flat, not meta.** State the thing. Do not describe what the sentence is doing
  or loop back on it ("writing it this way is what lets the function use it, and that
  same value then becomes...").
- **Concrete over abstract.** "`y[t]` is the sum of the last `h` returns" beats "a
  trailing quantity."
- **Introduce a name only where it is used, and give its purpose when you introduce it.**
  Never leave the reader carrying an unexplained term and wondering why it matters.
- **Use the thing's own terms, and stay unit-agnostic where it is.** Do not borrow a
  domain unit for a general tool: a shift is a `count` of events or a `duration` of
  time, not "bars." Name a position `index`; name one increment of a specific example
  series `step`.
- **Cut what serves no purpose, and do not teach generic knowledge the library does not
  provide.** A train/test split is in scope only to the extent screamer provides it;
  purge/embargo methodology is generic machine learning, so it is cut, not explained.

### Tone of voice (how to write it)

- **No hooks or lead-with-a-strong-statement openers.** State what the thing is or what
  the section covers. Not "Predicting the future from the present is a classic trap."
- **No personification of code.** Code does not trick, earn its place, take over, want,
  or have things creep into it or drop into it. Say what it does.
- **No editorializing adjectives.** Cut `genuine`, `classic`, `real`, `clean`,
  `powerful`, `elegant`, `correctly`. Replace with a precise term or nothing.
- **No emphasis for effect.** No bold for drama, no capitals for shouting (not "FADE").
  Emphasis comes from sentence structure.
- **Report, do not narrate.** "The coefficient is negative" (a fact), not "the model
  learned to fade the signal" (a story).
- **No em dashes** (ASCII hyphens only), and none of the usual tics: no
  negation-affirmation for effect when the contrast carries no information, no "same X,
  same Y" anaphora, no "so/because/therefore" gluing two true facts with no causal link.

### Before you commit

- **Verify, do not spot-fix.** Fixing the one sentence someone flagged while leaving the
  same defect elsewhere is the most common failure. On each pass, hold the whole
  document to these rules: grep for the term you changed, re-read every cell, and
  re-execute the notebook or rebuild the docs.
- **When unsure whether a claim is true, read the code.** Plausible is not correct.

## Pull requests

1. Branch off `main`.
2. Make your change with tests. Keep the suite green (`poetry run pytest -q`).
3. Update docs and the `CHANGELOG.md` where relevant.
4. Open a PR describing the change and the reasoning. CI runs the tests and clang-tidy;
   docs build as a PR check.

## Releases (maintainers)

Version bumps are automated. Do **not** edit version strings by hand; they live in
several files plus a git tag and are kept in sync by `bump-my-version`:

```bash
make patch   # 0.6.2 -> 0.6.3
make minor   # 0.6.2 -> 0.7.0
make major   # 0.6.2 -> 1.0.0
```

Each target bumps the version, commits, tags, and pushes. The pushed tag triggers
`build-wheels.yml`, which runs the test suite, builds wheels for Python 3.11-3.14 on
Linux, macOS, and Windows, and publishes to PyPI via OIDC Trusted Publishing.
