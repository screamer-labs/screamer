# Contributing to screamer

Thanks for your interest in improving screamer. This guide covers the development
setup, the build/test loop, and the conventions that keep the library fast, correct,
and well documented. By participating you agree to the
[Code of Conduct](CODE_OF_CONDUCT.md).

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
Python will import a stale binding.

Run the tests directly with `poetry run pytest -q`. The suite has 3600+ tests and
should be green with zero skips before you open a PR.

## Core rules

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
