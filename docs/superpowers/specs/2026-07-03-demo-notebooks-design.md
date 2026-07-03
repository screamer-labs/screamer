# Demo notebooks — capability showcase (Phase 1: local)

**Status:** design approved, ready for planning
**Date:** 2026-07-03
**Scope:** Phase 1 — author 10 small, self-contained `.ipynb` notebooks that
showcase screamer's capabilities, use cases, and variants, plus a one-command
local execute-and-verify workflow. RTD publication, versioning, PyPI, and DevOps
are Phase 2 (deferred, sketched here only for drop-in cleanliness).

## Motivation

screamer now spans a lot of ground — the polymorphic single-series API, rolling/
EW statistics, financial indicators, signal filters, NaN policies, the
multi-stream layer (`merge`/`combine_latest`/`pace`/`dropna`/`filter`/`split`),
and the computational DAG (define once, run batch or live). A focused set of
runnable notebooks is the fastest way for a user to see what the library does
and to serve as executable, always-current documentation. Testing them locally
first (build the lib + bindings, run every notebook, confirm green) de-risks the
later ReadTheDocs publication.

## Principles / constraints

- **Small and focused.** Each notebook covers one topic in a few sections. Prose
  (markdown cells) explains the *why*; code shows the *how*; a short "takeaways"
  closes each.
- **Self-contained & reproducible.** Every notebook generates its own data
  (synthetic, **seeded** RNG) — no network, no external files. It runs identically
  on the author's machine and on a future RTD build server. Outputs are
  deterministic so a diff or a re-execute is stable.
- **Public API only.** Notebooks use the public `screamer` surface (functors,
  `screamer.streams`/combinators, `Node`/`Input`/`Dag`) — never the underscore
  bindings — so they double as user-facing docs.
- **Executable as tests.** `poetry run pytest --nbmake docs/notebooks/` runs every
  notebook top-to-bottom against the installed library and fails loudly on any
  error. This is the local acceptance gate.
- **Plots** via matplotlib (existing dep); pandas / pandas-ta-classic cross-checks
  where illustrative (existing deps). Keep figures small and captioned.

## Non-goals (Phase 2, deferred)

- ReadTheDocs wiring: adding **myst-nb** to `docs/conf.py`, the notebooks to the
  `index.rst` toctree, and a `.readthedocs.yaml`. (myst-nb is the MyST-native
  successor to the `nbsphinx` that was previously configured and removed.)
- Version bump / release, PyPI, RTD hosting, CI for docs, DevOps.

The notebooks are authored so Phase 2 is a clean drop-in (self-contained,
deterministic, public-API-only, already under `docs/`).

## The 10 notebooks

Each lives at `docs/notebooks/NN-slug.ipynb` (numbered for order). Structure:
title + one-paragraph intro → 2–4 short sections → "Takeaways" cell.

1. **`01-quickstart-polymorphic-api.ipynb`** — the core idea: one functor works on
   every input shape. `RollingMean(5)` applied to a scalar, a 1-D array, a 2-D
   `(T,K)` array, a Python list, and an iterator; show batch and streaming give
   identical numbers. Links the "one callable, all shapes, no train/serve skew"
   story.
2. **`02-rolling-and-ew-statistics.ipynb`** — the rolling/EW families on a price
   series: `RollingMean`/`RollingStd`/`RollingZscore`/`RollingMin`/`Max`,
   `EwMean`/`EwStd`. Show the `start_policy` variants (`strict`/`expanding`/`zero`)
   and a pandas `.rolling()` cross-check for equality.
3. **`03-streaming-live-events.ipynb`** — functors as generators/iterators: feed a
   simulated live source, consume results one at a time; the backtest↔live "same
   code" property (array result == streamed result).
4. **`04-financial-indicators.ipynb`** — OHLC indicators on synthetic OHLC bars:
   `ATR`, `RollingRSI`, `MACD`, `BollingerBands`, `Stoch`. Show the multi-input
   `(T,N)` array form and plot indicators under a price panel.
5. **`05-nan-handling.ipynb`** — the three NaN policies (`ignore` / `propagate` /
   `nan-aware`) demonstrated on a series with injected gaps; `FillNa`, `Ffill`,
   and (stream) `dropna`. Show how each family responds and recovers.
6. **`06-signal-processing.ipynb`** — `Butter`/`ButterBandpass`/`Bessel`(if bound),
   smoothing (`HullMA`/`KAMA`), `Detrend`. Before/after plots on a noisy signal;
   note causality (no lookahead).
7. **`07-aligning-async-streams.ipynb`** — the multi-stream foundation: two async
   price feeds with different timestamps; `combine_latest` (spread updates on any
   tick, forward-fill carry), the order-key model, `merge`. Plot the aligned
   spread. Contrast `emit="when_all"` vs `"on_any"`.
8. **`08-replay-backtest-live.ipynb`** — `pace`: turn a stored `(keys, values)`
   series into an event stream; max-speed backtest (`speed=inf`) vs wall-clock
   replay (`speed=…`, injectable clock for a deterministic demo). Emphasize values
   are identical; only *timing* differs.
9. **`09-stream-shaping.ipynb`** — `dropna` (`how="any"/"all"`), `filter` (a
   predicate), `split` (inverse of `merge`) — clean bad readings, route a merged
   stream back into per-source streams. Show cardinality changing.
10. **`10-computational-dag.ipynb`** — define a graph with `Input`/`combine_latest`/
    functors (`Sub()(combine_latest(a, b))` → `RollingMean`), build a `Dag`, run it
    **batch** (`dag(...)`) and **live** (`dag.stream(...)`), and assert the outputs
    are byte-identical. The capstone: define once, run anywhere.

## Local run / verification

- Add **`nbmake`** to the dev dependency group (`pyproject.toml`). matplotlib,
  pandas, pandas-ta-classic already present.
- Acceptance command: `poetry run pytest --nbmake docs/notebooks/` — executes all
  10 notebooks; green == every example runs against the built library.
- Convenience target `make notebooks` (in the top-level `Makefile`): runs
  `make install-dev` then the nbmake command, so "build the lib + bindings + run
  notebooks" is one step. (An `invoke` task is an acceptable alternative if it
  fits `tasks.py` better.)
- A short `docs/notebooks/README.md` explains how to run/edit them.

## Testing

- **Every notebook executes clean** under `pytest --nbmake docs/notebooks/`
  (the acceptance gate).
- **Determinism** — a second run produces the same results (seeded RNG, no
  network); a notebook that computes an equality (e.g. batch == stream, or a
  pandas cross-check) asserts it in-cell so a regression fails the run.
- Notebooks use only the **public API**; a grep check that no notebook imports
  `screamer_bindings` or a `_underscore` name.

## Implementation-plan decomposition

A single plan, grouped so each task is a reviewable batch of related notebooks
plus the shared tooling:
1. **Tooling + notebook 1** — add `nbmake` dep, the `make notebooks` target, the
   `docs/notebooks/README.md`, and the quickstart notebook; establish the
   structure/conventions and the passing `pytest --nbmake` gate.
2. **Single-series notebooks (2–6)** — rolling/EW, streaming, financial, NaN,
   signal.
3. **Multi-stream + DAG notebooks (7–10)** — align, replay, shaping, DAG.

Each notebook ends green under nbmake before its task is done.

## Open decisions carried into planning

- Exact synthetic-data recipe per notebook (a small shared helper cell vs
  inline) — lean inline + seeded for self-containment, a tiny repeated snippet is
  acceptable for notebook independence.
- `make notebooks` vs an `invoke` task — pick whichever matches the repo's
  primary workflow (`tasks.py` uses `invoke`; the `Makefile` has `build`/`test`).
