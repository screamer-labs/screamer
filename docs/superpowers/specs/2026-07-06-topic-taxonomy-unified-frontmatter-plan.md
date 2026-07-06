# Unified frontmatter + topic taxonomy — plan

**Date:** 2026-07-06
**Status:** plan for review
**Inputs:** [`2026-07-06-topic-taxonomy-proposal.md`](2026-07-06-topic-taxonomy-proposal.md)
(20 topics + per-function assignments + intended overlaps) and
[`2026-07-06-topics.yml`](2026-07-06-topics.yml) (the slug -> name/desc registry).

## Goal

1. **Every** public function (150 functors + 15 stream operators + 3 DAG names =
   168) carries YAML frontmatter, so all metadata is machine-readable and no
   function can ship undocumented.
2. Each function is enriched with a manual, many-to-many **`topics: [slug, ...]`**
   assignment. Topics become the **main categories in the left nav** (a single
   topic-based FUNCTIONS index; one canonical page per function, linked from every
   topic it belongs to).
3. `topics.yml` is the single source of truth for topic slug -> display name +
   description; the index and per-page "Topics:" links read from it. CI fails on an
   unknown slug or a function with zero topics.

## Validation already done

Cross-checked the proposal's assignments against the full function universe:
**168/168 functions covered, zero orphans, zero non-existent names.** Distribution:
123 functions in one topic, 45 in two. So the taxonomy is complete and the
"every function >= 1 topic" CI rule will pass on day one.

## Current state (what changes)

- `build_help_registry.py` reads `docs/functions_*/<Name>.md` frontmatter into
  `screamer/data/help.json`, validating each by **instantiating the pybind class**
  with documented defaults and calling it. Works only for functor classes.
- Stream/DAG pages are plain `.. autofunction::`/`.. autoclass::` autodoc with **no
  frontmatter** -> absent from `help.json`.
- Topic display names live **hardcoded** in `build_topic_pages.py` as a 17-entry
  `TOPICS` dict (an older taxonomy: `math`, `transforms`, `data-handling`,
  `oscillator`, ... — different slugs from the new 20).
- The nav has two function sections: **Functions** (family toctrees `topic_math`,
  `topic_rolling`, ... in `index.rst`) and **Browse by topic** (`by_topic_index`,
  generated). The new design collapses these into one topic-based index.

## Decisions (the proposal's 5 open items, resolved)

1. **Statistics stays one broad topic** (search-friendly; Volatility/Risk/Momentum
   give sharper entry points into the same functions).
2. **Streams stays one topic** — `topics.yml` already commits to a single `streams`
   slug. (The notebook-style combining/reshaping split is a docs narrative, not an
   index axis.)
3. **`_iter` variants get frontmatter** (so coverage passes and they appear in
   `help.json`) but are **hidden from the topic index** via an `index: false`
   frontmatter flag; they are documented on/beside their parent operator. This is
   the one genuinely new frontmatter field.
4. **Standardization & normalization stays** (z-score is a distinct search).
5. **Canonical names/slugs are locked to `topics.yml` verbatim** — frontmatter uses
   the slugs, the index renders the `name`.

## Tasks

### Task A — Land the topic registry
Add `topics.yml` (provided) at `docs/topics.yml` — it is docs-only (no runtime
consumer, so not shipped like `help.json`) and hand-edited alongside the `topics:`
frontmatter. A small loader (`devtools/topics.py`) exposes `load_topics() -> dict`
(ordered slug -> {name, desc}) for the generators and tests.

### Task B — Teach the help registry about functions
Extend `build_help_registry.py`:
- Support a `kind:` frontmatter field (`functor` default, or `function`).
- For `kind: function`: validate documented params against `inspect.signature(fn)`
  (not the pybind constructor) and **skip** the instantiate-and-call round-trip.
  DAG handles (`Node`) and other non-callables take a minimal entry (`short` +
  `title` + `topics`, no `parameters`/`nan_policy`).
- Make `nan_policy` required only where it applies (stream operators: yes; DAG: no).
- **New validation:** every `topics` slug must exist in `topics.yml`; every entry
  must declare >= 1 topic. Fail the build otherwise.
- Honor `index: false` (carried into `help.json`) for the `_iter` variants.

### Task C — Frontmatter for the 18 stream/DAG pages
Add frontmatter to `functions_streams/*.md` (15) and `functions_dag/*.md` (3):
`name`, `title`, `short`, `kind: function`, `topics:` (per the proposal — Streams
for all stream ops, plus Missing-data for `dropna`/`dropna_iter`; Computation graphs
for the DAG names), `parameters` from the signature, `nan_policy` for stream ops,
and `index: false` on the `_iter` variants. Keep the existing autodoc body.

### Task D — Re-taxonomize the 150 functor pages
Rewrite each functor page's `topics:` list to the new slugs from the proposal's
assignments. Mechanical: invert the proposal (function -> [slugs]) and rewrite the
`topics:` block in place, leaving all other frontmatter and prose untouched. A
migration script (`devtools/migrate_topics.py`) does this from the proposal file so
it is auditable and re-runnable.

### Task E — Regenerate `help.json` and the topic index
- `build_help_registry.py` -> `help.json` now includes all 168 entries with the new
  topics. (Also verifies the param-capture / compliance consumers still pass.)
- Rewrite `build_topic_pages.py` to read `topics.yml` (drop the hardcoded `TOPICS`),
  emit topic pages in the registry's display order, list each function once per topic
  it declares, skip `index: false` entries, and link each function's single canonical
  page (`functions_<family>/<Name>.md`, resolved via `implementation_family`).

### Task F — Restructure the nav
Merge the two current function nav sections into **one**, keeping the caption
**"Functions"**: its toctree is the 20 topic pages (in `topics.yml` order), replacing
both the family toctrees (`topic_math`, `topic_rolling`, ...) and the separate
"Browse by topic" section (`by_topic_index`). The family reference pages
(`topic_*.rst`) are **retired**. Canonical per-function pages stay where they are
(`functions_<family>/<Name>.md`) as the content; a function in two topics is listed
under both topic pages. Each function page gains a rendered "Topics:" line linking
back to its topics.

Resulting left nav: `User Guide -> Examples -> Concepts -> Functions (20 topic
pages) -> Release notes`.

### Task G — Guardrails (the safety net)
- `tests/test_doc_coverage.py`: enumerate every public name
  (`get_module_public_classes` + `streams.__all__` + `dag.__all__`) and assert each
  has a `help.json` entry; the failure message lists exactly which names are
  undocumented. Now achievable because *all* functions have frontmatter.
- A test asserting every `help.json` entry's topics are valid `topics.yml` slugs and
  non-empty (belt-and-suspenders alongside the build-time check).

## Ordering

A (registry) -> B (registry supports functions) -> C (stream/DAG frontmatter) +
D (functor re-taxonomy) -> E (regenerate help.json + topic pages) -> F (nav) ->
G (coverage/validation tests) -> docs build green.

## Resolved decisions

- `topics.yml` lives at `docs/topics.yml` (docs-only, not shipped).
- `_iter` variants get frontmatter (so the coverage test counts them) but
  `index: false`, so the topic index skips their rows; they are documented beside
  their parent operator.
- The "Functions" and "Browse by topic" nav sections merge into a single
  "Functions" section (the 20 topic pages, `topics.yml` order); the family
  reference pages (`topic_*.rst`) are retired.

Resolved: the "Functions" and "Browse by topic" nav sections merge into a single
"Functions" section (the 20 topic pages); the family reference pages are retired.
