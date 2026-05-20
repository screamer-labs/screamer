# Help-registry JSON structure refactor

**Status:** approved design, ready for implementation planning
**Date:** 2026-05-20
**Scope:** `screamer/data/help.json`, `devtools/build_help_registry.py`, all `docs/functions_*/<Name>.md` source files, the sphinx render of those files

## Motivation

The library ships `screamer/data/help.json` (146 entries) for external frontends to deeply integrate API docs. Each entry currently has a freeform `body_markdown` string that mixes prose, math, and code fences. Consumers cannot reliably strip code out of the prose, so frontends that want to render only `[short, details]` (i.e. without inline code) have no clean way to do so. Separately, the `## Usage Example` sections currently sit *below* the `<!-- HELP_END -->` marker, so the most useful runnable examples never reach JSON consumers at all.

This refactor:

1. Replaces `body_markdown` with two structured fields: a `details` markdown string guaranteed to contain no code fences, and an `examples` list of `{language, caption, code}` objects.
2. Restructures every source `.md` file so that examples live under an explicit `## Examples` section above HELP_END, with each example introduced by a `### Caption` heading.
3. Brings the sphinx render in line with the new structure: sphinx pages also show an `## Examples` H2 with captioned `### Caption` sub-sections, and continue to render plotly figures inline.

No backwards compatibility shim is provided — the existing field `body_markdown` is removed outright. This is treated as a breaking change to the JSON contract, gated only by a version bump and a CHANGELOG entry.

## Non-goals

- Shipping pre-rendered plot output (plotly figure JSON, base64 PNGs, output tables) in `help.json`. JSON consumers receive source code only. Sphinx remains the only place where examples are executed and figures are rendered.
- Adding a richer summary paragraph alongside `short`. The existing one-line `short` field stays as-is and continues to play the role of the chip/tagline.
- Reshaping the `parameters` schema or any other existing frontmatter field.
- Migrating the post-HELP_END `## Reference` / `## Implementation Details` sections into JSON. They stay sphinx-only.

## JSON schema (per entry)

Unchanged frontmatter fields: `name`, `title`, `implementation_family`, `topics`, `tags`, `short`, `inputs`, `outputs`, `parameters`.

Removed: `body_markdown`.

New:

- `details` — **string, required**. Markdown body covering the description, math, parameters notes, formulas, edge cases, etc. Guaranteed to contain no fenced code blocks. Includes whatever `## H2` sub-structure the author wants (`## Description`, `## Formula`, `## Notes`, …). The leading `# \`Name\`` H1 line from the source `.md` is stripped because consumers already have `name` and `title`.
- `examples` — **list, required (possibly empty)**. Each element:
  - `language` — **string, required**. Defaults to `"python"`. Sourced from the fence info string in the markdown.
  - `caption` — **string, required**. A short description of what the example demonstrates. Sourced from the preceding `### H3` heading.
  - `code` — **string, required**. Raw source code, no fences, no `eval-rst` directive wrapper, dedented.

Example JSON entry:

```jsonc
{
  "EwCorr": {
    "name": "EwCorr",
    "title": "Exponentially-weighted correlation",
    "implementation_family": "ew",
    "topics": ["correlation", "statistics"],
    "tags": ["ew", "correlation", "pair"],
    "short": "EW Pearson correlation of two parallel streams.",
    "inputs": 2, "outputs": 1,
    "parameters": [ /* unchanged */ ],
    "details": "## Description\n\n`EwCorr` computes the exponentially weighted moving Pearson correlation …\n\n## Formula\n\n…",
    "examples": [
      {
        "language": "python",
        "caption": "Tracking a regime shift in correlation",
        "code": "import numpy as np\nimport plotly.graph_objects as go\nfrom screamer import EwCorr\n…"
      },
      {
        "language": "python",
        "caption": "Identity vs EwCov / sqrt(EwVar(x) * EwVar(y))",
        "code": "from screamer import EwCorr, EwCov, EwVar\n…"
      }
    ]
  }
}
```

## Source markdown convention

Canonical layout for every `docs/functions_*/<Name>.md` (outer 4-backtick fence shown only so the inner 3-backtick fences render as literal markdown):

````markdown
---
name: EwCorr
title: Exponentially-weighted correlation
... (frontmatter unchanged) ...
---

# `EwCorr`

## Description

`EwCorr` computes the exponentially weighted moving Pearson correlation …

## Formula

…math, prose, more H2 sections as needed, NO code fences here…

## Examples

### Tracking a regime shift in correlation

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from screamer import EwCorr
    …
```

### Identity vs EwCov / sqrt(EwVar(x) * EwVar(y))

```python
from screamer import EwCorr, EwCov, EwVar
…
```

<!-- HELP_END -->

## Reference

…sphinx-only content (academic refs, implementation notes)…
````

Rules:

- `## Examples` is the structural marker. **Every code fence destined for JSON must live under it.** Code anywhere else in the pre-HELP_END region is a build error.
- Each `### <Caption>` under `## Examples` defines one example object. The heading text becomes `caption`.
- Two fence flavors are supported:
  - A fence whose info string starts with `{eval-rst}` and whose body is a `.. plotly::` directive → sphinx renders the live plot; the build script unwraps the indented python under the directive and dedents it. `language = "python"`.
  - A plain fence with a language tag (typically `python`) → no plot; body is taken as-is, fence info string becomes `language`.
- `<!-- HELP_END -->` stays as the boundary between "in JSON" (above) and "sphinx-only extras" (below). It now sits after `## Examples`.
- Files with no examples simply omit the `## Examples` section. `examples` becomes `[]` in JSON.

Sphinx renders this source unchanged: `## Examples` becomes a normal H2 in the rendered docs, `### Captions` become H3 sub-sections, and `{eval-rst} .. plotly::` blocks still produce live plots — so the sphinx pages also get the cleaner structure with no sphinx-side configuration changes.

## Build script changes (`devtools/build_help_registry.py`)

New `parse_file()` splits the post-frontmatter text into three regions by structural markers:

1. **Details region** — from the start of the body up to either `## Examples` or `<!-- HELP_END -->`, whichever comes first.
2. **Examples region** — from `## Examples` up to `<!-- HELP_END -->` (or end of file).
3. **Sphinx-only region** — after `<!-- HELP_END -->`. Discarded for JSON.

Processing:

- `details` ← Details region, with the leading `# \`Name\`` H1 line stripped and surrounding whitespace trimmed.
- `examples` ← walk Examples region, splitting on `### ` H3 headings. For each H3:
  - `caption` ← the H3 heading text.
  - The next fenced code block becomes `code`:
    - If the fence info string starts with `{eval-rst}` and the body is a `.. plotly::` directive, the indented python under it is dedented and extracted; `language = "python"`. Directive options like `:include-source:` are discarded.
    - Otherwise the fence info string is the `language` (defaulted to `"python"` if empty) and the body is taken as-is.

Build-time validation (fail loudly with the path and a precise message):

- Details region contains any fenced code → error: "code fence outside `## Examples` section".
- An `### Heading` under `## Examples` is not immediately followed by a code block → error.
- A code block appears under `## Examples` with no preceding `### Heading` in the same section → error.
- The existing pybind-vs-schema parameter-name drift check is preserved.
- The existing live round-trip (instantiate with defaults, call on a synthetic array) is preserved.

## Migration (`devtools/migrate_docs_v2.py`, deletable after run)

A one-shot script that walks all 146 files and rewrites in place. The script must be idempotent (running it twice on a converted file is a no-op).

File counts and behaviour by shape (categories are mutually exclusive when defined as below; total = 146):

- **Only a post-HELP_END `## Usage Example*` section** (85 files): move that section above HELP_END, demote the H2 to a new `## Examples` parent containing one `### <caption>` child. Starter caption derived from the original heading (e.g. `## Usage Example and Plot` → `### Usage example`). Every migrated file is logged so the team can hand-tune captions in a follow-up pass.
- **Only pre-HELP_END code fences embedded in prose** (13 files: ADX, ButterBandpass, ButterBandstop, EwKurt, EwMean, EwRms, EwSkew, EwStd, EwVar, EwZscore, RollingCalmar, RollingMaxDrawdown, RollingResidualStd): extract each fence, append it as a `### <caption>` block under a new `## Examples` section just before HELP_END, and remove the fence from its original location. Surrounding prose stays in place; a dangling intro line left behind ("By definition…" with nothing under it) is preserved as plain prose. These files are flagged in the run log for manual review — they're the most likely to need an authoring pass to reconnect prose.
- **Both** (3 files: EwCorr, MACD, WilliamsR): combine — extract pre-HELP_END fences as separate `### <caption>` children, and also migrate the post-HELP_END `## Usage Example*` section as another `### <caption>` child, all under a single `## Examples` section. These files are flagged for manual review.
- **No code anywhere** (45 files: ATR etc.): no-op; no `## Examples` section is added.

The migration script prints a summary table: per file, what was moved/extracted and whether a hand-review flag was set.

## Validation / verification plan

1. Run the migration script.
2. Run `poetry run python devtools/build_help_registry.py` — it must succeed for every file, or fail with a clear pointer to the file that still needs hand-editing.
3. Open `screamer/data/help.json` and confirm:
   - No entry has a `body_markdown` field.
   - Every entry has `details` (string) and `examples` (list).
   - `details` for at least the 16 previously-mixed entries contains no triple-backtick.
4. Build sphinx docs locally (`cd docs && make html`) and spot-check 8–10 pages spanning families (`ew`, `rolling`, `fin`, `math`, `signal`, …):
   - The new `## Examples` H2 renders.
   - `### Captions` render as H3 sub-sections.
   - Plotly figures still render inline (no broken directives).
5. Manual caption pass: review the migration log, hand-tune captions across the migrated files. Pure text edits, no code changes.

## Consumer / cleanup sweep

- `devtools/report_docs.py` — replace any reference to `body_markdown` with `details` / `examples` as appropriate.
- Any tests reading `help.json` (search the test tree for `body_markdown` and for `help.json`).
- CHANGELOG entry under the next version describing the schema change and listing the removed/added fields.
- Version bump if not already covered by an in-flight bump.

## Open / deferred items

None at design time. Caption-quality refinement after migration is anticipated but explicitly out of scope for the structural change — it is a routine docs follow-up.
