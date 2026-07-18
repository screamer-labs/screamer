# Docs Navigation Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate the 26 flat doc topic categories into 8 broad groups, switch to pydata-sphinx-theme, and turn the flat example list into a card gallery, without retagging any function page.

**Architecture:** A `groups:` layer added to `docs/topics.yml` maps the existing 26 topic slugs into 8 ordered groups. `devtools/build_topic_pages.py` changes from one page per topic to one page per group (each group page has a section per member topic with its function table). `docs/index.rst` lists the 8 group pages and a notebook gallery page; `docs/conf.py` switches the theme. Function-page frontmatter is untouched.

**Tech Stack:** Sphinx, myst-nb, pydata-sphinx-theme, sphinx-design, PyYAML, pytest.

## Global Constraints

- No change to any function page's `topics:` frontmatter. The 26 topics stay as atomic tags and become sub-sections.
- No migration of notebooks to sphinx-gallery. Notebooks stay as executed myst-nb `.ipynb`.
- No old-URL redirects. `docs/by_topic/` and its pages are removed; URLs move with the version.
- The 8 group slugs, names, and order are exactly as in the spec (see Task 1).
- Group names must not contain "rolling", "window", or "moving"; no "Miscellaneous" group.
- No em-dashes in any prose (ASCII hyphens); `smartquotes = False` is already set.
- After any change, `make docs` must build clean (`nb_execution_raise_on_error = True`).
- Commit as `simu.ai <claude@sitmo.com>` with the standard `Co-Authored-By` + `Claude-Session` footer. Do not push.

---

## File Structure

- `docs/topics.yml` (modify) - add a `groups:` block above `topics:`.
- `devtools/topics.py` (modify) - add `load_groups()` + `validate_groups()`.
- `tests/test_doc_coverage.py` (modify) - add the group-coverage test.
- `devtools/build_topic_pages.py` (rewrite) - emit `docs/by_group/<slug>.rst` + `by_group_index.rst`.
- `docs/by_topic/` (delete) - replaced by `docs/by_group/`.
- `pyproject.toml` (modify) - swap `sphinx-rtd-theme` for `pydata-sphinx-theme`, add `sphinx-design`, in both `[project.optional-dependencies].docs` and `[tool.poetry.group.docs.dependencies]`.
- `docs/conf.py` (modify) - `sphinx_design` extension, `html_theme`, `html_theme_options`.
- `docs/index.rst` (modify) - Functions toctree -> 8 group pages; Examples toctree -> gallery page.
- `docs/notebooks/index.md` (create) - the card gallery.

---

## Task 1: Group layer (topics.yml + topics.py + coverage test)

**Files:**
- Modify: `docs/topics.yml`
- Modify: `devtools/topics.py`
- Test: `tests/test_doc_coverage.py`

**Interfaces:**
- Produces: `devtools.topics.load_groups() -> dict[str, {"name": str, "desc": str, "topics": list[str]}]` in display order; `devtools.topics.validate_groups()` raising `ValueError` if any topic is unmapped or double-mapped.

- [ ] **Step 1: Add the `groups:` block to `docs/topics.yml`** (above the existing `topics:` block, after the comment header)

```yaml
# Broad groups for the left-nav. Each group lists its member topic slugs (from
# `topics:` below) in display order. Every topic must belong to exactly one group.
# Group order here is the display order in the sidebar. Names avoid "rolling",
# "window", and "moving" (every operator is windowed, so those do not distinguish).
groups:
  statistics:
    name: "Statistics"
    desc: "Measures of a window: central tendency, dispersion, quantiles, moments, z-scores, running totals, and cross-series relationships."
    topics: [statistics, volatility, standardization, cumulative, regression]

  smoothing-filters:
    name: "Smoothing & filters"
    desc: "Track and denoise the level of a series: moving averages and designed filters."
    topics: [smoothing, filtering]

  indicators:
    name: "Technical indicators"
    desc: "Market indicators: trend, momentum, bands, volume, and returns."
    topics: [trend, momentum, bands, volume, returns]

  microstructure:
    name: "Market microstructure"
    desc: "Order flow and liquidity from trades and quotes: trade signing, imbalance, price impact, and arrival intensity."
    topics: [trade-signing, order-flow-imbalance, price-impact, order-flow-arrivals]

  backtesting-risk:
    name: "Backtesting & risk"
    desc: "Turn signals into costed equity curves, and measure drawdown and performance."
    topics: [risk, backtesting]

  math-logic:
    name: "Math & logic"
    desc: "Elementwise numeric, trigonometric, activation, and boolean operations."
    topics: [arithmetic, trig, activations, logic]

  data-prep:
    name: "Data preparation"
    desc: "Clean a series before analysis: fill or drop missing values, and despike outliers."
    topics: [missing-data, outliers]

  streaming:
    name: "Streaming & pipelines"
    desc: "Align and reshape event streams, and wire operators into a runnable pipeline."
    topics: [streams, graphs]
```

- [ ] **Step 2: Add `load_groups()` and `validate_groups()` to `devtools/topics.py`** (append after `topic_slugs()`)

```python
def load_groups():
    """Return an ordered dict: group slug -> {"name", "desc", "topics": [slug,...]}."""
    data = yaml.safe_load(TOPICS_YML.read_text())
    groups = data.get("groups") or {}
    for slug, entry in groups.items():
        if "name" not in entry or "desc" not in entry or "topics" not in entry:
            raise ValueError(
                f"topics.yml: group {slug!r} needs 'name', 'desc', and 'topics'")
    return groups


def validate_groups():
    """Every topic must belong to exactly one group. Raises ValueError otherwise."""
    topics = set(load_topics())
    seen = {}
    for gslug, entry in load_groups().items():
        for tslug in entry["topics"]:
            if tslug not in topics:
                raise ValueError(f"group {gslug!r} lists unknown topic {tslug!r}")
            if tslug in seen:
                raise ValueError(
                    f"topic {tslug!r} is in two groups: {seen[tslug]!r} and {gslug!r}")
            seen[tslug] = gslug
    missing = topics - set(seen)
    if missing:
        raise ValueError(f"topics not assigned to any group: {sorted(missing)}")
```

- [ ] **Step 3: Write the failing coverage test** in `tests/test_doc_coverage.py` (append)

```python
def test_every_topic_belongs_to_exactly_one_group():
    from devtools.topics import load_groups, load_topics, validate_groups
    validate_groups()                       # raises if any topic is unmapped/double-mapped
    groups = load_groups()
    assert len(groups) == 8
    mapped = [t for g in groups.values() for t in g["topics"]]
    assert sorted(mapped) == sorted(load_topics())     # exact cover, no dupes
```

- [ ] **Step 4: Run the test**

Run: `poetry run python -m pytest tests/test_doc_coverage.py -q -k "group"`
Expected: PASS (8 groups, exact cover). If it fails with a `ValueError`, a topic slug in the `groups:` block is misspelled or a topic is unmapped; fix `topics.yml`.

- [ ] **Step 5: Confirm nothing else broke** (the `groups:` key is additive; `load_topics` ignores it)

Run: `poetry run python -m pytest tests/test_doc_coverage.py -q`
Expected: all pass, including `test_topics_registry_loads_in_display_order` (the `topics:` block order is unchanged).

- [ ] **Step 6: Commit**

```bash
git add docs/topics.yml devtools/topics.py tests/test_doc_coverage.py
git commit -m "docs(nav): add 8-group layer over the 26 topics in topics.yml"
```

---

## Task 2: Switch to pydata-sphinx-theme

**Files:**
- Modify: `pyproject.toml`
- Modify: `docs/conf.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: docs build on pydata-sphinx-theme with `sphinx_design` available. The nav is still the current 26-topic structure at this point.

- [ ] **Step 1: Swap the theme dependency in `pyproject.toml`** (two places)

In `[project.optional-dependencies]` `docs = [ ... ]`, replace `"sphinx-rtd-theme>=3.0.1",` with:
```toml
    "pydata-sphinx-theme>=0.16",
    "sphinx-design>=0.6",
```
In `[tool.poetry.group.docs.dependencies]`, replace `sphinx-rtd-theme = "^3.0.1"` with:
```toml
pydata-sphinx-theme = "^0.16"
sphinx-design = "^0.6"
```

- [ ] **Step 2: Install the updated docs deps**

Run: `poetry lock && poetry install --with docs`
Expected: `pydata-sphinx-theme` and `sphinx-design` install; `sphinx-rtd-theme` is dropped.

- [ ] **Step 3: Enable `sphinx_design` and switch the theme in `docs/conf.py`**

Add `"sphinx_design",` to the `extensions` list (after `"sphinx_exec_code",`). Replace the theme block:

```python
html_theme = 'pydata_sphinx_theme'

html_static_path = ['_static']
html_css_files = ['css/custom.css']

html_theme_options = {
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],           # top-level toctree captions become nav items
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "secondary_sidebar_items": ["page-toc"],   # right "on this page" sidebar
    "show_nav_level": 1,                        # sidebar shows the current section only
    "navigation_depth": 3,
    "icon_links": [
        {"name": "GitHub",
         "url": "https://github.com/screamer-labs/screamer",
         "icon": "fa-brands fa-github"},
        {"name": "PyPI",
         "url": "https://pypi.org/project/screamer/",
         "icon": "fa-solid fa-box"},
    ],
}
```

- [ ] **Step 4: Build the docs on the new theme**

Run: `make docs`
Expected: exit 0. The page renders with a top navbar (User Guide, Functions, Examples, References from the existing `index.rst` captions) and a contextual left sidebar. The 26-topic Functions list is still present (unchanged in this task). Investigate any `sphinx_rtd_theme`-only option warnings and confirm none remain.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml poetry.lock docs/conf.py
git commit -m "docs(nav): switch to pydata-sphinx-theme + enable sphinx-design"
```

---

## Task 3: Group pages (rewrite build_topic_pages.py + index.rst Functions toctree)

**Files:**
- Rewrite: `devtools/build_topic_pages.py`
- Modify: `docs/index.rst`
- Delete: `docs/by_topic/` (the script removes it)

**Interfaces:**
- Consumes: `load_groups()` and `load_topics()` from Task 1; `sphinx_design` grid/card directives from Task 2.
- Produces: `docs/by_group/<group>.rst` (8 pages) and `docs/by_group_index.rst` (card-grid landing). The Functions sidebar shows the 8 group pages.

- [ ] **Step 1: Replace `devtools/build_topic_pages.py` with the group version**

```python
#!/usr/bin/env python3
"""Auto-generate the grouped FUNCTIONS index from docs/topics.yml + help.json.

Output:
    docs/by_group_index.rst    -- the FUNCTIONS landing: a sphinx-design card grid
                                  of the 8 groups, plus a hidden toctree that homes
                                  every function reference page (so none are orphaned).
    docs/by_group/<slug>.rst   -- one page per GROUP; each has a section per member
                                  topic (docs/topics.yml order) with that topic's
                                  function table.

Run after every help.json update:
    poetry run python devtools/build_topic_pages.py
"""
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from devtools.topics import load_topics, load_groups, validate_groups

HELP_JSON = ROOT / "screamer" / "data" / "help.json"
DOCS = ROOT / "docs"
OUT_DIR = DOCS / "by_group"
OLD_DIR = DOCS / "by_topic"
INDEX_FILE = DOCS / "by_group_index.rst"

GENERATED_BANNER = (
    ".. NOTE: This file is auto-generated by devtools/build_topic_pages.py\n"
    ".. Do not edit by hand. Re-run the script after updating help.json.\n\n"
)
FM = re.compile(r"---\n(.*?)\n---\n", re.S)


def _page_refs() -> dict[str, str]:
    """name -> 'functions_<subdir>/<Name>' for every reference page that exists."""
    refs = {}
    for md in sorted(DOCS.glob("functions_*/*.md")):
        m = FM.match(md.read_text())
        if not m:
            continue
        name = (yaml.safe_load(m.group(1)) or {}).get("name")
        if name:
            refs[name] = f"{md.parent.name}/{md.stem}"
    return refs


def _table(members, refs, shorts, titles) -> list[str]:
    lines = [".. list-table::", "   :header-rows: 1", "   :widths: 30 70", "",
             "   * - Function", "     - Description"]
    for fn in members:
        ref = refs.get(fn)
        display = titles.get(fn, fn)
        doc_ref = f":doc:`{display} </{ref}>`" if ref else f"``{display}``"
        short = (shorts.get(fn) or "").replace("|", "\\|")
        lines += [f"   * - {doc_ref}", f"     - {short}"]
    lines.append("")
    return lines


def build_group_page(group, topics, members, refs, shorts, titles) -> str:
    title = group["name"]
    lines = [GENERATED_BANNER + title, "=" * len(title), "", group["desc"], ""]
    for tslug in group["topics"]:
        meta = topics[tslug]
        heading = meta["name"]
        lines += [heading, "-" * len(heading), "", meta["desc"], ""]
        lines += _table(sorted(members.get(tslug, []), key=str.lower),
                        refs, shorts, titles)
    return "\n".join(lines)


def build_index(groups, all_refs) -> str:
    lines = [GENERATED_BANNER + "Functions", "=========", "",
             "Every function, grouped by what it does. Pick a group to browse; "
             "the search box finds a function by name.", "",
             ".. grid:: 1 2 2 3", "   :gutter: 3", ""]
    for slug, g in groups.items():
        lines += [f"   .. grid-item-card:: {g['name']}",
                  f"      :link: by_group/{slug}", "      :link-type: doc", "",
                  f"      {g['desc']}", ""]
    # Hidden toctree so every reference page is homed exactly once (no orphans).
    lines += [".. toctree::", "   :hidden:", ""]
    lines += [f"   {ref}" for ref in sorted(all_refs)]
    lines.append("")
    return "\n".join(lines)


def main():
    validate_groups()
    topics = load_topics()
    groups = load_groups()
    help_data = json.loads(HELP_JSON.read_text())
    refs = _page_refs()
    shorts = {n: e.get("short", "") for n, e in help_data.items()}
    titles = {n: e.get("title", n) for n, e in help_data.items()}

    members: dict[str, list[str]] = {slug: [] for slug in topics}
    for name, entry in help_data.items():
        if entry.get("index") is False:      # covered twins (_iter, Input, Node)
            continue
        for slug in entry.get("topics", []):
            if slug in members:
                members[slug].append(name)

    if OLD_DIR.exists():                      # drop the previous per-topic taxonomy
        for stale in OLD_DIR.glob("*.rst"):
            stale.unlink()
        OLD_DIR.rmdir()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale in OUT_DIR.glob("*.rst"):
        stale.unlink()
    for slug, group in groups.items():
        page = build_group_page(group, topics, members, refs, shorts, titles)
        (OUT_DIR / f"{slug}.rst").write_text(page)

    INDEX_FILE.write_text(build_index(groups, set(refs.values())))
    listed = sum(len(members[t]) for g in groups.values() for t in g["topics"])
    print(f"Wrote {INDEX_FILE.name} and {len(groups)} group pages "
          f"({listed} listings over {len(refs)} pages).")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the generator and verify coverage**

Run: `poetry run python devtools/build_topic_pages.py`
Expected: `Wrote by_group_index.rst and 8 group pages (... listings over 219 pages).` and `docs/by_topic/` is gone.

Run: `ls docs/by_group/ && test ! -d docs/by_topic && echo "old dir removed"`
Expected: 8 `.rst` files listed, `old dir removed`.

Verify no function is dropped:
```bash
poetry run python - <<'PY'
import re, glob
refs = set()
for f in glob.glob('docs/by_group/*.rst') + ['docs/by_group_index.rst']:
    refs |= set(re.findall(r'functions_\w+/\w+', open(f).read()))
import glob as g
pages = {f'{p.split("/")[-2]}/{p.split("/")[-1][:-3]}' for p in g.glob('docs/functions_*/*.md')}
missing = pages - refs
print("pages not reachable from a group page or the index:", sorted(missing))
PY
```
Expected: empty list.

- [ ] **Step 3: Point `docs/index.rst` Functions toctree at the 8 group pages**

Replace the entire `Functions` toctree block (the caption plus its 26 `by_topic/<topic>` entries) with:

```rst
.. toctree::
   :maxdepth: 1
   :caption: Functions
   :hidden:

   by_group_index
   by_group/statistics
   by_group/smoothing-filters
   by_group/indicators
   by_group/microstructure
   by_group/backtesting-risk
   by_group/math-logic
   by_group/data-prep
   by_group/streaming
```

- [ ] **Step 4: Build and eyeball the grouped nav**

Run: `make docs`
Expected: exit 0. The Functions navbar section opens the card-grid landing (`by_group_index`); the sidebar lists the 8 group pages; each group page shows its member topics as sub-sections with function tables.

- [ ] **Step 5: Commit**

```bash
git add devtools/build_topic_pages.py docs/index.rst docs/by_group docs/by_group_index.rst
git rm -r --cached docs/by_topic 2>/dev/null || true
git commit -m "docs(nav): generate 8 group pages, retire the 26 per-topic pages"
```

---

## Task 4: Example gallery (card grid over the notebooks)

**Files:**
- Create: `docs/notebooks/index.md`
- Modify: `docs/index.rst`

**Interfaces:**
- Consumes: `sphinx_design` from Task 2.
- Produces: an Examples landing page whose sidebar entry is the gallery, not the 15 notebooks.

- [ ] **Step 1: Create `docs/notebooks/index.md`** (the gallery; myst markdown with sphinx-design grids)

````markdown
# Examples

Runnable notebooks, grouped by task. Each one executes on real or seeded data at
build time.

## Getting started

```{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Quickstart: the polymorphic API
:link: 01-quickstart-polymorphic-api
:link-type: doc
One operator on scalars, arrays, and live streams.
:::

:::{grid-item-card} NaN handling
:link: 05-nan-handling
:link-type: doc
How missing values flow through operators and warm up.
:::
```

## Statistics & indicators

```{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Window statistics
:link: 02-window-statistics
:link-type: doc
Rolling mean, dispersion, quantiles, and ranks.
:::

:::{grid-item-card} Financial indicators
:link: 03-financial-indicators
:link-type: doc
Moving averages, momentum, bands, and volume.
:::

:::{grid-item-card} Signal processing
:link: 04-signal-processing
:link-type: doc
Filters and smoothers on noisy series.
:::
```

## Streaming & pipelines

```{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Streaming live events
:link: 06-streaming-live-events
:link-type: doc
Feed events one at a time; identical results to batch.
:::

:::{grid-item-card} Multi-stream operators
:link: 07-multi-stream-operators
:link-type: doc
Align and combine streams that do not tick together.
:::

:::{grid-item-card} Pipelines
:link: 08-pipelines
:link-type: doc
Wire operators into a reusable graph.
:::

:::{grid-item-card} Bars from ticks
:link: 09-bars-from-ticks
:link-type: doc
Resample a trade tape into OHLC bars.
:::

:::{grid-item-card} Custom and multi-column bars
:link: 10-custom-and-multi-column-bars
:link-type: doc
Build bespoke bar aggregations.
:::
```

## Market microstructure

```{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Order flow and trade signs
:link: 11-microstructure-order-flow
:link-type: doc
Recover trade direction and order-flow imbalance.
:::

:::{grid-item-card} Price impact and liquidity
:link: 12-microstructure-price-impact
:link-type: doc
Kyle's lambda, Amihud, Roll spread, and the propagator.
:::

:::{grid-item-card} Toxicity, book pressure, and spreads
:link: 13-microstructure-toxicity-and-book
:link-type: doc
VPIN, queue imbalance, micro-price, and spread decomposition.
:::
```

## Backtesting

```{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Backtesting a signal
:link: 14-backtesting-a-signal
:link-type: doc
From a position signal to a costed equity curve.
:::

:::{grid-item-card} Event-driven backtests
:link: 15-event-driven-backtests
:link-type: doc
Fills on bars, the trade tape, and top-of-book quotes.
:::
```

```{toctree}
:hidden:

01-quickstart-polymorphic-api
02-window-statistics
03-financial-indicators
04-signal-processing
05-nan-handling
06-streaming-live-events
07-multi-stream-operators
08-pipelines
09-bars-from-ticks
10-custom-and-multi-column-bars
11-microstructure-order-flow
12-microstructure-price-impact
13-microstructure-toxicity-and-book
14-backtesting-a-signal
15-event-driven-backtests
```
````

- [ ] **Step 2: Point `docs/index.rst` Examples toctree at the gallery page**

Replace the entire `Examples` toctree block (the caption plus its 15 `notebooks/NN-...` entries) with:

```rst
.. toctree::
   :maxdepth: 1
   :caption: Examples
   :hidden:

   notebooks/index
```

- [ ] **Step 3: Build and check the gallery**

Run: `make docs`
Expected: exit 0. The Examples navbar section opens the gallery page; the sidebar shows "Examples" (the gallery), not 15 flat entries; every card links to a notebook that still executes.

Confirm the notebook README stays excluded and the notebooks still render:
```bash
ls docs/_build/html/notebooks/index.html docs/_build/html/notebooks/14-backtesting-a-signal.html
```
Expected: both exist.

- [ ] **Step 4: Commit**

```bash
git add docs/notebooks/index.md docs/index.rst
git commit -m "docs(nav): example gallery card grid over the notebooks"
```

---

## Task 5: Landing card grid + final verification

**Files:**
- Modify: `docs/index.rst`

**Interfaces:**
- Consumes: the 8 group pages (Task 3) and the gallery (Task 4).

- [ ] **Step 1: Add a card grid to the landing page** in `docs/index.rst` (after the short intro/example, before the first hidden toctree). Use the `eval-rst`-free RST grid directive:

```rst
.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Statistics
      :link: by_group/statistics
      :link-type: doc

      Central tendency, dispersion, quantiles, moments, and correlation.

   .. grid-item-card:: Smoothing & filters
      :link: by_group/smoothing-filters
      :link-type: doc

      Moving averages and designed filters.

   .. grid-item-card:: Technical indicators
      :link: by_group/indicators
      :link-type: doc

      Trend, momentum, bands, volume, and returns.

   .. grid-item-card:: Market microstructure
      :link: by_group/microstructure
      :link-type: doc

      Trade signing, imbalance, price impact, and arrivals.

   .. grid-item-card:: Backtesting & risk
      :link: by_group/backtesting-risk
      :link-type: doc

      Costed equity curves, drawdown, and performance.

   .. grid-item-card:: Examples
      :link: notebooks/index
      :link-type: doc

      Runnable notebooks grouped by task.
```

- [ ] **Step 2: Full build and regen chain**

Run:
```bash
poetry run python devtools/build_help_registry.py
poetry run python devtools/build_topic_pages.py
make regen-init
make docs
```
Expected: help registry unchanged and valid; 8 group pages regenerated; `make docs` exit 0.

- [ ] **Step 3: Full test suite**

Run: `poetry run python -m pytest -q`
Expected: all pass, including the Task 1 group test and `test_doc_coverage.py`.

- [ ] **Step 4: Commit**

```bash
git add docs/index.rst
git commit -m "docs(nav): landing-page card grid to the function groups and examples"
```

---

## Self-Review

**Spec coverage:**
- Lever 1 (26 -> 8 groups, mapping layer, no retagging): Task 1 (data model) + Task 3 (group pages + index). Covered.
- Lever 2 (pydata-sphinx-theme): Task 2. Covered.
- Lever 3 (card gallery over notebooks): Task 4. Covered.
- Landing card grid (scikit-learn pattern): Task 5. Covered.
- `topics.py` group reader + validation; `test_doc_coverage` group check: Task 1. Covered.
- Docs deps (`pydata-sphinx-theme`, `sphinx-design`) in both dependency tables: Task 2. Covered.
- `docs/by_topic/` removed, no redirects: Task 3 (generator deletes it; index stops referencing it). Covered.
- Spec's "per-function Topics: back-link -> group anchor": N/A. The sphinx function pages render no such link (that is the separate `help.json` frontend), confirmed by inspection; nothing to change. Noted here so a reviewer does not look for it.

**Placeholder scan:** No TBD/TODO. Every code step shows complete file content or an exact replacement block. The two `index.rst` toctree replacements name the exact caption and entries to swap.

**Type consistency:** `load_groups()` returns `{"name","desc","topics"}` and every consumer (`validate_groups`, `build_group_page`, `build_index`) uses those keys. The 8 group slugs are identical across `topics.yml`, `build_topic_pages.py` output paths, and the `index.rst` toctree (`statistics`, `smoothing-filters`, `indicators`, `microstructure`, `backtesting-risk`, `math-logic`, `data-prep`, `streaming`). Notebook card `:link:` targets match the notebook file stems.

---

## Notes for the implementer

- The `groups:` block is additive to `topics.yml`; `load_topics()` reads only the `topics:` key, so existing behavior is unchanged.
- `make docs` executes all 15 notebooks; keep it as the final gate after Tasks 3, 4, 5.
- pydata-sphinx-theme builds the navbar from `index.rst`'s captioned top-level toctrees (User Guide, Functions, Examples, References). No extra navbar config is needed beyond `navbar_center: ["navbar-nav"]`.
- Do not reorder the `topics:` block in `topics.yml`; `test_topics_registry_loads_in_display_order` asserts its first/last entries.
