# Docs navigation redesign

## Context

The screamer docs have 219 function reference pages, a User Guide, 15 example
notebooks, and a handful of concept pages, built with Sphinx on the
`sphinx_rtd_theme`. The left sidebar carries three problems:

- The "Functions" caption is a flat list of **26 topic categories**. There are too
  many, several names are cryptic, and the order is close to arbitrary.
- The "Examples" caption is a flat list of **15 notebooks**, which reads as a long
  undifferentiated menu.
- The RTD theme puts everything in one sidebar at once, so the whole surface is
  visible on every page and nothing feels scoped.

Every large PyData library (NumPy, SciPy, pandas, scikit-learn, matplotlib) solves
the same scale problem the same way: a top navbar splits the docs into a few
sections, the sidebar shows only the current section, the API reference is a small
set of broad guessable groups (PyTorch's `torch` page is ~8-9), and examples are a
thumbnail card grid grouped into themes rather than a flat list.

This redesign applies that pattern to screamer.

## Goals

1. Replace the 26 flat topic categories with **8 broad, guessable groups**, in a
   deliberate order, without retagging any function page.
2. Replace the flat example list with a **card-grid gallery** grouped into themes.
3. Switch to **`pydata-sphinx-theme`** so the navbar carries the top-level
   sections and the sidebar only ever shows the current one.

## Non-goals

- No change to any function's `topics:` frontmatter (the 26 topics survive as
  sub-sections).
- No migration of notebooks to `sphinx-gallery` scripts (the notebooks stay as
  executed myst-nb `.ipynb`).
- No rewrite of function reference prose or examples.

## Lever 1: 26 topics into 8 groups (mapping layer, not retagging)

The 26 topics stay as the atomic tags on every function page and become the
**sub-sections inside each group**. A new group layer in `docs/topics.yml` maps
them into 8 groups. No function page changes.

### The 8 groups (slug, display name, member topics in order)

1. `statistics` **Statistics** - statistics, volatility, standardization,
   cumulative, regression
2. `smoothing-filters` **Smoothing & filters** - smoothing, filtering
3. `indicators` **Technical indicators** - trend, momentum, bands, volume, returns
4. `microstructure` **Market microstructure** - trade-signing,
   order-flow-imbalance, price-impact, order-flow-arrivals
5. `backtesting-risk` **Backtesting & risk** - risk, backtesting
6. `math-logic` **Math & logic** - arithmetic, trig, activations, logic
7. `data-prep` **Data preparation** - missing-data, outliers
8. `streaming` **Streaming & pipelines** - streams, graphs

Every one of the 26 existing topic slugs appears in exactly one group. Group names
avoid the words "rolling", "window", and "moving" because every screamer operator
is windowed, so those words carry no distinguishing information. There is no
"Miscellaneous" catch-all.

### `docs/topics.yml` change

Add a `groups:` block above the existing `topics:` block. Each group entry has a
`name`, a one-line `desc`, and an ordered `topics:` list of member slugs. The
existing per-topic `name`/`desc` entries are unchanged and become sub-section
headings. `devtools/topics.py` gains a `load_groups()` reader alongside
`load_topics()`, and validates that every topic belongs to exactly one group.

### `devtools/build_topic_pages.py` change

Today it writes one `docs/by_topic/<topic>.rst` page per topic. It changes to
write one `docs/by_group/<group>.rst` page per group. Each group page has a
top-level title (the group name and description), then one section per member
topic (the topic name and description) holding that topic's function table, in the
group's declared topic order. The generated `by_topic_index.rst` becomes a
`by_group_index.rst` card grid of the 8 groups. The per-function "Topics:"
back-link points at the member topic's section anchor within its group page.

### `docs/index.rst` change

The "Functions" toctree lists the 8 `by_group/<group>` pages in group order,
instead of the 26 `by_topic/<topic>` pages.

## Lever 2: `pydata-sphinx-theme`

Switch `html_theme` from `sphinx_rtd_theme` to `pydata_sphinx_theme` in
`docs/conf.py`, and adjust `html_theme_options` to the pydata schema (navbar
start/center/end, secondary sidebar for the on-this-page TOC, search, light/dark
toggle). Add `pydata-sphinx-theme` and `sphinx-design` to the docs dependency
group in `pyproject.toml` and to `docs/requirements.txt` (or equivalent) so
Read the Docs installs them.

Top-level navbar sections, driven by the top-level toctrees:

- **User Guide** - `usage` plus the concept pages (`polymorphic_api`,
  `nan_and_warmup`, `multistream`, `pipelines`, `microstructure`).
- **Functions** - the 8 group pages.
- **Examples** - the gallery.

The landing page (`index.rst`) keeps its short intro and short example, and gains a
`sphinx-design` card grid linking to the 8 function groups and to the Examples
gallery, so the home page doubles as task-oriented navigation (the scikit-learn
pattern).

## Lever 3: example gallery (card grid over existing notebooks)

Replace the flat "Examples" toctree with a gallery landing page,
`docs/notebooks/index.md` (or `.rst`), built from `sphinx-design` `grid`/`card`
directives. Each card links to an existing notebook with its title and a one-line
description. Cards are grouped under section headings into 5 themes:

- **Getting started** - 01-quickstart, 05-nan-handling
- **Statistics & indicators** - 02-window-statistics, 03-financial-indicators,
  04-signal-processing
- **Streaming & pipelines** - 06-streaming-live-events, 07-multi-stream-operators,
  08-pipelines, 09-bars-from-ticks, 10-custom-and-multi-column-bars
- **Microstructure** - 11-microstructure-order-flow, 12-microstructure-price-impact,
  13-microstructure-toxicity-and-book
- **Backtesting** - 14-backtesting-a-signal, 15-event-driven-backtests

The 15 notebooks stay in a hidden toctree under the gallery page so they still
build and cross-link, but they leave the visible sidebar top level. Cards are
title-and-description to start; a small committed thumbnail image per notebook is a
later enhancement, not part of this change.

## Migration and compatibility

- Old `by_topic/<topic>.html` URLs change to `by_group/<group>.html#<topic>`. If a
  URL-stable redirect matters for the public site, add a `sphinx-reredirects` map
  from each old topic page to its group-page anchor. Otherwise accept that the
  versioned docs move (older tags keep the old URLs).
- `tests/test_doc_coverage.py` currently checks topic-slug validity and that every
  function has at least one topic. It gains a check that every topic belongs to a
  group, and the topic-index assertions point at the group structure.

## Validation

- `poetry run python devtools/build_help_registry.py` still validates every
  function page (unchanged frontmatter).
- `poetry run python devtools/build_topic_pages.py` emits 8 group pages plus the
  group index, with every function reachable.
- `poetry run pytest -q tests/test_doc_coverage.py` passes with the group checks.
- `make docs` builds clean on the new theme: the navbar shows the sections, the
  sidebar is scoped to the current section, the Functions landing shows the 8
  groups, and the Examples page shows the card gallery.
