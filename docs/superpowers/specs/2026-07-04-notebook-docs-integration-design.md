# Notebook → Sphinx docs integration (myst-nb)

**Status:** design approved 2026-07-04
**Scope:** Render the 10 `docs/notebooks/*.ipynb` demo notebooks inside the
generated Sphinx docs so they become part of the published documentation.
Repo-side only — the ReadTheDocs hosting config (`.readthedocs.yaml`) is
deferred to the later devops/RTD pass.

## Background

The docs are Sphinx (`docs/conf.py`, `sphinx_rtd_theme`), built via
`make docs` → `docs/_build/html`, published on ReadTheDocs. Both `conf.py`
and `pyproject.toml` already carry notes: nbsphinx was configured then removed
because it was abandoned (called `docutils.utils.error_reporting`, deleted in
docutils 0.21) and no notebooks existed yet; the notes say to re-add notebook
support with **myst-nb** (the maintained, MyST-native successor) once notebooks
exist. They now exist (notebooks 01–10, all green under `pytest --nbmake`).

The committed `.ipynb` files carry **no stored outputs** (clean diffs; the
nbmake gate executes but does not save outputs).

## Design

**Extension — myst-nb, execute at build.**
- Add `myst-nb` to the docs dependencies (`[project.optional-dependencies].docs`
  and `[tool.poetry.group.docs.dependencies]`), and `ipykernel` to the poetry
  docs group (myst-nb executes notebooks through a Jupyter kernel).
- In `conf.py`, replace `myst_parser` in `extensions` with `myst_nb`. myst-nb
  loads myst-parser itself; listing both raises a conflict. All existing
  `myst_enable_extensions` are preserved (myst-nb honours them).
- Register the notebook suffix: `source_suffix['.ipynb'] = 'myst-nb'`.
- Because the notebooks have no stored outputs, set
  `nb_execution_mode = "auto"` — myst-nb executes any notebook lacking outputs
  during the Sphinx build and captures fresh outputs (plots, printed results).
  This is the same build-time-execution model the existing `sphinx-exec-code`
  and `sphinx-plotly-directive` extensions already use. Notebooks are seeded and
  deterministic, so output is stable across rebuilds and RTD builds. Set a
  generous `nb_execution_timeout` and `nb_execution_raise_on_error = True` so a
  broken notebook fails the docs build loudly (docs stay honest, like nbmake).

**Navigation.** A new `Examples` toctree section in `index.rst`, placed right
after the *User Guide* section (notebooks are the best on-ramp), listing
`notebooks/01-…` through `notebooks/10-…` in order.

**Deferred (later devops pass):** `.readthedocs.yaml`. RTD currently resolves
docs deps via its web UI; ensuring the hosted build installs myst-nb + ipykernel
belongs to the "connect to ReadTheDocs / devops" phase.

## Testing / acceptance

- `make docs` builds clean: no new warnings beyond the two known cosmetic
  `sphinx-exec-code` `src/` scan warnings already accepted in `conf.py`.
- All 10 notebooks appear in the built HTML under an *Examples* nav section,
  each rendered with its executed outputs (plots + printed values).
- A deliberately-broken notebook would fail the build (guards against silently
  shipping a notebook that no longer runs), mirroring the nbmake gate.
- `make notebooks` (the existing nbmake gate) still passes unchanged.
