# Help-registry JSON Structure Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the freeform `body_markdown` field in `screamer/data/help.json` with structured `details` (markdown without code) + `examples` (typed list of `{language, caption, code}`), and restructure the source `docs/functions_*/<Name>.md` files so the split is unambiguous in both JSON and sphinx output.

**Architecture:** Three coordinated pieces — (1) a new pure-function parser inside `devtools/build_help_registry.py` that splits markdown into `details` + `examples` regions, validated by unit tests on synthetic inputs; (2) a one-shot migration script `devtools/migrate_docs_v2.py` (idempotent, deletable after run) that rewrites all 146 source `.md` files into the new canonical layout; (3) consumer-side cleanups (CHANGELOG, build verification). TDD throughout: parser and migration are tested on synthetic markdown strings before being run on real files.

**Tech Stack:** Python 3.11, pytest, PyYAML, pybind11-generated `screamer` bindings, sphinx (myst-parser + plotly directive).

**Spec:** `docs/superpowers/specs/2026-05-20-help-json-structure-refactor-design.md`

---

## File Structure

**Create:**
- `tests/test_build_help_registry.py` — unit tests for the new parser (`parse_help_file_text` + helpers).
- `devtools/migrate_docs_v2.py` — one-shot migration script (deletable after run; removed in the final task).
- `tests/test_migrate_docs_v2.py` — unit tests for the migration script on synthetic before/after markdown pairs.

**Modify:**
- `devtools/build_help_registry.py` — replace `parse_file()` with a pure-function parser; update `main()` to emit `details` + `examples` instead of `body_markdown`; update module docstring.
- `docs/functions_*/*.md` — 101 of 146 files restructured by the migration script. Hand-tuned captions in a follow-up pass.
- `screamer/data/help.json` — regenerated output (binary-ish; commit the regenerated file).
- `CHANGELOG.md` — entry under the next version describing the schema change.

**Single responsibility per new file:**
- `parse_help_file_text(text: str)` is a pure string-in / dict-out function (no file I/O, no side effects). Easy to unit-test, easy to reuse.
- `devtools/migrate_docs_v2.py` only rewrites source `.md` files; it does NOT touch `help.json` (that's `build_help_registry.py`'s job).

---

## Task 1 — Add test scaffolding for the new parser

**Files:**
- Create: `tests/test_build_help_registry.py`

- [ ] **Step 1: Write a smoke test that imports the new parser**

```python
# tests/test_build_help_registry.py
from devtools.build_help_registry import parse_help_file_text


def test_parser_is_importable():
    assert callable(parse_help_file_text)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
poetry run pytest tests/test_build_help_registry.py -v
```

Expected: FAIL with `ImportError: cannot import name 'parse_help_file_text' from 'devtools.build_help_registry'`.

- [ ] **Step 3: Add a placeholder function in `devtools/build_help_registry.py`**

Add this near the existing `parse_file` function (do not remove `parse_file` yet — main() still uses it):

```python
def parse_help_file_text(text: str) -> dict | None:
    """Parse a docs/functions_*/<Name>.md file into a help-registry entry.

    Returns a dict with frontmatter fields plus `details` (markdown string)
    and `examples` (list of {language, caption, code}), or None if the file
    has no frontmatter or no `name` field.

    Raises ValueError if the body violates the canonical layout.
    """
    raise NotImplementedError
```

- [ ] **Step 4: Re-run test to verify it passes**

```bash
poetry run pytest tests/test_build_help_registry.py::test_parser_is_importable -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_build_help_registry.py devtools/build_help_registry.py
git commit -m "test(help-registry): add scaffolding for new parser"
```

---

## Task 2 — Parser: frontmatter + details, no examples case

**Files:**
- Modify: `tests/test_build_help_registry.py`
- Modify: `devtools/build_help_registry.py`

- [ ] **Step 1: Add tests for the minimal happy path**

Append to `tests/test_build_help_registry.py`:

```python
import textwrap


def _md(text: str) -> str:
    """Helper: dedent and trim leading newline for inline markdown strings."""
    return textwrap.dedent(text).lstrip("\n")


def test_minimal_file_no_examples():
    text = _md('''
        ---
        name: Foo
        title: Foo function
        short: A foo.
        inputs: 1
        outputs: 1
        parameters: []
        ---

        # `Foo`

        ## Description

        `Foo` does the foo thing.

        <!-- HELP_END -->

        ## Reference

        Sphinx-only stuff.
    ''')
    entry = parse_help_file_text(text)
    assert entry["name"] == "Foo"
    assert entry["title"] == "Foo function"
    assert entry["short"] == "A foo."
    assert entry["examples"] == []
    assert entry["details"].startswith("## Description")
    assert "`Foo` does the foo thing." in entry["details"]
    assert "Sphinx-only stuff." not in entry["details"]
    assert "body_markdown" not in entry


def test_returns_none_without_frontmatter():
    assert parse_help_file_text("just some markdown\n") is None


def test_returns_none_without_name_field():
    text = _md('''
        ---
        title: Untitled
        ---

        body
    ''')
    assert parse_help_file_text(text) is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/test_build_help_registry.py -v
```

Expected: FAIL — three new tests fail with `NotImplementedError` (or `TypeError` on the `entry["name"]` access).

- [ ] **Step 3: Implement the parser core**

Replace the placeholder in `devtools/build_help_registry.py` with:

```python
import re
import textwrap

import yaml

FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
HELP_END_MARKER = "<!-- HELP_END -->"
H1_RE = re.compile(r"^# .*\n", re.M)
EXAMPLES_H2_RE = re.compile(r"^## Examples\s*\n", re.M)


def parse_help_file_text(text: str) -> dict | None:
    """Parse a docs/functions_*/<Name>.md file into a help-registry entry.

    Returns a dict with frontmatter fields plus `details` (markdown string)
    and `examples` (list of {language, caption, code}), or None if the file
    has no frontmatter or no `name` field.

    Raises ValueError if the body violates the canonical layout.
    """
    m = FRONTMATTER_RE.match(text)
    if not m:
        return None
    fm = yaml.safe_load(m.group(1))
    if not isinstance(fm, dict) or "name" not in fm:
        return None

    body = text[m.end():]
    pre_help_end = body.split(HELP_END_MARKER, 1)[0] if HELP_END_MARKER in body else body

    examples_m = EXAMPLES_H2_RE.search(pre_help_end)
    if examples_m:
        details_part = pre_help_end[:examples_m.start()]
        examples_part = pre_help_end[examples_m.end():]
    else:
        details_part = pre_help_end
        examples_part = ""

    details = H1_RE.sub("", details_part, count=1).strip()
    if "```" in details:
        raise ValueError("code fence outside `## Examples` section")

    examples = _parse_examples_region(examples_part) if examples_part.strip() else []

    entry = dict(fm)
    entry["details"] = details
    entry["examples"] = examples
    return entry


def _parse_examples_region(text: str) -> list[dict]:
    """Stub — implemented in Task 3."""
    return []
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/test_build_help_registry.py -v
```

Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add tests/test_build_help_registry.py devtools/build_help_registry.py
git commit -m "feat(help-registry): parse frontmatter, details, and HELP_END boundary"
```

---

## Task 3 — Parser: examples extraction (plain code fences)

**Files:**
- Modify: `tests/test_build_help_registry.py`
- Modify: `devtools/build_help_registry.py`

- [ ] **Step 1: Add test for single-example case**

Append to `tests/test_build_help_registry.py`:

```python
def test_single_plain_python_example():
    text = _md('''
        ---
        name: Foo
        ---

        # `Foo`

        ## Description

        Prose.

        ## Examples

        ### Basic usage

        ```python
        from screamer import Foo
        Foo()(arr)
        ```

        <!-- HELP_END -->
    ''')
    entry = parse_help_file_text(text)
    assert entry["examples"] == [
        {
            "language": "python",
            "caption": "Basic usage",
            "code": "from screamer import Foo\nFoo()(arr)",
        }
    ]
    assert "## Examples" not in entry["details"]
    assert "```" not in entry["details"]


def test_non_python_language_tag_preserved():
    text = _md('''
        ---
        name: Foo
        ---

        # `Foo`

        ## Examples

        ### Shell snippet

        ```bash
        echo hi
        ```

        <!-- HELP_END -->
    ''')
    entry = parse_help_file_text(text)
    assert entry["examples"][0]["language"] == "bash"
    assert entry["examples"][0]["code"] == "echo hi"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/test_build_help_registry.py -v
```

Expected: FAIL — examples list is empty (stub).

- [ ] **Step 3: Implement `_parse_examples_region`**

In `devtools/build_help_registry.py`, replace the stub with:

```python
H3_SPLIT_RE = re.compile(r"^### (.+)$", re.M)
FENCE_RE = re.compile(r"\A```(\S*)\s*\n(.*?)\n```", re.S)


def _parse_examples_region(text: str) -> list[dict]:
    """Split an Examples region into [{language, caption, code}, ...].

    `text` is the content AFTER the `## Examples` heading and BEFORE
    `<!-- HELP_END -->`. Headings are H3 (`### Caption`). The first
    fenced code block under each H3 becomes that example's `code`.
    """
    parts = H3_SPLIT_RE.split(text)
    # parts = [pre_first_h3, caption1, content1, caption2, content2, ...]
    if parts[0].strip():
        raise ValueError(
            "content under `## Examples` without a preceding `### Heading`"
        )
    examples = []
    for i in range(1, len(parts), 2):
        caption = parts[i].strip()
        content = parts[i + 1].lstrip()
        m = FENCE_RE.match(content)
        if not m:
            raise ValueError(
                f"`### {caption}` is not immediately followed by a code fence"
            )
        language = m.group(1).strip() or "python"
        code = m.group(2)
        # Plotly unwrap is added in Task 4.
        examples.append({"language": language, "caption": caption, "code": code})
    return examples
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/test_build_help_registry.py -v
```

Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add tests/test_build_help_registry.py devtools/build_help_registry.py
git commit -m "feat(help-registry): parse plain ### Caption + fenced code examples"
```

---

## Task 4 — Parser: plotly eval-rst directive unwrapping

**Files:**
- Modify: `tests/test_build_help_registry.py`
- Modify: `devtools/build_help_registry.py`

- [ ] **Step 1: Add test for plotly directive**

Append to `tests/test_build_help_registry.py`:

```python
def test_plotly_eval_rst_is_unwrapped():
    text = _md('''
        ---
        name: Foo
        ---

        # `Foo`

        ## Examples

        ### Plotly demo

        ```{eval-rst}
        .. plotly::
            :include-source: True

            import numpy as np
            from screamer import Foo
            arr = np.arange(5)
            print(Foo()(arr))
        ```

        <!-- HELP_END -->
    ''')
    entry = parse_help_file_text(text)
    assert entry["examples"] == [
        {
            "language": "python",
            "caption": "Plotly demo",
            "code": (
                "import numpy as np\n"
                "from screamer import Foo\n"
                "arr = np.arange(5)\n"
                "print(Foo()(arr))"
            ),
        }
    ]


def test_eval_rst_without_plotly_directive_is_rejected():
    text = _md('''
        ---
        name: Foo
        ---

        # `Foo`

        ## Examples

        ### Other directive

        ```{eval-rst}
        .. note:: not supported here
        ```

        <!-- HELP_END -->
    ''')
    try:
        parse_help_file_text(text)
    except ValueError as e:
        assert "eval-rst" in str(e) or "plotly" in str(e)
    else:
        raise AssertionError("expected ValueError")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/test_build_help_registry.py -v
```

Expected: FAIL — first test produces wrong language/code (eval-rst wrapper unstripped); second test does not raise.

- [ ] **Step 3: Implement the plotly unwrap**

In `devtools/build_help_registry.py`, add the helper and call it from `_parse_examples_region`:

```python
PLOTLY_DIRECTIVE_RE = re.compile(
    r"\A\.\. plotly::\s*\n"        # ".. plotly::" + newline
    r"((?:[ \t]+:[^\n]*\n)*)"      # zero or more ":option: value" lines
    r"\s*\n"                       # blank line
    r"(.+)",                       # indented body (captured)
    re.S,
)


def _unwrap_plotly(eval_rst_body: str) -> str:
    """Extract the indented python body from a `.. plotly::` directive.

    Raises ValueError if the eval-rst body is not a plotly directive.
    """
    m = PLOTLY_DIRECTIVE_RE.match(eval_rst_body)
    if not m:
        raise ValueError(
            "unsupported eval-rst directive in `## Examples` "
            "(only `.. plotly::` is recognized)"
        )
    indented_body = m.group(2)
    return textwrap.dedent(indented_body).strip("\n")
```

And inside `_parse_examples_region`, replace the `# Plotly unwrap is added in Task 4.` comment with:

```python
        if language == "{eval-rst}":
            code = _unwrap_plotly(code)
            language = "python"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/test_build_help_registry.py -v
```

Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add tests/test_build_help_registry.py devtools/build_help_registry.py
git commit -m "feat(help-registry): unwrap {eval-rst} .. plotly:: directives"
```

---

## Task 5 — Parser: multi-example case

**Files:**
- Modify: `tests/test_build_help_registry.py`

- [ ] **Step 1: Add multi-example test**

Append to `tests/test_build_help_registry.py`:

```python
def test_multiple_examples_in_order():
    text = _md('''
        ---
        name: Foo
        ---

        # `Foo`

        ## Examples

        ### First

        ```python
        a = 1
        ```

        ### Second

        ```python
        b = 2
        ```

        ### Third

        ```python
        c = 3
        ```

        <!-- HELP_END -->
    ''')
    entry = parse_help_file_text(text)
    captions = [e["caption"] for e in entry["examples"]]
    codes = [e["code"] for e in entry["examples"]]
    assert captions == ["First", "Second", "Third"]
    assert codes == ["a = 1", "b = 2", "c = 3"]
```

- [ ] **Step 2: Run test to verify it passes (no implementation needed)**

```bash
poetry run pytest tests/test_build_help_registry.py::test_multiple_examples_in_order -v
```

Expected: PASS. The H3-splitter built in Task 3 already handles N examples.

- [ ] **Step 3: Commit**

```bash
git add tests/test_build_help_registry.py
git commit -m "test(help-registry): cover multi-example case"
```

---

## Task 6 — Parser: validation — code fence outside Examples

**Files:**
- Modify: `tests/test_build_help_registry.py`

- [ ] **Step 1: Add test**

Append to `tests/test_build_help_registry.py`:

```python
def test_code_fence_in_details_is_rejected():
    text = _md('''
        ---
        name: Foo
        ---

        # `Foo`

        ## Description

        Inline example before Examples section:

        ```python
        oops()
        ```

        <!-- HELP_END -->
    ''')
    try:
        parse_help_file_text(text)
    except ValueError as e:
        assert "Examples" in str(e)
    else:
        raise AssertionError("expected ValueError")
```

- [ ] **Step 2: Run test to verify it passes (rule already implemented in Task 2)**

```bash
poetry run pytest tests/test_build_help_registry.py::test_code_fence_in_details_is_rejected -v
```

Expected: PASS. Task 2's parser already checks `"```" in details`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_build_help_registry.py
git commit -m "test(help-registry): reject code fence outside ## Examples"
```

---

## Task 7 — Parser: validation — H3 without code

**Files:**
- Modify: `tests/test_build_help_registry.py`

- [ ] **Step 1: Add test**

Append to `tests/test_build_help_registry.py`:

```python
def test_h3_without_code_fence_is_rejected():
    text = _md('''
        ---
        name: Foo
        ---

        # `Foo`

        ## Examples

        ### Caption with no code

        Some prose but no fence.

        <!-- HELP_END -->
    ''')
    try:
        parse_help_file_text(text)
    except ValueError as e:
        assert "code fence" in str(e)
        assert "Caption with no code" in str(e)
    else:
        raise AssertionError("expected ValueError")
```

- [ ] **Step 2: Run test to verify it passes (rule already implemented in Task 3)**

```bash
poetry run pytest tests/test_build_help_registry.py::test_h3_without_code_fence_is_rejected -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_build_help_registry.py
git commit -m "test(help-registry): reject ### heading with no code fence"
```

---

## Task 8 — Parser: validation — code without H3

**Files:**
- Modify: `tests/test_build_help_registry.py`

- [ ] **Step 1: Add test**

Append to `tests/test_build_help_registry.py`:

```python
def test_code_in_examples_without_h3_is_rejected():
    text = _md('''
        ---
        name: Foo
        ---

        # `Foo`

        ## Examples

        ```python
        floating_code()
        ```

        <!-- HELP_END -->
    ''')
    try:
        parse_help_file_text(text)
    except ValueError as e:
        assert "### Heading" in str(e)
    else:
        raise AssertionError("expected ValueError")
```

- [ ] **Step 2: Run test to verify it passes (rule already implemented in Task 3)**

```bash
poetry run pytest tests/test_build_help_registry.py::test_code_in_examples_without_h3_is_rejected -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_build_help_registry.py
git commit -m "test(help-registry): reject code under ## Examples without ### heading"
```

---

## Task 9 — Replace `parse_file` in `main()`; remove `body_markdown`

**Files:**
- Modify: `devtools/build_help_registry.py`

- [ ] **Step 1: Update module docstring**

Replace the docstring in `devtools/build_help_registry.py` (lines 2-27) with:

```python
"""Build a JSON help registry consumable by external frontends.

For every `docs/functions_*/<Name>.md` file that begins with a YAML
frontmatter block, this script:

1. Parses the frontmatter (structured schema: title, category, tags,
   parameter list with defaults / types / constraints, IO arity).
2. Parses the markdown body into two structured fields:
   - ``details``: prose / math / sub-sections, guaranteed no code fences.
   - ``examples``: a list of ``{language, caption, code}`` objects, one
     per ``### Caption`` under the ``## Examples`` heading. ``{eval-rst}
     .. plotly::`` directives are unwrapped to plain python.
   Anything after the ``<!-- HELP_END -->`` marker (sphinx-only sections
   like ``Implementation Details``, references) is excluded.
3. Validates the schema by instantiating the class with every parameter
   set to its documented default and calling it on a small synthetic
   array. If that round-trip fails, the schema and the binding have
   drifted and the build aborts.
4. Cross-checks the documented parameter list against the actual
   ``__init__`` signature parsed out of the pybind11 docstring, so a
   missing parameter in the frontmatter is caught even if defaults make
   the live call succeed.

Output: ``screamer/data/help.json`` — a dict keyed by class name. Each
entry contains the frontmatter fields plus ``details`` and ``examples``.

Run:
    poetry run python devtools/build_help_registry.py
"""
```

- [ ] **Step 2: Delete the old `parse_file` function**

Remove the old `parse_file(path: Path)` function (the version that returns `(fm, body)` and uses `HELP_END_MARKER` for splitting). The new `parse_help_file_text` replaces it entirely.

- [ ] **Step 3: Update `main()` to use the new parser**

Replace the body of `main(argv=None)` with:

```python
def main(argv=None):
    sys.path.insert(0, str(ROOT))
    screamer = importlib.import_module("screamer")

    files = sorted(DOCS.glob("functions_*/*.md"))
    if argv:
        wanted = set(argv)
        files = [p for p in files if p.stem in wanted]

    registry: dict[str, dict] = {}
    for md in files:
        try:
            entry = parse_help_file_text(md.read_text())
        except ValueError as e:
            raise SystemExit(f"{md.relative_to(ROOT)}: {e}") from None
        if entry is None:
            continue
        validate(entry, screamer)
        registry[entry["name"]] = entry
        n_ex = len(entry["examples"])
        print(f"  + {entry['name']:24s}  ({md.relative_to(ROOT)}, {n_ex} examples)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n")
    print(f"\nWrote {OUT.relative_to(ROOT)} with {len(registry)} entries")
```

The `validate(entry, screamer)` function from the existing code still operates on the entry dict and only reads `name`, `parameters`, and `inputs` — no changes needed to it.

- [ ] **Step 4: Do NOT run the build script yet**

The source `.md` files are still in the old format. Running the build now would fail on every file with embedded code or post-HELP_END Usage Example sections. Migration script comes next.

- [ ] **Step 5: Verify the parser tests still pass**

```bash
poetry run pytest tests/test_build_help_registry.py -v
```

Expected: PASS (all 11 tests from Tasks 1–8).

- [ ] **Step 6: Commit**

```bash
git add devtools/build_help_registry.py
git commit -m "refactor(help-registry): emit details + examples, drop body_markdown"
```

---

## Task 10 — Migration script scaffolding + idempotency

**Files:**
- Create: `devtools/migrate_docs_v2.py`
- Create: `tests/test_migrate_docs_v2.py`

- [ ] **Step 1: Write tests for the idempotent no-op cases**

```python
# tests/test_migrate_docs_v2.py
import textwrap

from devtools.migrate_docs_v2 import migrate_text


def _md(text: str) -> str:
    return textwrap.dedent(text).lstrip("\n")


def test_already_migrated_is_noop():
    """A file that already has `## Examples` above HELP_END and no other code is unchanged."""
    text = _md('''
        ---
        name: Foo
        ---

        # `Foo`

        ## Description

        Prose only.

        ## Examples

        ### Basic

        ```python
        Foo()
        ```

        <!-- HELP_END -->

        ## Reference

        Refs.
    ''')
    assert migrate_text(text) == text


def test_no_code_anywhere_is_noop():
    """A file with no code at all is unchanged."""
    text = _md('''
        ---
        name: Foo
        ---

        # `Foo`

        ## Description

        Prose only.

        <!-- HELP_END -->

        ## Reference

        Refs.
    ''')
    assert migrate_text(text) == text
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/test_migrate_docs_v2.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'devtools.migrate_docs_v2'`.

- [ ] **Step 3: Implement the migration scaffold**

```python
# devtools/migrate_docs_v2.py
#!/usr/bin/env python3
"""One-shot migration: restructure docs/functions_*/<Name>.md to the new layout.

Restructure rules (see docs/superpowers/specs/2026-05-20-help-json-structure-refactor-design.md):

- Post-HELP_END `## Usage Example*` section: move above HELP_END,
  demote to a `## Examples` parent with a `### Usage example` child.
- Pre-HELP_END code fences embedded in prose: extract, append under a
  `## Examples` section just before HELP_END, caption = enclosing H2 heading.
- If both shapes are present, combine into a single `## Examples` section.

The script is idempotent: running it twice on a converted file is a no-op.

After running, hand-tune captions in the flagged files (see run log).

Usage:
    poetry run python devtools/migrate_docs_v2.py                # all 146 files
    poetry run python devtools/migrate_docs_v2.py path/to/X.md   # specific files
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
HELP_END_MARKER = "<!-- HELP_END -->"


def migrate_text(text: str) -> str:
    """Return the migrated markdown. Idempotent."""
    # Implementation grows in Tasks 11–13. For now, return text unchanged
    # whenever the file is already in the target shape or has no code at all.
    pre, _, post = text.partition(HELP_END_MARKER)

    # Already migrated: has `## Examples` above HELP_END.
    has_examples_section = bool(re.search(r"^## Examples\s*$", pre, re.M))
    # No code anywhere.
    has_any_fence = "```" in text

    if has_examples_section or not has_any_fence:
        # Still might have a stray post-HELP_END `## Usage Example*`; that case
        # is handled in Task 11. For now, no-op.
        return text

    return text  # Other cases get real logic in Tasks 11–13.


def main(argv: list[str]) -> int:
    if argv:
        paths = [Path(p) for p in argv]
    else:
        paths = sorted(DOCS.glob("functions_*/*.md"))
    changed = 0
    for path in paths:
        original = path.read_text()
        migrated = migrate_text(original)
        if migrated != original:
            path.write_text(migrated)
            changed += 1
            print(f"  migrated: {path.relative_to(ROOT)}")
        else:
            print(f"    no-op:  {path.relative_to(ROOT)}")
    print(f"\n{changed} file(s) migrated, {len(paths) - changed} no-op")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/test_migrate_docs_v2.py -v
```

Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add devtools/migrate_docs_v2.py tests/test_migrate_docs_v2.py
git commit -m "feat(migrate-docs): scaffold idempotent one-shot migration"
```

---

## Task 11 — Migration: post-HELP_END Usage Example section move

**Files:**
- Modify: `tests/test_migrate_docs_v2.py`
- Modify: `devtools/migrate_docs_v2.py`

- [ ] **Step 1: Add test**

Append to `tests/test_migrate_docs_v2.py`:

```python
def test_post_help_end_usage_example_is_moved():
    """The 85-file case: only a `## Usage Example*` section sits below HELP_END."""
    before = _md('''
        ---
        name: Foo
        ---

        # `Foo`

        ## Description

        Prose.

        <!-- HELP_END -->

        ## Usage Example and Plot

        ```{eval-rst}
        .. plotly::
            :include-source: True

            from screamer import Foo
            print(Foo()(arr))
        ```

        ## Reference

        Refs.
    ''')
    after = _md('''
        ---
        name: Foo
        ---

        # `Foo`

        ## Description

        Prose.

        ## Examples

        ### Usage example

        ```{eval-rst}
        .. plotly::
            :include-source: True

            from screamer import Foo
            print(Foo()(arr))
        ```

        <!-- HELP_END -->

        ## Reference

        Refs.
    ''')
    assert migrate_text(before) == after
    # Idempotency: applying migration to the result is a no-op.
    assert migrate_text(after) == after
```

- [ ] **Step 2: Run test to verify it fails**

```bash
poetry run pytest tests/test_migrate_docs_v2.py::test_post_help_end_usage_example_is_moved -v
```

Expected: FAIL.

- [ ] **Step 3: Implement the post-HELP_END move**

Replace the body of `migrate_text` in `devtools/migrate_docs_v2.py` with:

```python
USAGE_H2_RE = re.compile(r"^## Usage[^\n]*\n", re.M)


def migrate_text(text: str) -> str:
    """Return the migrated markdown. Idempotent."""
    if HELP_END_MARKER not in text:
        return text

    pre, _, post = text.partition(HELP_END_MARKER)

    # Already migrated: `## Examples` above HELP_END.
    has_examples_section = bool(re.search(r"^## Examples\s*$", pre, re.M))
    if has_examples_section:
        return text

    # Locate optional post-HELP_END `## Usage Example*` section.
    usage_block, post_after_extract = _extract_usage_block(post)

    # Pre-HELP_END embedded code: handled in Task 12.
    pre_examples: list[tuple[str, str]] = []  # list of (caption, fenced-block)

    if usage_block is not None:
        pre_examples.append(("Usage example", usage_block))

    if not pre_examples:
        return text  # nothing to do

    new_pre = pre.rstrip() + "\n\n## Examples\n\n"
    for caption, fenced in pre_examples:
        new_pre += f"### {caption}\n\n{fenced.strip()}\n\n"

    return new_pre + HELP_END_MARKER + post_after_extract


def _extract_usage_block(post: str) -> tuple[str | None, str]:
    """Find the first `## Usage*` section in `post` and return its inner code fence.

    Returns (fenced_block_or_None, post_with_section_removed).
    The "inner code fence" includes the triple-backtick lines so it can be
    inserted under a `### caption` heading verbatim.
    """
    m = USAGE_H2_RE.search(post)
    if not m:
        return None, post
    start = m.start()
    # Find the end of this section: next H2 or end of string.
    next_h2 = re.search(r"^## ", post[m.end():], re.M)
    end = m.end() + next_h2.start() if next_h2 else len(post)
    section_body = post[m.end():end]
    # The body should contain exactly one fenced code block. Extract it verbatim.
    fence_m = re.search(r"```[^\n]*\n.*?\n```", section_body, re.S)
    if not fence_m:
        # No code in the section; treat as nothing to migrate.
        return None, post
    fenced = fence_m.group(0)
    new_post = post[:start] + post[end:]
    return fenced, new_post
```

- [ ] **Step 4: Run all migration tests**

```bash
poetry run pytest tests/test_migrate_docs_v2.py -v
```

Expected: PASS (3 tests, including idempotency on the already-migrated input).

- [ ] **Step 5: Commit**

```bash
git add devtools/migrate_docs_v2.py tests/test_migrate_docs_v2.py
git commit -m "feat(migrate-docs): move post-HELP_END Usage Example sections"
```

---

## Task 12 — Migration: pre-HELP_END embedded code extraction

**Files:**
- Modify: `tests/test_migrate_docs_v2.py`
- Modify: `devtools/migrate_docs_v2.py`

- [ ] **Step 1: Add test**

Append to `tests/test_migrate_docs_v2.py`:

```python
def test_pre_help_end_embedded_code_is_extracted():
    """The 13-file case: code fences live inside a pre-HELP_END section."""
    before = _md('''
        ---
        name: Foo
        ---

        # `Foo`

        ## Description

        Prose.

        ## Identity check

        Foo equals Bar:

        ```python
        from screamer import Foo
        Foo()(arr)
        ```

        <!-- HELP_END -->

        ## Reference

        Refs.
    ''')
    after = _md('''
        ---
        name: Foo
        ---

        # `Foo`

        ## Description

        Prose.

        ## Identity check

        Foo equals Bar:

        ## Examples

        ### Identity check

        ```python
        from screamer import Foo
        Foo()(arr)
        ```

        <!-- HELP_END -->

        ## Reference

        Refs.
    ''')
    assert migrate_text(before) == after
    assert migrate_text(after) == after  # idempotent
```

NOTE: the migration intentionally leaves the dangling "Foo equals Bar:" line in place — the spec calls these out for manual review.

- [ ] **Step 2: Run test to verify it fails**

```bash
poetry run pytest tests/test_migrate_docs_v2.py::test_pre_help_end_embedded_code_is_extracted -v
```

Expected: FAIL.

- [ ] **Step 3: Implement pre-HELP_END extraction**

In `devtools/migrate_docs_v2.py`, add a helper and call it from `migrate_text`:

```python
PRE_HELP_FENCE_RE = re.compile(r"```[^\n]*\n.*?\n```", re.S)


def _extract_embedded_fences(pre: str) -> tuple[list[tuple[str, str]], str]:
    """Find all fenced blocks in the pre-HELP_END region and remove them in place.

    Returns (list_of_(caption, fenced_block), pre_with_fences_removed).
    Caption = the enclosing `## H2` heading text, or "Usage example" if none.
    """
    extracted: list[tuple[str, str]] = []

    def _caption_for(match_start: int) -> str:
        # Walk backwards to find the nearest preceding `## H2` heading.
        h2_iter = list(re.finditer(r"^## (.+)$", pre[:match_start], re.M))
        if h2_iter:
            return h2_iter[-1].group(1).strip()
        return "Usage example"

    pieces: list[str] = []
    cursor = 0
    for m in PRE_HELP_FENCE_RE.finditer(pre):
        pieces.append(pre[cursor:m.start()])
        extracted.append((_caption_for(m.start()), m.group(0)))
        cursor = m.end()
        # Also swallow a trailing newline if present so we don't leave a blank line.
        if cursor < len(pre) and pre[cursor] == "\n":
            cursor += 1
    pieces.append(pre[cursor:])

    return extracted, "".join(pieces)
```

Update `migrate_text` to call it (full replacement of the function):

```python
def migrate_text(text: str) -> str:
    """Return the migrated markdown. Idempotent."""
    if HELP_END_MARKER not in text:
        return text

    pre, _, post = text.partition(HELP_END_MARKER)

    if re.search(r"^## Examples\s*$", pre, re.M):
        return text  # already migrated

    embedded, pre = _extract_embedded_fences(pre)
    usage_block, post = _extract_usage_block(post)

    pre_examples: list[tuple[str, str]] = list(embedded)
    if usage_block is not None:
        pre_examples.append(("Usage example", usage_block))

    if not pre_examples:
        return text

    new_pre = pre.rstrip() + "\n\n## Examples\n\n"
    for caption, fenced in pre_examples:
        new_pre += f"### {caption}\n\n{fenced.strip()}\n\n"

    return new_pre + HELP_END_MARKER + post
```

- [ ] **Step 4: Run all migration tests**

```bash
poetry run pytest tests/test_migrate_docs_v2.py -v
```

Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add devtools/migrate_docs_v2.py tests/test_migrate_docs_v2.py
git commit -m "feat(migrate-docs): extract pre-HELP_END embedded code fences"
```

---

## Task 13 — Migration: combined case (3-file class)

**Files:**
- Modify: `tests/test_migrate_docs_v2.py`

- [ ] **Step 1: Add test for combined case**

Append to `tests/test_migrate_docs_v2.py`:

```python
def test_combined_pre_and_post_help_end_code_merge_into_one_examples_section():
    """The 3-file case (EwCorr, MACD, WilliamsR): both shapes present."""
    before = _md('''
        ---
        name: Foo
        ---

        # `Foo`

        ## Identity check

        Foo equals Bar:

        ```python
        identity_demo()
        ```

        <!-- HELP_END -->

        ## Usage Example and Plot

        ```{eval-rst}
        .. plotly::

            plotly_demo()
        ```
    ''')
    after = _md('''
        ---
        name: Foo
        ---

        # `Foo`

        ## Identity check

        Foo equals Bar:

        ## Examples

        ### Identity check

        ```python
        identity_demo()
        ```

        ### Usage example

        ```{eval-rst}
        .. plotly::

            plotly_demo()
        ```

        <!-- HELP_END -->
    ''')
    assert migrate_text(before) == after
    assert migrate_text(after) == after
```

- [ ] **Step 2: Run test (no new implementation needed)**

```bash
poetry run pytest tests/test_migrate_docs_v2.py::test_combined_pre_and_post_help_end_code_merge_into_one_examples_section -v
```

Expected: PASS — Task 12's `migrate_text` already concatenates `embedded + usage_block`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_migrate_docs_v2.py
git commit -m "test(migrate-docs): cover combined pre+post HELP_END case"
```

---

## Task 14 — Run migration on all 146 source `.md` files

**Files:**
- Modify: `docs/functions_*/*.md` (101 files rewritten by the script)

- [ ] **Step 1: Capture the pre-migration state for diffing**

```bash
git status   # confirm working tree clean
git rev-parse HEAD > /tmp/pre-migration-sha.txt
```

- [ ] **Step 2: Run the migration**

```bash
poetry run python devtools/migrate_docs_v2.py 2>&1 | tee /tmp/migration.log
```

Expected output: ≈101 lines beginning `  migrated:`, ≈45 `    no-op:`, then `101 file(s) migrated, 45 no-op`.

- [ ] **Step 3: Sanity-check the diff**

```bash
git diff --stat docs/functions_*/ | tail -5
```

Expected: ~101 files changed, with mostly small line-count deltas (sections moved, not rewritten).

Spot-check the three "combined" cases by eye:

```bash
git diff docs/functions_ew/EwCorr.md | head -80
git diff docs/functions_signal/MACD.md | head -80
git diff docs/functions_misc/WilliamsR.md | head -80
```

(WilliamsR is in a different family; adjust path if needed: `grep -l "name: WilliamsR" docs/functions_*/*.md`.)

- [ ] **Step 4: Commit the migrated source**

```bash
git add docs/functions_*/
git commit -m "docs: migrate function .md files to ## Examples / ### Caption layout"
```

---

## Task 15 — Regenerate `help.json` and verify schema

**Files:**
- Modify: `screamer/data/help.json`

- [ ] **Step 1: Run the build script against migrated source**

```bash
poetry run python devtools/build_help_registry.py 2>&1 | tail -20
```

Expected: `Wrote screamer/data/help.json with 146 entries`. No traceback. If any file fails, the error message names the file and the violation — fix that file by hand (or fix the migration script and re-run Task 14) and repeat.

- [ ] **Step 2: Add a smoke test that the regenerated JSON is well-formed**

Append to `tests/test_build_help_registry.py`:

```python
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HELP_JSON = REPO_ROOT / "screamer" / "data" / "help.json"


def test_help_json_has_new_schema():
    """End-to-end: every entry has details + examples, no body_markdown."""
    data = json.loads(HELP_JSON.read_text())
    assert len(data) >= 146
    for name, entry in data.items():
        assert "body_markdown" not in entry, f"{name} still has body_markdown"
        assert isinstance(entry.get("details"), str), f"{name} missing details"
        assert "```" not in entry["details"], f"{name} has code fence in details"
        assert isinstance(entry.get("examples"), list), f"{name} missing examples"
        for i, ex in enumerate(entry["examples"]):
            assert set(ex.keys()) == {"language", "caption", "code"}, (
                f"{name}.examples[{i}] has unexpected keys: {ex.keys()}"
            )
            assert ex["language"], f"{name}.examples[{i}] has empty language"
            assert ex["caption"], f"{name}.examples[{i}] has empty caption"
            assert ex["code"], f"{name}.examples[{i}] has empty code"
```

- [ ] **Step 3: Run the schema test**

```bash
poetry run pytest tests/test_build_help_registry.py::test_help_json_has_new_schema -v
```

Expected: PASS.

- [ ] **Step 4: Confirm no other consumer reads `body_markdown`**

```bash
grep -rn body_markdown . --include="*.py" --include="*.md" --include="*.rst" \
  | grep -v build/ | grep -v docs/superpowers/specs | grep -v devtools/build_help_registry.py
```

Expected: no output. (Confirmed at plan-writing time: only `devtools/build_help_registry.py` referenced the field, and Task 9 already updated it. If this command surfaces anything new, update that consumer before continuing.)

- [ ] **Step 5: Commit regenerated JSON and the schema test**

```bash
git add screamer/data/help.json tests/test_build_help_registry.py
git commit -m "build(help-registry): regenerate help.json with new schema"
```

---

## Task 16 — Build sphinx docs and spot-check rendering

**Files:** none modified (verification only)

- [ ] **Step 1: Build the docs**

```bash
cd docs && poetry run make html 2>&1 | tail -30
```

Expected: clean build (warnings unrelated to this refactor are tolerable; new errors are not).

- [ ] **Step 2: Spot-check sample pages across families**

Open each of the following in a browser (or `open` on macOS) and confirm:

```bash
open docs/_build/html/functions_ew/EwCorr.html
open docs/_build/html/functions_fin/Return.html
open docs/_build/html/functions_math/Abs.html
open docs/_build/html/functions_rolling/ATR.html
open docs/_build/html/functions_signal/MACD.html
```

Visual checks per page:
- An `Examples` heading appears (where the function had examples).
- Sub-captions appear as smaller headings (H3).
- Plotly figures still render where they did before (interactive plot widget present).
- No raw triple-backtick / `eval-rst` text leaking into the rendered HTML.
- The `Reference` / `Implementation Details` sections (post-HELP_END) still appear at the bottom.

If a page looks broken, note the file and either fix the source `.md` by hand or improve the migration script and re-run Task 14.

- [ ] **Step 3: No commit needed unless source files were hand-edited.**

If any `.md` was hand-edited during this task:

```bash
git add docs/functions_*/
git commit -m "docs: hand-touch <file> after sphinx spot-check"
```

---

## Task 17 — Version bump + CHANGELOG entry

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `pyproject.toml` (version field)
- Possibly modify: any other version-defining file in the repo (e.g. `screamer/__init__.py` if a `__version__` is defined there)

- [ ] **Step 1: Inspect current version locations**

```bash
grep -nE '^version *=|__version__' pyproject.toml screamer/__init__.py 2>/dev/null
head -40 CHANGELOG.md
```

Expected current version: `0.3.0` (per last bump in commit `3276f19`). This refactor is a **breaking change to the JSON consumer contract** (removed `body_markdown` field), so bump minor: `0.3.0` → `0.4.0`.

- [ ] **Step 2: Bump the version in `pyproject.toml`**

```bash
poetry version 0.4.0
```

If a separate `__version__` string exists in `screamer/__init__.py`, edit it to match. (The codebase uses `poetry version` historically; do not introduce a new versioning system here.)

- [ ] **Step 3: Add a CHANGELOG entry under the new version heading**

Add (or insert under the current unreleased heading):

```markdown
### Changed (breaking, JSON consumers)

- `screamer/data/help.json` schema: the freeform `body_markdown` field is
  removed and replaced with two structured fields:
  - `details` (string) — markdown prose, guaranteed to contain no fenced
    code blocks. Use this when rendering the description / math / notes.
  - `examples` (list of `{language, caption, code}`) — extracted code
    examples, one entry per `### Caption` heading in the source markdown.
    `{eval-rst} .. plotly::` directives are unwrapped to plain python.

  Consumers that read `body_markdown` must switch to `details` (and
  optionally render `examples` separately). No backwards-compatibility
  shim is provided.

### Changed

- Function reference docs (`docs/functions_*/<Name>.md`) now follow a
  canonical layout: prose lives under H2 sub-headings (Description,
  Formula, …), examples live under a single `## Examples` H2 with one
  `### Caption` per example. The sphinx-rendered pages adopt the same
  structure.
```

- [ ] **Step 4: Commit version bump + changelog together**

```bash
git add CHANGELOG.md pyproject.toml screamer/__init__.py 2>/dev/null
git commit -m "chore(release): bump to 0.4.0 for help.json schema break"
```

(The `2>/dev/null` swallows the warning if `screamer/__init__.py` was not edited.)

---

## Task 18 — Remove the migration script

**Files:**
- Delete: `devtools/migrate_docs_v2.py`
- Delete: `tests/test_migrate_docs_v2.py`

- [ ] **Step 1: Confirm migration is complete and committed**

```bash
git log --oneline | head -10
git status   # should be clean
```

- [ ] **Step 2: Delete the migration script and its tests**

```bash
git rm devtools/migrate_docs_v2.py tests/test_migrate_docs_v2.py
```

- [ ] **Step 3: Run the remaining test suite**

```bash
poetry run pytest tests/test_build_help_registry.py -v
```

Expected: PASS (all parser tests + `test_help_json_has_new_schema`).

- [ ] **Step 4: Commit**

```bash
git commit -m "chore: remove one-shot migrate_docs_v2 after successful run"
```

---

## End-of-plan verification

After Task 18, run the full project test suite to confirm nothing else regressed:

```bash
poetry run pytest -q
```

If any unrelated test fails, investigate before considering the refactor done.

---

## Hand-tuning follow-up (out of scope for this plan)

Per the spec, the migration produces starter captions like `"Usage example"` and `"Identity check"`. After this plan completes, do a manual pass across the 101 migrated files to give each example a more descriptive caption that actually says what the demo shows. That is a routine docs follow-up and does not require code changes or schema work — it is intentionally not tracked here.
