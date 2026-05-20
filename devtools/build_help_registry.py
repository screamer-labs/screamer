#!/usr/bin/env python3
"""Build a JSON help registry consumable by external frontends.

For every `docs/functions_*/<Name>.md` file that begins with a YAML
frontmatter block, this script:

1. Parses the frontmatter (structured schema: title, category, tags,
   parameter list with defaults / types / constraints, IO arity).
2. Extracts the markdown body up to a `<!-- HELP_END -->` marker. Anything
   after that marker (sphinx-only sections like ``Implementation Details``,
   plotly examples, references) is excluded from the body the frontend
   renders. Math and prose stay.
3. Validates the schema by instantiating the class with every parameter
   set to its documented default and calling it on a small synthetic
   array. If that round-trip fails, the schema and the binding have
   drifted and the build aborts.
4. Cross-checks the documented parameter list against the actual
   ``__init__`` signature parsed out of the pybind11 docstring, so a
   missing parameter in the frontmatter is caught even if defaults make
   the live call succeed.

Output: ``screamer/data/help.json`` — a dict keyed by class name. Each
entry contains the frontmatter fields plus a ``body_markdown`` string.

Run:
    poetry run python devtools/build_help_registry.py
"""
import json
import re
import sys
import textwrap
import importlib
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
OUT = ROOT / "screamer" / "data" / "help.json"

FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
HELP_END_MARKER = "<!-- HELP_END -->"
H1_RE = re.compile(r"^# .*\n", re.M)
EXAMPLES_H2_RE = re.compile(r"^## Examples\s*\n", re.M)
H3_SPLIT_RE = re.compile(r"^### (.+)$", re.M)
FENCE_RE = re.compile(r"\A```(\S*)\s*\n(.*?)\n```", re.S)
# pybind11 emits ``__init__(self: <type>, foo: int, bar: str = 'x') -> None``
# as the first line of the docstring. Capture the parameter names.
PYBIND_SIG_RE = re.compile(r"__init__\(self[^,)]*(?:,\s*([^)]*))?\)")


def parse_file(path: Path):
    text = path.read_text()
    m = FRONTMATTER_RE.match(text)
    if not m:
        return None
    fm = yaml.safe_load(m.group(1))
    if not isinstance(fm, dict) or "name" not in fm:
        return None
    body = text[m.end():]
    if HELP_END_MARKER in body:
        body = body.split(HELP_END_MARKER, 1)[0]
    return fm, body.strip()


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


def pybind_param_names(cls) -> list[str]:
    """Extract parameter names from a pybind11 class' __init__ docstring.

    Naive comma-splitting breaks on container default values like
    ``taps: list[float] = [0.25, 0.5, 0.25]``. We split only at commas
    that are at bracket-depth zero.
    """
    doc = (cls.__init__.__doc__ or "").splitlines()[0]
    m = PYBIND_SIG_RE.search(doc)
    if not m or not m.group(1):
        return []
    raw = m.group(1)
    chunks, buf, depth = [], "", 0
    for ch in raw:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            chunks.append(buf)
            buf = ""
        else:
            buf += ch
    if buf:
        chunks.append(buf)
    return [c.strip().split(":", 1)[0].strip() for c in chunks if c.strip()]


def validate(entry: dict, screamer_module) -> None:
    name = entry["name"]
    cls = getattr(screamer_module, name, None)
    if cls is None:
        raise RuntimeError(f"screamer has no class named {name!r}")

    schema_params = entry.get("parameters", []) or []
    kwargs = {}
    for p in schema_params:
        if "default" not in p:
            raise ValueError(
                f"{name}.{p['name']}: every parameter must have a 'default' "
                "so the frontend can render live results without user input"
            )
        kwargs[p["name"]] = p["default"]

    # Schema-vs-binding parameter-name drift check.
    binding_names = pybind_param_names(cls)
    schema_names = [p["name"] for p in schema_params]
    extra_in_schema = set(schema_names) - set(binding_names)
    missing_in_schema = set(binding_names) - set(schema_names)
    if extra_in_schema:
        raise ValueError(
            f"{name}: schema lists parameters not in the binding: {sorted(extra_in_schema)}"
        )
    if missing_in_schema:
        raise ValueError(
            f"{name}: binding has parameters not in the schema: {sorted(missing_in_schema)}"
        )

    # Live round-trip: build with defaults, call on a small array.
    instance = cls(**kwargs)
    n_in = int(entry.get("inputs", 1))
    arr = np.random.default_rng(0).standard_normal(1024)
    if n_in == 1:
        out = instance(arr)
    else:
        out = instance(*(arr.copy() for _ in range(n_in)))
    if out is None:
        raise RuntimeError(f"{name}: live call with defaults returned None")


def main(argv=None):
    sys.path.insert(0, str(ROOT))
    screamer = importlib.import_module("screamer")

    files = sorted(DOCS.glob("functions_*/*.md"))
    if argv:
        wanted = set(argv)
        files = [p for p in files if p.stem in wanted]

    registry: dict[str, dict] = {}
    for md in files:
        parsed = parse_file(md)
        if parsed is None:
            continue
        fm, body = parsed
        validate(fm, screamer)
        entry = dict(fm)
        entry["body_markdown"] = body
        registry[fm["name"]] = entry
        print(f"  + {fm['name']:24s}  ({md.relative_to(ROOT)})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n")
    print(f"\nWrote {OUT.relative_to(ROOT)} with {len(registry)} entries")


if __name__ == "__main__":
    main(sys.argv[1:])
