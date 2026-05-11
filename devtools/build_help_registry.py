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
import importlib
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
OUT = ROOT / "screamer" / "data" / "help.json"

FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
HELP_END_MARKER = "<!-- HELP_END -->"
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


def pybind_param_names(cls) -> list[str]:
    """Extract parameter names from a pybind11 class' __init__ docstring."""
    doc = (cls.__init__.__doc__ or "").splitlines()[0]
    m = PYBIND_SIG_RE.search(doc)
    if not m or not m.group(1):
        return []
    params = []
    for chunk in m.group(1).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        # name: type [= default]
        name = chunk.split(":", 1)[0].strip()
        params.append(name)
    return params


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
