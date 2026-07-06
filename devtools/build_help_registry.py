#!/usr/bin/env python3
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
import json
import inspect
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
PLOTLY_DIRECTIVE_RE = re.compile(
    r"\A\.\. plotly::\s*\n"        # ".. plotly::" + newline
    r"((?:[ \t]+:[^\n]*\n)*)"      # zero or more ":option: value" lines
    r"\s*\n"                       # blank line
    r"(.+)",                       # indented body (captured)
    re.S,
)
# pybind11 emits ``__init__(self: <type>, foo: int, bar: str = 'x') -> None``
# as the first line of the docstring. Capture the parameter names.
PYBIND_SIG_RE = re.compile(r"__init__\(self[^,)]*(?:,\s*([^)]*))?\)")


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
        if language == "{eval-rst}":
            code = _unwrap_plotly(code)
            language = "python"
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


ALLOWED_NAN_POLICIES = ("ignore", "propagate", "nan-aware")


def _first_sentence(obj) -> str:
    """First sentence of a callable's docstring (for covered-name shorts)."""
    doc = (obj.__doc__ or "").strip()
    if not doc:
        return ""
    para = re.sub(r"\s+", " ", doc.split("\n\n")[0])
    m = re.match(r"(.+?\.)(\s|$)", para)
    return (m.group(1) if m else para).strip()


def validate(entry: dict, screamer_module, valid_topics=None) -> None:
    """Validate one help entry.

    Topic assignment is required for every function. ``kind: functor`` (default)
    gets the full class validation (param drift + live round-trip); any other
    kind (``function`` for stream operators, ``graph`` for the DAG names) gets a
    lighter, signature-based check with no round-trip, since those call
    conventions vary (``*values``, predicates, async generators, ...).
    """
    name = entry["name"]
    obj = getattr(screamer_module, name, None)
    if obj is None:
        raise RuntimeError(f"screamer has no name {name!r}")

    # Every function must belong to at least one known topic (the topic index and
    # the left nav are generated from these).
    topics = entry.get("topics") or []
    if not topics:
        raise ValueError(
            f"{name}: frontmatter must declare at least one topic (see docs/topics.yml)")
    if valid_topics is not None:
        unknown = [t for t in topics if t not in valid_topics]
        if unknown:
            raise ValueError(
                f"{name}: unknown topic slug(s) {sorted(unknown)}; "
                "valid slugs are the keys of docs/topics.yml")

    if entry.get("kind", "functor") != "functor":
        _validate_function(entry, obj)
    else:
        _validate_functor(entry, obj)


def _validate_functor(entry: dict, cls) -> None:
    name = entry["name"]
    # Every functor must declare its NaN policy. The contract is defined in
    # docs/nan_policy.md; refusing to publish without an explicit policy is what
    # makes the contract dogmatic.
    policy = entry.get("nan_policy")
    if policy is None:
        raise ValueError(
            f"{name}: frontmatter must declare nan_policy "
            f"(one of {list(ALLOWED_NAN_POLICIES)}). See docs/nan_policy.md.")
    if policy not in ALLOWED_NAN_POLICIES:
        raise ValueError(f"{name}: nan_policy={policy!r} not in {list(ALLOWED_NAN_POLICIES)}")

    schema_params = entry.get("parameters", []) or []
    kwargs = {}
    for p in schema_params:
        if "default" not in p:
            raise ValueError(
                f"{name}.{p['name']}: every parameter must have a 'default' "
                "so the frontend can render live results without user input")
        kwargs[p["name"]] = p["default"]

    # Schema-vs-binding parameter-name drift check.
    binding_names = pybind_param_names(cls)
    schema_names = [p["name"] for p in schema_params]
    extra_in_schema = set(schema_names) - set(binding_names)
    missing_in_schema = set(binding_names) - set(schema_names)
    if extra_in_schema:
        raise ValueError(
            f"{name}: schema lists parameters not in the binding: {sorted(extra_in_schema)}")
    if missing_in_schema:
        raise ValueError(
            f"{name}: binding has parameters not in the schema: {sorted(missing_in_schema)}")

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


def _validate_function(entry: dict, fn) -> None:
    """Light validation for non-functor entries (stream operators, DAG names).

    No instantiate-and-call round-trip. nan_policy is optional. Any documented
    parameters must have a default and match the callable's signature.
    """
    name = entry["name"]
    policy = entry.get("nan_policy")
    if policy is not None and policy not in ALLOWED_NAN_POLICIES:
        raise ValueError(f"{name}: nan_policy={policy!r} not in {list(ALLOWED_NAN_POLICIES)}")

    schema_params = entry.get("parameters", []) or []
    for p in schema_params:
        if "default" not in p:
            raise ValueError(f"{name}.{p['name']}: every parameter must have a 'default'")
    if schema_params and callable(fn) and not isinstance(fn, type):
        try:
            sig_names = set(inspect.signature(fn).parameters)
        except (ValueError, TypeError):
            sig_names = None
        if sig_names is not None:
            extra = [p["name"] for p in schema_params if p["name"] not in sig_names]
            if extra:
                raise ValueError(
                    f"{name}: schema lists parameters not in the signature: {sorted(extra)}")


def main(argv=None):
    sys.path.insert(0, str(ROOT))
    screamer = importlib.import_module("screamer")
    from devtools.topics import topic_slugs
    valid_topics = topic_slugs()

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
        validate(entry, screamer, valid_topics)
        registry[entry["name"]] = entry
        n_ex = len(entry["examples"])
        print(f"  + {entry['name']:24s}  ({md.relative_to(ROOT)}, {n_ex} examples)")

        # Names documented on this same page (the _iter twins, Input/Node on
        # Dag.md). Emit an index:false entry each, inheriting the page's topics,
        # so every public name is covered without a dedicated page.
        for cname in entry.get("covers", []) or []:
            cobj = getattr(screamer, cname, None)
            if cobj is None:
                raise SystemExit(f"{md.relative_to(ROOT)}: covers unknown name {cname!r}")
            registry[cname] = {
                "name": cname,
                "title": cname,
                "short": _first_sentence(cobj),
                "kind": entry.get("kind", "function"),
                "topics": entry["topics"],
                "index": False,
                "details": "",
                "examples": [],
            }
            print(f"  + {cname:24s}  (covered by {entry['name']}, index:false)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n")
    print(f"\nWrote {OUT.relative_to(ROOT)} with {len(registry)} entries")


if __name__ == "__main__":
    main(sys.argv[1:])
