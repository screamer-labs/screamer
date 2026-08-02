#!/usr/bin/env python3
"""Generate ``wasm_manifest.json`` by parsing screamer's nanobind bindings.

This is the Task 1 code-generation input for the WASM/Embind phase of the
screamer.js build: it walks every ``bindings/*.cpp`` translation unit
(except the Pipeline/DAG surface, which is out of scope for this phase),
finds each nanobind class registration for a point op --
``nb::class_<CppType, screamer::ScreamerBase>(m, "Name")`` (1-in/1-out ops)
or ``nb::class_<CppType, screamer::EvalOp>(m, "Name")`` (N-in/M-out ops
deriving FunctorBase) -- and records its constructor signature.

Parsing is plain text scanning, not a C++ parser: the bindings follow a
narrow, consistent style (one class template argument list, one
``nb::init<...>`` per op, arguments that are themselves simple types or
``std::optional<double>``), so a small hand-rolled bracket-matcher is
enough. The two things that make naive regexes unsafe here are (a)
``nb::init<...>`` calls that span multiple lines, and (b) nested ``<...>``
and function-pointer ``(...)`` inside a class's own template argument,
e.g. ``screamer::Transform<(double (*)(double)) std::abs>``. Both are
handled by matching angle brackets by depth rather than by regex.

Usage::

    python3 devtools/wasm/gen_wasm_manifest.py           # writes wasm_manifest.json
    python3 devtools/wasm/gen_wasm_manifest.py --check    # validates + prints a summary, no write
    python3 devtools/wasm/gen_wasm_manifest.py --stdout   # writes the manifest JSON to stdout, no write
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BINDINGS_DIR = REPO_ROOT / "bindings"
OUTPUT_PATH = Path(__file__).resolve().parent / "wasm_manifest.json"

# The Pipeline/DAG surface (nb::list-based multi-arg constructors, a
# different binding style entirely) is a later phase; skip it here.
EXCLUDED_FILES = {"bindings_streams.cpp", "bindings_dag.cpp"}

_CLASS_RE = re.compile(r"nb::class_\s*<")
_INIT_RE = re.compile(r"nb::init\s*<")
_NAME_RE = re.compile(r'\s*\(\s*m\s*,\s*"([A-Za-z0-9_]+)"\s*\)')

_BASE_MAP = {
    "screamer::ScreamerBase": "ScreamerBase",
    # N-in/M-out ops are registered against screamer::EvalOp (the common
    # nanobind-visible base for FunctorBase-derived ops); the manifest
    # schema calls this family "FunctorBase".
    "screamer::EvalOp": "FunctorBase",
}


def _matching_angle_close(text: str, open_pos: int) -> int:
    """Return the index of the '>' that matches the '<' at open_pos.

    Only angle-bracket depth is tracked. That is sufficient here because
    every '(' / ')' pair inside a screamer binding template argument (e.g.
    a function-pointer cast) is itself balanced and never contains a
    stray '<' or '>'.
    """
    assert text[open_pos] == "<"
    depth = 1
    i = open_pos + 1
    n = len(text)
    while i < n:
        c = text[i]
        if c == "<":
            depth += 1
        elif c == ">":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError(f"unmatched '<' at position {open_pos}")


def _split_top_level(s: str) -> list[str]:
    """Split s on top-level commas, respecting <...> and (...) nesting."""
    parts = []
    depth = 0
    current: list[str] = []
    for c in s:
        if c in "<(":
            depth += 1
            current.append(c)
        elif c in ">)":
            depth -= 1
            current.append(c)
        elif c == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(c)
    tail = "".join(current).strip()
    if tail or parts:
        parts.append(tail)
    return [p.strip() for p in parts if p.strip() != ""]


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _normalize_ctor_arg(arg: str) -> str:
    """Normalize a single ctor argument type: 'const X&' -> 'X'."""
    a = _normalize_ws(arg)
    if a.startswith("const "):
        a = a[len("const "):].strip()
    if a.endswith("&"):
        a = a[:-1].strip()
    return a


def parse_file(path: Path) -> list[dict]:
    text = path.read_text()
    ops: list[dict] = []

    for m in _CLASS_RE.finditer(text):
        open_pos = m.end() - 1  # index of the '<' itself
        try:
            close_pos = _matching_angle_close(text, open_pos)
        except ValueError:
            continue

        template_content = text[open_pos + 1 : close_pos]
        template_args = [_normalize_ws(a) for a in _split_top_level(template_content)]
        if not template_args:
            continue

        cpp_type = template_args[0]
        base_raw = template_args[1] if len(template_args) > 1 else None
        base = _BASE_MAP.get(base_raw)
        if base is None:
            # Bare nb::class_<T>(...) or an unrecognized base: core
            # infrastructure (EvalOp, LazyEvalIterator, ...), not a point op.
            continue

        after = text[close_pos + 1 :]
        name_match = _NAME_RE.match(after)
        if not name_match:
            continue
        name = name_match.group(1)

        block_start = close_pos + 1 + name_match.end()
        next_class = text.find("nb::class_<", block_start)
        block_end = next_class if next_class != -1 else len(text)
        block = text[block_start:block_end]

        init_match = _INIT_RE.search(block)
        if not init_match:
            # No constructor registered in this block (e.g. ScreamerBase /
            # EvalOp's own nb::class_ registrations expose no nb::init) ->
            # not an instantiable op.
            continue

        init_open_abs = block_start + init_match.end() - 1
        init_close_abs = _matching_angle_close(text, init_open_abs)
        ctor_content = text[init_open_abs + 1 : init_close_abs]
        ctor = [_normalize_ctor_arg(a) for a in _split_top_level(ctor_content)]

        if cpp_type.startswith("screamer::Transform<"):
            ctor_kind = "transform"
        elif any(a == "std::optional<double>" for a in ctor):
            ctor_kind = "ew_optional"
        else:
            ctor_kind = "plain"

        ops.append(
            {
                "name": name,
                "cpp_type": cpp_type,
                "base": base,
                "ctor": ctor,
                "ctor_kind": ctor_kind,
                "source": str(path.relative_to(REPO_ROOT)),
            }
        )

    return ops


def build_manifest() -> list[dict]:
    files = sorted(
        p for p in BINDINGS_DIR.glob("bindings*.cpp") if p.name not in EXCLUDED_FILES
    )
    ops: list[dict] = []
    for f in files:
        ops.extend(parse_file(f))
    ops.sort(key=lambda o: o["name"])
    return ops


def validate(ops: list[dict]) -> None:
    assert len(ops) >= 200, f"expected manifest length >= 200, got {len(ops)}"

    by_name = {op["name"]: op for op in ops}

    assert "RollingMean" in by_name, "RollingMean missing from manifest"
    rm = by_name["RollingMean"]
    expected_rolling_mean = {
        "name": "RollingMean",
        "cpp_type": "screamer::RollingMean",
        "base": "ScreamerBase",
        "ctor": ["int", "std::string"],
        "ctor_kind": "plain",
    }
    for key, expected_value in expected_rolling_mean.items():
        assert rm[key] == expected_value, (
            f"RollingMean.{key}: expected {expected_value!r}, got {rm[key]!r}"
        )

    assert "Abs" in by_name, "Abs missing from manifest"
    abs_op = by_name["Abs"]
    assert abs_op["ctor_kind"] == "transform", (
        f"Abs.ctor_kind: expected 'transform', got {abs_op['ctor_kind']!r}"
    )
    assert abs_op["ctor"] == [], f"Abs.ctor: expected [], got {abs_op['ctor']!r}"
    assert abs_op["cpp_type"].startswith("screamer::Transform<"), (
        f"Abs.cpp_type: expected prefix 'screamer::Transform<', got {abs_op['cpp_type']!r}"
    )

    ew_optional_ops = [op for op in ops if op["ctor_kind"] == "ew_optional"]
    assert len(ew_optional_ops) >= 1, "expected at least one ew_optional op"
    assert "EwMean" in by_name, "EwMean missing from manifest"
    assert by_name["EwMean"]["ctor_kind"] == "ew_optional", (
        f"EwMean.ctor_kind: expected 'ew_optional', got {by_name['EwMean']['ctor_kind']!r}"
    )


def main() -> None:
    ops = build_manifest()
    if "--check" in sys.argv:
        validate(ops)
        print(f"MANIFEST OK: {len(ops)} ops")
    elif "--stdout" in sys.argv:
        # Byte-identical to what a plain run writes to OUTPUT_PATH, just
        # routed to stdout so callers (e.g. the freshness-gate test) can
        # diff without touching the committed file.
        sys.stdout.write(json.dumps(ops, indent=2) + "\n")
    else:
        OUTPUT_PATH.write_text(json.dumps(ops, indent=2) + "\n")
        print(f"Wrote {len(ops)} ops to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
