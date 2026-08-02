#!/usr/bin/env python3
"""Generate the typed TS factory layer for screamer.js (Phase 3, Task 3).

Reads ``devtools/wasm/wasm_manifest.json`` (one entry per registered op, each
with ``name``, ``ctor`` = the C++ constructor argument types, and ``ctor_args``
= the aligned ``{"name", "default"}`` list parsed from the committed
``"name"_a = default`` binding annotations) and emits:

    js/src/generated/ops.ts    one synchronous factory per op
    js/src/generated/ops.d.ts  matching declarations

This generator is pure source -> source: it reads only the manifest and the
committed ``screamer/data/help.json`` (never imports the ``screamer``
runtime), so its output is identical in every environment. Constructor
argument NAMES and DEFAULTS come from ``ctor_args`` (deterministic, sourced
from committed C++). When ``ctor_args`` is empty but the op has ctor types
(annotation/arity mismatch upstream), we fall back to positional
``arg0..argN`` names with no defaults.

Each factory is preceded by a JSDoc block sourced from ``help.json``'s
``short`` (one-line description) and per-parameter ``description`` fields --
the same op descriptions that back the Python docs -- so editor tooltips and
the generated API reference share one source of truth.

The C++ ctor types drive the emitted TS types (not the annotation defaults):

    int / double / std::optional<double> -> number
    std::string                          -> string
    std::vector<double>                  -> number[]

``std::optional<double>`` parameters default to ``NaN`` (the missing-optional
sentinel that the C++ side reads back as an absent value). ``std::vector<double>``
parameters are materialized into an Embind ``VectorDouble`` inside the factory,
passed to the constructor, then freed (the ctor copies the vector).

Modes:
    (no args)     regenerate ops.ts and ops.d.ts
    --check       assert every manifest op has a factory + a declaration
    --stdout      print ops.ts to stdout (freshness gate, Task 5)
    --stdout-dts  print ops.d.ts to stdout (freshness gate, Task 5)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MANIFEST = os.path.join(HERE, "wasm_manifest.json")
HELP_JSON = os.path.join(REPO, "screamer", "data", "help.json")
GEN_DIR = os.path.join(REPO, "js", "src", "generated")
OPS_TS = os.path.join(GEN_DIR, "ops.ts")
OPS_DTS = os.path.join(GEN_DIR, "ops.d.ts")
PYTHON_DOCS_URL = "https://screamer.readthedocs.io/en/latest/"

# C++ ctor type -> TS type.
TS_TYPE = {
    "int": "number",
    "double": "number",
    "std::optional<double>": "number",
    "std::string": "string",
    "std::vector<double>": "number[]",
}


def _snake_to_camel(s: str) -> str:
    head, *rest = s.split("_")
    return head + "".join(w[:1].upper() + w[1:] for w in rest)


def _num_literal(cpp_type: str, tok: str) -> str:
    """Render a single numeric C++ literal token as a TS numeric literal.

    ``int`` keeps its integer form; ``double`` is normalized through Python's
    float repr (e.g. ``1e-5`` -> ``1e-05``, ``252`` -> ``252.0``), matching how
    the runtime signature previously rendered these defaults.
    """
    t = tok.strip()
    low = t.lstrip("-")
    if "infinity" in low or low in ("HUGE_VAL", "std::numeric_limits<double>::infinity()"):
        return "-Infinity" if t.startswith("-") else "Infinity"
    if cpp_type == "int":
        return str(int(t))
    return repr(float(t))


def _ts_default(cpp_type: str, raw: str | None) -> str | None:
    """Map a C++ default literal token to a TS literal for the given C++ type.

    ``std::optional<double>`` always defaults to ``NaN`` (the missing sentinel).
    A ``None`` token (a bare annotation or an ``nb::none()`` / ``std::nullopt``
    sentinel) means no default.
    """
    if cpp_type == "std::optional<double>":
        return "NaN"
    if raw is None:
        return None
    d = raw.strip()
    if cpp_type == "std::string":
        inner = d[1:-1] if len(d) >= 2 and d[0] in "'\"" else d
        return '"' + inner.replace('"', '\\"') + '"'
    if cpp_type == "std::vector<double>":
        # e.g. 'std::vector<double>{0.25, 0.5, 0.25}' -> '[0.25, 0.5, 0.25]'.
        inner = d[d.index("{") + 1 : d.rindex("}")]
        elems = [e.strip() for e in inner.split(",") if e.strip()]
        return "[" + ", ".join(_num_literal("double", e) for e in elems) + "]"
    # numeric (int / double)
    return _num_literal(cpp_type, d)


def _build_params(op: dict):
    """Return a list of dicts: name, ts_type, default (TS literal or None), is_vec."""
    ctor = op["ctor"]
    ctor_args = op.get("ctor_args") or []
    # ctor_args, when present, is aligned 1:1 with ctor by the manifest
    # generator; an empty list means positional fallback.
    use_args = len(ctor_args) == len(ctor)
    out = []
    for i, cpp_type in enumerate(ctor):
        ts_type = TS_TYPE.get(cpp_type)
        if ts_type is None:
            raise SystemExit(f"unmapped ctor type {cpp_type!r} on op {op['name']}")
        if use_args:
            arg_name = ctor_args[i]["name"]
            arg_default = ctor_args[i]["default"]
        else:
            arg_name = f"arg{i}"
            arg_default = None
        out.append(
            {
                "name": _snake_to_camel(arg_name),
                "orig_name": arg_name,
                "ts_type": ts_type,
                "default": _ts_default(cpp_type, arg_default),
                "is_vec": cpp_type == "std::vector<double>",
            }
        )
    return out


def _sig_ts(params: list[dict]) -> str:
    """Parameter list for an implementation (with `= default`)."""
    bits = []
    for p in params:
        d = "" if p["default"] is None else f" = {p['default']}"
        bits.append(f"{p['name']}: {p['ts_type']}{d}")
    return ", ".join(bits)


def _sig_decl(params: list[dict]) -> str:
    """Parameter list for a declaration (`?` instead of `= default`)."""
    bits = []
    for p in params:
        opt = "?" if p["default"] is not None else ""
        bits.append(f"{p['name']}{opt}: {p['ts_type']}")
    return ", ".join(bits)


def _factory_body(name: str, params: list[dict]) -> str:
    vecs = [p for p in params if p["is_vec"]]
    lines = ["  const M = current();"]
    call_args = []
    for idx, p in enumerate(params):
        if p["is_vec"]:
            v = f"_v{idx}"
            lines.append(f"  const {v} = new M.VectorDouble();")
            lines.append(f"  for (const _x of {p['name']}) {v}.push_back(_x);")
            call_args.append(v)
        else:
            call_args.append(p["name"])
    if vecs:
        lines.append(f"  const _op = new M.{name}({', '.join(call_args)});")
        for idx, p in enumerate(params):
            if p["is_vec"]:
                lines.append(f"  _v{idx}.delete();")
        lines.append("  return wrapOp(M, _op);")
    else:
        lines.append(f"  return wrapOp(M, new M.{name}({', '.join(call_args)}));")
    return "\n".join(lines)


HEADER = (
    "// AUTO-GENERATED by devtools/wasm/gen_ts_api.py. Do not edit by hand.\n"
    "// Regenerate: poetry run python devtools/wasm/gen_ts_api.py\n"
)


def load_manifest() -> list[dict]:
    with open(MANIFEST) as f:
        return json.load(f)


def load_help() -> dict:
    """Load ``screamer/data/help.json`` -> ``{op_name: entry}``.

    ``help.json`` is a committed data file (op descriptions shared with the
    Python docs), read here as plain JSON -- never via a runtime ``import
    screamer`` -- so this generator stays deterministic and dependency-free.
    The committed shape is a dict keyed by op name; a list of entries (each
    with a ``"name"`` field) is also accepted defensively.
    """
    with open(HELP_JSON) as f:
        data = json.load(f)
    if isinstance(data, list):
        return {entry["name"]: entry for entry in data if "name" in entry}
    return data


def _escape_comment(text: str) -> str:
    """Escape a `*/` sequence so it can't prematurely close a JSDoc block."""
    return text.replace("*/", "*\\/")


def _jsdoc_lines(op_name: str, help_entry: dict | None, params: list[dict]) -> list[str]:
    """Render the JSDoc block preceding a factory, as a list of source lines."""
    short = (help_entry or {}).get("short")
    if not short:
        return [f"/** {op_name} operator. */"]

    text = short.strip()
    if not text.endswith("."):
        text += "."
    param_help = {p["name"]: p.get("description") for p in (help_entry or {}).get("parameters") or []}

    lines = ["/**"]
    lines.append(f" * {_escape_comment(text)}")
    lines.append(" *")
    for p in params:
        desc = param_help.get(p["orig_name"])
        if desc:
            lines.append(f" * @param {p['name']} {_escape_comment(desc.strip())}")
        else:
            lines.append(f" * @param {p['name']}")
    lines.append(f" * @see {PYTHON_DOCS_URL} for the Python reference and full details.")
    lines.append(" */")
    return lines


def render_ops_ts(ops: list[dict], help_map: dict) -> str:
    out = [HEADER]
    out.append('import { wrapOp, type ScreamerOp } from "../runtime.js";')
    out.append('import { current } from "../index.js";')
    out.append("")
    for op in ops:
        name = op["name"]
        params = _build_params(op)
        out.extend(_jsdoc_lines(name, help_map.get(name), params))
        out.append(
            f"export function {name}({_sig_ts(params)}): ScreamerOp {{"
        )
        out.append(_factory_body(name, params))
        out.append("}")
        out.append("")
    return "\n".join(out)


def render_ops_dts(ops: list[dict], help_map: dict) -> str:
    out = [HEADER]
    out.append('import type { ScreamerOp } from "../runtime.js";')
    out.append("")
    for op in ops:
        name = op["name"]
        params = _build_params(op)
        out.extend(_jsdoc_lines(name, help_map.get(name), params))
        out.append(
            f"export declare function {name}({_sig_decl(params)}): ScreamerOp;"
        )
    out.append("")
    return "\n".join(out)


def check(ops: list[dict]) -> int:
    for path in (OPS_TS, OPS_DTS):
        if not os.path.exists(path):
            print(f"MISSING generated file: {path}", file=sys.stderr)
            return 1
    ts = open(OPS_TS).read()
    dts = open(OPS_DTS).read()
    missing = []
    for op in ops:
        name = op["name"]
        if not re.search(rf"export function {re.escape(name)}\(", ts):
            missing.append(("factory", name))
        if not re.search(rf"export declare function {re.escape(name)}\(", dts):
            missing.append(("decl", name))
    if missing:
        for kind, name in missing:
            print(f"MISSING {kind}: {name}", file=sys.stderr)
        return 1
    print(f"TS API OK: {len(ops)} factories")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify coverage")
    ap.add_argument("--stdout", action="store_true", help="print ops.ts to stdout")
    ap.add_argument(
        "--stdout-dts", action="store_true", help="print ops.d.ts to stdout"
    )
    args = ap.parse_args()

    ops = load_manifest()

    if args.check:
        return check(ops)

    help_map = load_help()
    ops_ts = render_ops_ts(ops, help_map)
    ops_dts = render_ops_dts(ops, help_map)

    if args.stdout:
        sys.stdout.write(ops_ts)
        return 0

    if args.stdout_dts:
        sys.stdout.write(ops_dts)
        return 0

    os.makedirs(GEN_DIR, exist_ok=True)
    with open(OPS_TS, "w") as f:
        f.write(ops_ts)
    with open(OPS_DTS, "w") as f:
        f.write(ops_dts)
    print(f"wrote {OPS_TS}")
    print(f"wrote {OPS_DTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
