#!/usr/bin/env python3
"""Generate the typed TS factory layer for screamer.js (Phase 3, Task 3).

Reads ``devtools/wasm/wasm_manifest.json`` (one entry per registered op, each with
``name`` and ``ctor`` = the C++ constructor argument types) and emits:

    js/src/generated/ops.ts    one synchronous factory per op
    js/src/generated/ops.d.ts  matching declarations

Constructor argument NAMES and DEFAULTS come from the installed ``screamer``
Python package. pybind11 hides the real signature behind ``*args, **kwargs`` on
``__init__``, but records it in ``__init__.__doc__`` (e.g.
``__init__(self, window_size: int = 20, start_policy: str = 'strict') -> None``).
We parse that first line. If a signature cannot be recovered, we fall back to
positional ``arg0..argN`` names with no defaults.

The C++ ctor types drive the emitted TS types (not the Python annotations):

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
GEN_DIR = os.path.join(REPO, "js", "src", "generated")
OPS_TS = os.path.join(GEN_DIR, "ops.ts")
OPS_DTS = os.path.join(GEN_DIR, "ops.d.ts")

# C++ ctor type -> TS type.
TS_TYPE = {
    "int": "number",
    "double": "number",
    "std::optional<double>": "number",
    "std::string": "string",
    "std::vector<double>": "number[]",
}


def _split_top_level(s: str) -> list[str]:
    """Split on commas that are not nested inside brackets/parens."""
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch in "[({":
            depth += 1
        elif ch in "])}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    return [p.strip() for p in parts]


def _parse_doc_params(doc: str) -> list[dict] | None:
    """Parse ``__init__(self, name: type = default, ...) -> None`` docstrings.

    Returns a list of ``{"name": str, "default": str | None}`` for the non-self
    parameters, or None if the first line is not a recognizable signature.
    """
    if not doc:
        return None
    first = doc.strip().splitlines()[0]
    if "(" not in first or ")" not in first:
        return None
    inner = first[first.index("(") + 1 : first.rindex(")")]
    raw = _split_top_level(inner)
    if not raw or raw[0] != "self":
        return None
    out = []
    for p in raw[1:]:
        # p looks like "window_size: int = 20" or "lower: float | None = None"
        # or "n: int" (required, no default).
        default = None
        if "=" in p:
            name_type, default = p.split("=", 1)
            default = default.strip()
        else:
            name_type = p
        name = name_type.split(":", 1)[0].strip()
        out.append({"name": name, "default": default})
    return out


def _get_signatures() -> dict[str, list[dict] | None]:
    """Introspect the installed screamer package for each op's ctor params.

    Missing package or op yields None (positional fallback downstream).
    """
    try:
        import screamer  # noqa: F401
    except Exception:
        return {}
    sigs = {}
    for name in dir(screamer):
        obj = getattr(screamer, name)
        init = getattr(obj, "__init__", None)
        doc = getattr(init, "__doc__", None)
        sigs[name] = _parse_doc_params(doc)
    return sigs


def _snake_to_camel(s: str) -> str:
    head, *rest = s.split("_")
    return head + "".join(w[:1].upper() + w[1:] for w in rest)


def _ts_default(cpp_type: str, py_default: str | None) -> str | None:
    """Map a Python default token to a TS literal for the given C++ type.

    ``std::optional<double>`` always defaults to ``NaN`` (missing sentinel).
    """
    if cpp_type == "std::optional<double>":
        return "NaN"
    if py_default is None:
        return None
    d = py_default.strip()
    if cpp_type == "std::string":
        # Python repr uses single quotes; TS prefers double quotes.
        inner = d[1:-1] if len(d) >= 2 and d[0] in "'\"" else d
        return '"' + inner.replace('"', '\\"') + '"'
    if cpp_type == "std::vector<double>":
        # e.g. "[0.25, 0.5, 0.25]" -> valid TS array literal already.
        return d
    # numeric (int / double)
    if d == "None":
        return "NaN"
    if d in ("inf", "float('inf')"):
        return "Infinity"
    if d in ("-inf", "-float('inf')"):
        return "-Infinity"
    if d == "nan":
        return "NaN"
    return d


def _build_params(op: dict, params: list[dict] | None):
    """Return a list of dicts: name, ts_type, default (TS literal or None), is_vec."""
    ctor = op["ctor"]
    out = []
    for i, cpp_type in enumerate(ctor):
        ts_type = TS_TYPE.get(cpp_type)
        if ts_type is None:
            raise SystemExit(f"unmapped ctor type {cpp_type!r} on op {op['name']}")
        if params is not None and i < len(params) and len(params) == len(ctor):
            py_name = params[i]["name"]
            py_default = params[i]["default"]
        else:
            py_name = f"arg{i}"
            py_default = None
        out.append(
            {
                "name": _snake_to_camel(py_name),
                "ts_type": ts_type,
                "default": _ts_default(cpp_type, py_default),
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


def render_ops_ts(ops: list[dict], sigs: dict) -> str:
    out = [HEADER]
    out.append('import { wrapOp, type ScreamerOp } from "../runtime.js";')
    out.append('import { current } from "../index.js";')
    out.append("")
    for op in ops:
        name = op["name"]
        params = _build_params(op, sigs.get(name))
        out.append(
            f"export function {name}({_sig_ts(params)}): ScreamerOp {{"
        )
        out.append(_factory_body(name, params))
        out.append("}")
        out.append("")
    return "\n".join(out)


def render_ops_dts(ops: list[dict], sigs: dict) -> str:
    out = [HEADER]
    out.append('import type { ScreamerOp } from "../runtime.js";')
    out.append("")
    for op in ops:
        name = op["name"]
        params = _build_params(op, sigs.get(name))
        out.append(
            f"export declare function {name}({_sig_decl(params)}): ScreamerOp;"
        )
    out.append("")
    return "\n".join(out)


def load_manifest() -> list[dict]:
    with open(MANIFEST) as f:
        return json.load(f)


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

    sigs = _get_signatures()
    ops_ts = render_ops_ts(ops, sigs)
    ops_dts = render_ops_dts(ops, sigs)

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
