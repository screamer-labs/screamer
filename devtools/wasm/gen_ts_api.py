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
    std::vector<double>                  -> number | number[] (a bare number is one entry)

``std::optional<double>`` parameters default to ``NaN`` (the missing-optional
sentinel that the C++ side reads back as an absent value). ``std::vector<double>``
parameters are materialized into an Embind ``VectorDouble`` inside the factory,
passed to the constructor, then freed (the ctor copies the vector).

Each factory also takes one plain object as its only argument, the settings by
camelCase name (``EwMean({ span: 20 })``), typed by an ``<Op>Options`` type and
an overload. The object is unpacked into the parameters, and the positional
checks then run once on the values. ``null`` and ``undefined`` mean a setting's
default in both forms; a required setting (no default) raises TypeError. An op
without settings takes an empty options object or nothing.

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
    "std::vector<double>": "number | number[]",
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
                "is_str": cpp_type == "std::string",
            }
        )
    return out


def _sig_decl(params: list[dict]) -> str:
    """Positional parameter list for an overload (`?` marks a default)."""
    bits = []
    for p in params:
        opt = "?" if p["default"] is not None else ""
        bits.append(f"{p['name']}{opt}: {p['ts_type']}")
    return ", ".join(bits)


def _sig_impl(params: list[dict]) -> str:
    """Implementation parameter list: every value is checked in the body."""
    return ", ".join(f"{p['name']}?: any" for p in params)


# The TS types a value of which is never a plain object. Only an op whose first
# parameter has one of these types takes an options object; an op whose first
# parameter could itself be an object (an NdArray `{data, shape}`) would not,
# decided here from the type and never from the value.
_OPTIONS_SAFE_FIRST = {"number", "string", "number | number[]"}


def _takes_options(params: list[dict]) -> bool:
    return not params or params[0]["ts_type"] in _OPTIONS_SAFE_FIRST


def _options_type(name: str, params: list[dict], help_entry: dict | None, decl: bool) -> list[str]:
    """The `<Op>Options` type: the settings by camelCase name."""
    if not params:
        return [f"/** {name} has no settings: its options object is empty. */",
                f"export type {name}Options = Record<string, never>;"]
    param_help = {p["name"]: p.get("description") for p in (help_entry or {}).get("parameters") or []}
    lines = [f"/** The settings of {name} by name, for `{name}({{ ... }})`. */",
             f"export interface {name}Options {{"]
    for p in params:
        desc = param_help.get(p["orig_name"])
        if desc:
            lines.append(f"  /** {_escape_comment(desc.strip())} */")
        opt = "?" if p["default"] is not None else ""
        lines.append(f"  {p['name']}{opt}: {p['ts_type']};")
    lines.append("}")
    return lines


def _factory_body(name: str, params: list[dict], wrapper: str = "wrapOp") -> str:
    vecs = [p for p in params if p["is_vec"]]
    call_args = [f"_v{idx}" if p["is_vec"] else p["name"] for idx, p in enumerate(params)]
    lines = []
    names = [p["name"] for p in params]
    if not params:
        lines.append(f'  if (_isOptions(options)) _options("{name}", options, arguments.length, []);')
    elif _takes_options(params):
        # One plain object as the only argument holds the settings by name; it
        # is unpacked into the parameters, then checked as a positional call.
        lst = ", ".join(f'"{n}"' for n in names)
        lines.append(f"  if (_isOptions({names[0]})) [{', '.join(names)}] = "
                     f'_options("{name}", {names[0]}, arguments.length, [{lst}]);')
    # null or undefined is the default; a required setting has none.
    for p in params:
        n = p["name"]
        if p["default"] is None:
            lines.append(f"  if ({n} == null) throw new TypeError(`{name}: {n} is required`);")
        else:
            lines.append(f"  {n} ??= {p['default']};")
    lines.append("  const M = current();")
    # A string parameter reaches C++ as a val and is read there; checking it here
    # keeps C++ from reading a non-string object (and running its getters).
    for p in params:
        if p.get("is_str"):
            n = p["name"]
            lines.append(f'  if (typeof {n} !== "string") throw new TypeError(`{n} must be a string, got ${{typeof {n}}}`);')
        elif not p["is_vec"]:
            # A number setting is a number: a string ("3") is a wrong kind, never
            # parsed (R-a), as in Python.
            n = p["name"]
            lines.append(f'  if (_isText({n})) throw new TypeError(`{n} must be a number, got a string`);')
    # Copy array parameters once, now: the closure below runs again for every
    # copy of the op, and must not see later edits to the caller's array.
    for p in vecs:
        n = p["name"]
        lines.append(f"  const {n}_ = _vec(\"{n}\", {n});")
        # A string, a String object, or an array holding one is a wrong kind
        # (R-a), checked once on the copy, as in Python.
        lines.append(f"  if (_isText({n}_) || (Array.isArray({n}_) && {n}_.some(_isText))) "
                     f"throw new TypeError(`{n} must be a number or an array of numbers, got a string`);")
    if not vecs:
        lines.append(f"  return {wrapper}(M, () => new M.{name}({', '.join(call_args)}));")
        return "\n".join(lines)
    lines.append(f"  return {wrapper}(M, () => {{")
    # The vectors are deleted however the constructor ends, so a rejected
    # argument does not leak them.
    vs = [f"_v{idx}" for idx, p in enumerate(params) if p["is_vec"]]
    lines.append(f"    const {', '.join(f'{v} = new M.VectorDouble()' for v in vs)};")
    lines.append("    try {")
    for idx, p in enumerate(params):
        if p["is_vec"]:
            # A bare number is the one-entry vector (one value for all assets).
            # Every element is a number, whatever iterable holds it (an Array,
            # a Set, a generator): a string or a nested array is a wrong kind,
            # never read by embind's ToNumber (R-a).
            n = p["name"]
            lines.append(f"      for (const _x of (typeof {n}_ === \"number\" ? [{n}_] : {n}_)) {{")
            lines.append(f"        if (typeof _x !== \"number\") throw new TypeError(`{n} must be a number or an array of numbers, got ${{_what(_x)}}`);")
            lines.append(f"        _v{idx}.push_back(_x);")
            lines.append("      }")
    lines.append(f"      return new M.{name}({', '.join(call_args)});")
    lines.append("    } finally {")
    for v in vs:
        lines.append(f"      {v}.delete();")
    lines.append("    }")
    lines.append("  });")
    return "\n".join(lines)


def _wrapper_for(op: dict) -> str:
    """The runtime wrapper a factory hands its raw Embind op to.

    Dynamic-width reducers take the variable-group wrapper; every other op
    takes the fixed-width one.
    """
    if op.get("row_op"):
        return "wrapRowOp"
    return "wrapReducerOp" if op.get("dynamic_reducer") else "wrapOp"


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
    page = (help_entry or {}).get("doc_path")
    url = f"{PYTHON_DOCS_URL}{page}.html" if page else PYTHON_DOCS_URL
    lines.append(f" * @see {url} for the full reference.")
    lines.append(" */")
    return lines


def render_ops_ts(ops: list[dict], help_map: dict) -> str:
    wrappers = ["wrapOp"]
    if any(op.get("dynamic_reducer") for op in ops):
        wrappers.append("wrapReducerOp")
    if any(op.get("row_op") for op in ops):
        wrappers.append("wrapRowOp")
    out = [HEADER]
    out.append(
        f'import {{ {", ".join(wrappers)}, type ScreamerOp }} from "../runtime.js";'
    )
    out.append('import { current } from "../index.js";')
    out.append("")
    out.append("// A string or String object: never a number setting (R-a).")
    out.append('const _isText = (x: unknown): boolean => typeof x === "string" || x instanceof String;')
    out.append('const _article = (k: string): string => (/^[aeiou]/i.test(k) ? "an " : "a ") + k;')
    out.append("// The kind of a setting value that is not a number, for the message.")
    out.append('const _what = (x: any): string => _isText(x) ? "a string" : Array.isArray(x) ? "an array" : x === null ? "null" : typeof x === "object" ? _article(x.constructor?.name ?? "Object") : typeof x;')
    out.append("// A vector setting is a number, an Array or a typed array, as in Python; any")
    out.append("// other container (a Set, a generator) is a wrong kind. An array is copied")
    out.append("// once, so every copy of the op reads the same values.")
    out.append("const _vec = (name: string, x: any): any => {")
    out.append('  if (typeof x === "number") return x;')
    out.append("  if (Array.isArray(x) || (ArrayBuffer.isView(x) && !(x instanceof DataView))) return Array.from(x as ArrayLike<unknown>);")
    out.append("  throw new TypeError(`${name} must be a number or an array of numbers, got ${_what(x)}`);")
    out.append("};")
    out.append("// Only a plain object (prototype Object.prototype or null) is an options")
    out.append("// object; an array, a typed array or a class instance never is.")
    out.append("const _isOptions = (x: unknown): boolean => {")
    out.append('  if (x === null || typeof x !== "object") return false;')
    out.append("  const p = Object.getPrototypeOf(x);")
    out.append("  return p === Object.prototype || p === null;")
    out.append("};")
    out.append("// The settings in an options object, in parameter order. Only own enumerable")
    out.append("// keys count, so an inherited key is never read and an own `__proto__` key")
    out.append("// is an unknown option.")
    out.append("const _options = (op: string, o: any, nArgs: number, names: string[]): any[] => {")
    out.append("  if (nArgs > 1) throw new TypeError(`${op}: an options object must be the only argument`);")
    out.append("  for (const k of Object.keys(o))")
    out.append("    if (!names.includes(k))")
    out.append("      throw new TypeError(`${op}: unknown option '${k}'; ` + (names.length ? `valid options are ${names.join(\", \")}` : `${op} has no options`));")
    out.append("  return names.map((k) => (Object.prototype.propertyIsEnumerable.call(o, k) ? o[k] : undefined));")
    out.append("};")
    out.append("")
    for op in ops:
        out.extend(_render_op(op, help_map, decl=False))
        out.append("")
    return "\n".join(out)


def _options_jsdoc(op_name: str, help_entry: dict | None) -> list[str]:
    """JSDoc for the options-object overload."""
    short = ((help_entry or {}).get("short") or f"{op_name} operator").strip()
    if not short.endswith("."):
        short += "."
    page = (help_entry or {}).get("doc_path")
    url = f"{PYTHON_DOCS_URL}{page}.html" if page else PYTHON_DOCS_URL
    return ["/**", f" * {_escape_comment(short)}", " *",
            f" * @param options The settings by name, each one optional where the positional form's is.",
            f" * @see {url} for the full reference.", " */"]


def _render_op(op: dict, help_map: dict, decl: bool) -> list[str]:
    """The options type, the overloads, and (for ops.ts) the implementation."""
    name = op["name"]
    params = _build_params(op)
    help_entry = help_map.get(name)
    fn = "export declare function" if decl else "export function"
    lines = []
    opts = _takes_options(params)
    if opts:
        lines.extend(_options_type(name, params, help_entry, decl))
        lines.append("")
    if params:
        if opts:
            lines.extend(_options_jsdoc(name, help_entry))
            lines.append(f"{fn} {name}(options: {name}Options): ScreamerOp;")
        lines.extend(_jsdoc_lines(name, help_entry, params))
        lines.append(f"{fn} {name}({_sig_decl(params)}): ScreamerOp;")
        if not decl:
            lines.append(f"export function {name}({_sig_impl(params)}): ScreamerOp {{")
    else:
        lines.extend(_jsdoc_lines(name, help_entry, params))
        lines.append(f"{fn} {name}(options?: {name}Options): ScreamerOp" + (";" if decl else " {"))
    if not decl:
        lines.append(_factory_body(name, params, _wrapper_for(op)))
        lines.append("}")
    return lines


def render_ops_dts(ops: list[dict], help_map: dict) -> str:
    out = [HEADER]
    out.append('import type { ScreamerOp } from "../runtime.js";')
    out.append("")
    for op in ops:
        out.extend(_render_op(op, help_map, decl=True))
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
