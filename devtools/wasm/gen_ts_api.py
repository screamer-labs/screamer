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
checks then run once on the values. ``undefined`` means a setting's default in
both forms; ``null`` does too for a ``std::optional`` setting, the settings
Python takes ``None`` for, and is a wrong kind for every other. A required
setting (no default) left out raises TypeError, as do more arguments than
settings. An op without settings takes an empty options object or nothing.
Every message the factories raise starts with the op name.

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
    """Return a list of dicts: name, ts_type, default (TS literal or None), kind flags."""
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
                "is_int": cpp_type == "int",
                # Python takes None for exactly these (as "not given"); so does
                # JavaScript with null. Every other setting refuses null.
                "nullable": cpp_type.startswith("std::optional<"),
            }
        )
    return out


def _ts_param_type(p: dict) -> str:
    return f"{p['ts_type']} | null" if p["nullable"] else p["ts_type"]


def _sig_decl(params: list[dict]) -> str:
    """Positional parameter list for an overload (`?` marks a default)."""
    bits = []
    for p in params:
        opt = "?" if p["default"] is not None else ""
        bits.append(f"{p['name']}{opt}: {_ts_param_type(p)}")
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


def _kind(p: dict) -> str:
    """What a setting must be, for messages."""
    if p["is_str"]:
        return "a string"
    if p["is_vec"]:
        return "a number or an array of numbers"
    return "an integer" if p.get("is_int") else "a number"


def _options_type(name: str, params: list[dict], help_entry: dict | None) -> list[str]:
    """The `<Op>Options` type: the settings by camelCase name."""
    if not params:
        return [f"/** {name} has no settings: its options object is empty. */",
                f"export type {name}Options = Record<string, never>;"]
    param_help = {p["name"]: p.get("description") for p in (help_entry or {}).get("parameters") or []}
    lines = [f"/** The settings of {name} by name, for `{name}({{ ... }})`. */",
             f"export interface {name}Options {{"]
    for p in params:
        doc = []
        desc = param_help.get(p["orig_name"])
        if desc:
            doc.append(_escape_comment(desc.strip()))
        if p["default"] is not None and not p["nullable"]:
            doc.append(f"@defaultValue `{p['default']}`")
        if doc:
            lines.append(f"  /** {' '.join(doc)} */")
        opt = "?" if p["default"] is not None else ""
        lines.append(f"  {p['name']}{opt}: {_ts_param_type(p)};")
    lines.append("}")
    return lines


def _factory_body(name: str, params: list[dict], wrapper: str = "wrapOp") -> str:
    vecs = [p for p in params if p["is_vec"]]
    call_args = [f"_v{idx}" if p["is_vec"] else p["name"] for idx, p in enumerate(params)]
    lines = []
    names = [p["name"] for p in params]
    if not params:
        lines.append(f'  if (arguments.length) _noSettings("{name}", options, arguments.length);')
    else:
        if _takes_options(params):
            # One plain object as the only argument holds the settings by name;
            # it is unpacked into the parameters, then checked as a positional call.
            lst = ", ".join(f'"{n}"' for n in names)
            lines.append(f"  if (_isOptions({names[0]})) [{', '.join(names)}] = "
                         f'_options("{name}", {names[0]}, arguments.length, [{lst}], "{names[0]} must be {_kind(params[0])}");')
            lines.append(f"  else if (arguments.length > {len(params)}) _tooMany(\"{name}\", {len(params)}, arguments.length);")
        else:
            lines.append(f"  if (arguments.length > {len(params)}) _tooMany(\"{name}\", {len(params)}, arguments.length);")
    # The settings as the caller gave them, before the defaults, by their
    # Python names: the "params" of a functor node in Pipeline.toJSON, as
    # Python records them (_functor_params.bind_params). An optional
    # setting's NaN, like null, is "not given".
    given = ", ".join(f"(Number.isNaN({p['name']}) ? undefined : {p['name']})" if p["nullable"]
                      else p["name"] for p in params)
    lines.append(f"  const _given: unknown[] = [{given}];")
    spec_tail = f'"{name}", [{", ".join(chr(34) + p["orig_name"] + chr(34) for p in params)}], _given'
    # undefined is the default; null is too where Python takes None (an
    # optional setting); a required setting has no default.
    for p in params:
        n = p["name"]
        if p["default"] is None:
            lines.append(f"  if ({n} === undefined) throw new TypeError(`{name}: {n} is required`);")
        elif p["nullable"]:
            lines.append(f"  {n} ??= {p['default']};")
        else:
            lines.append(f"  if ({n} === undefined) {n} = {p['default']};")
    lines.append("  const M = current();")
    # Each setting is checked here, before C++ reads it: embind would convert
    # any value to a number (true, [3], an object), and would read a non-string
    # object for a string (running its getters). A string ("3") is never parsed
    # (R-a), as in Python.
    for p in params:
        n = p["name"]
        if p["is_str"]:
            lines.append(f'  if (typeof {n} !== "string") throw new TypeError(`{name}: {n} must be a string, '
                         f'got ${{{n} instanceof String ? "a String object" : _what({n})}}`);')
        elif p.get("is_int"):
            # A fraction, NaN or an infinity is not an integer, as in Python.
            lines.append(f'  if (typeof {n} !== "number" || !Number.isInteger({n})) throw new TypeError(`{name}: {n} must be {_kind(p)}, '
                         f'got ${{typeof {n} === "number" ? {n} : _what({n})}}`);')
        elif not p["is_vec"]:
            lines.append(f'  if (typeof {n} !== "number") throw new TypeError(`{name}: {n} must be {_kind(p)}, got ${{_what({n})}}`);')
    # Copy array parameters once, now: the closure below runs again for every
    # copy of the op, and must not see later edits to the caller's array.
    for p in vecs:
        n = p["name"]
        lines.append(f"  const {n}_ = _vec(\"{name}\", \"{n}\", {n});")
    # A float setting that is not finite, or an integer setting below its
    # smallest value: the constructor's RangeError becomes the one message
    # form (finite_settings.h, int_settings.h), on the error path only.
    floats = [p for p in params if not p["is_str"]]
    if floats:
        lines.append("  try {")
        ind = "  "
    else:
        ind = ""
    def close_try():
        if floats:
            names = ", ".join(f'"{p["name"]}"' for p in floats)
            # An optional setting holds NaN when it is not given (embind_runtime.h
            # opt): that is not a value the caller passed.
            vals = ", ".join((f"{p['name']}_" if p["is_vec"] else
                              f"(Number.isNaN({p['name']}) ? null : {p['name']})" if p["nullable"] else p["name"])
                             for p in floats)
            lines.append(f"  }} catch (e) {{ throw _finite(M, \"{name}\", [{names}], [{vals}], e); }}")
    if not vecs:
        lines.append(f"{ind}  return _params({wrapper}(M, () => new M.{name}({', '.join(call_args)})), {spec_tail});")
        close_try()
        return "\n".join(lines)
    lines.append(f"  return _params({wrapper}(M, () => {{")
    # The vectors are deleted however the constructor ends, so a rejected
    # argument does not leak them.
    vs = [f"_v{idx}" for idx, p in enumerate(params) if p["is_vec"]]
    lines.append(f"    const {', '.join(f'{v} = new M.VectorDouble()' for v in vs)};")
    lines.append("    try {")
    for idx, p in enumerate(params):
        if p["is_vec"]:
            # A bare number is the one-entry vector (one value for all assets).
            n = p["name"]
            lines.append(f"      if (typeof {n}_ === \"number\") _v{idx}.push_back({n}_);")
            lines.append(f"      else for (const _x of {n}_) _v{idx}.push_back(_x);")
    lines.append(f"      return new M.{name}({', '.join(call_args)});")
    lines.append("    } finally {")
    for v in vs:
        lines.append(f"      {v}.delete();")
    lines.append("    }")
    lines.append(f"  }}), {spec_tail});")
    if floats:   # indent the block inside the try
        start = lines.index("  try {") + 1
        lines[start:] = [("  " + l) for l in lines[start:]]
    close_try()
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


# Shared by every factory in ops.ts: the value checks and the options object.
RUNTIME_HELPERS = r"""
// A string or String object: never a number setting (R-a).
const _isText = (x: unknown): boolean => typeof x === "string" || x instanceof String;
const _article = (k: string): string => (/^[aeiou]/i.test(k) ? "an " : "a ") + k;
// An NdArray-like {data, shape}: data for the operator, never a setting.
const _isNd = (x: any): boolean => x !== null && typeof x === "object" && "data" in x && "shape" in x;
// The kind of a setting value that is the wrong kind, for the message.
const _what = (x: any): string => _isText(x) ? "a string" : Array.isArray(x) ? "an array" : x === null ? "null" : _isNd(x) ? "an NdArray" : typeof x === "object" ? _article(x.constructor?.name ?? "Object") : typeof x;
// A vector setting is a number, or an Array or a typed array of numbers, as in
// Python; any other container (a Set, a generator) or element is a wrong kind.
// An array is copied once, so every copy of the op reads the same values.
const _vec = (op: string, name: string, x: any): number | number[] => {
  const bad = (y: unknown) => new TypeError(`${op}: ${name} must be a number or an array of numbers, got ${_what(y)}`);
  if (typeof x === "number") return x;
  if (!Array.isArray(x) && !(ArrayBuffer.isView(x) && !(x instanceof DataView))) throw bad(x);
  const a = Array.from(x as ArrayLike<unknown>);
  for (const e of a) if (typeof e !== "number") throw bad(e);
  return a as number[];
};
// Only a plain object (prototype Object.prototype or null) is an options
// object; an array, a typed array or a class instance never is.
const _isOptions = (x: unknown): boolean => {
  if (x === null || typeof x !== "object") return false;
  const p = Object.getPrototypeOf(x);
  return p === Object.prototype || p === null;
};
// The settings in an options object, in parameter order. Every own string key
// counts, enumerable or not; an inherited key is never read, and an own
// `__proto__` key is an unknown option. `first` says what the first positional
// argument must be, for an NdArray passed where the settings belong.
const _options = (op: string, o: any, nArgs: number, names: string[], first: string): any[] => {
  if (nArgs > 1) throw new TypeError(`${op}: an options object must be the only argument`);
  for (const k of Reflect.ownKeys(o)) {
    if (typeof k !== "string" || names.includes(k)) continue;
    if (_isNd(o))
      throw new TypeError(names.length ? `${op}: ${first} or an options object, got an NdArray`
                                       : `${op}: takes no settings, got an NdArray; apply the operator with ${op}()(x)`);
    throw new TypeError(`${op}: unknown option '${k}'; ` + (names.length ? `valid options are ${names.join(", ")}` : `${op} has no options`));
  }
  return names.map((k) => (Object.prototype.hasOwnProperty.call(o, k) ? o[k] : undefined));
};
// An operator without settings takes nothing, or one empty options object.
const _noSettings = (op: string, x: unknown, nArgs: number): void => {
  if (_isOptions(x)) { _options(op, x, nArgs, [], ""); return; }
  if (nArgs === 1 && x === undefined) return;
  throw new TypeError(`${op}: takes no settings, got ${nArgs > 1 ? `${nArgs} arguments` : _what(x)}`);
};
// A constructor's RangeError for a float setting that is not finite, as the
// one message "<Op>: <setting> must be finite, got Infinity" (C++
// finite_settings.h), or for an integer setting below its smallest value,
// "<Op>: <setting> must be at least 1, got -1" (int_settings.h); any other
// error as it is.
const _finite = (M: any, op: string, names: string[], values: unknown[], e: unknown): unknown => {
  if (!(e instanceof RangeError)) return e;
  const ns: string[] = [], vs: unknown[] = [];
  names.forEach((n, i) => {
    const v = values[i];
    if (typeof v === "number" || Array.isArray(v)) { ns.push(n); vs.push(v); }
  });
  const msg = M.finiteSettingError(op, ns, vs);
  return msg ? new RangeError(msg) : e;
};
// Records the settings the caller gave on the op, for Pipeline.toJSON: its
// class, the Python names and the values as given (pipeline_io.ts reads them
// into { cls, params } only when a pipeline is saved, so a construction pays
// one property write).
const _params = (op: ScreamerOp, cls: string, names: string[], given: unknown[]): ScreamerOp => {
  (op as any)[SETTINGS] = { cls, names, given };
  return op;
};
const _tooMany = (op: string, n: number, nArgs: number): never => {
  throw new TypeError(`${op}: takes at most ${n} arguments, got ${nArgs}`);
};
"""[1:-1]


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
        f'import {{ {", ".join(wrappers)}, SETTINGS, type ScreamerOp }} from "../runtime.js";'
    )
    out.append('import { current } from "../index.js";')
    out.append("")
    out.append(RUNTIME_HELPERS)
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
        lines.extend(_options_type(name, params, help_entry))
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
        doc = _jsdoc_lines(name, help_entry, params)
        param = f" * @param options Nothing, or an empty object: {name} has no settings."
        if len(doc) > 1:
            doc.insert(-2, param)
        else:
            doc = ["/**", f" * {name} operator.", " *", param, " */"]
        lines.extend(doc)
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
