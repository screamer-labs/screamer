#!/usr/bin/env python3
"""Generate the JavaScript package's built-in documentation.

Reads screamer/data/help.json, screamer/data/topics.json, and the agent guide,
and writes:

- ``js/src/generated/docs_data.ts``: the documentation in JavaScript terms, as
  one JSON string that ``js/src/docs.ts`` parses on first use. Names follow the
  JavaScript API (``positionTest``, ``combineLatest``), parameters are
  camelCase, and each entry carries its JavaScript signature: built from the
  parameters for a functor, and copied from the TypeScript declarations in
  ``js/src`` for a function or class. Python-only entries are left out.
- ``js/test/docs_oracle.json``: the results of ``screamer.docs.search`` for a
  fixed list of queries, in JavaScript names, so the JavaScript tests can check
  that ``search`` ranks exactly as the Python one does.

Run after help.json or the agent guide changes, and commit the result:

    python3 devtools/wasm/gen_js_docs.py

``--check`` writes nothing and exits 1 when a generated file is stale.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "screamer" / "data"
JS_SRC = ROOT / "js" / "src"
OUT_DATA = JS_SRC / "generated" / "docs_data.ts"
OUT_ORACLE = ROOT / "js" / "test" / "docs_oracle.json"

# Python-only entries: they convert to and from pandas.
PYTHON_ONLY = {"from_pandas", "to_pandas"}
# Stream operators are classes in Python and lowercase functions in JavaScript.
STREAM_OPS = {"CombineLatest", "Delay", "Dropna", "Filter", "Merge", "Resample", "Select"}

ORACLE_QUERIES = [
    ("rolling volatility", None, 10), ("bollinger", None, 5),
    ("exponential moving average", None, 8), ("order flow imbalance", None, 6),
    ("p-value randomization", None, 5), ("kalman", None, 5), ("resample bars", None, 6),
    ("drawdown", None, 6), ("correlation", None, 6), ("test", "significance", 10),
    ("align streams", None, 6), ("dominant cycle", None, 4), ("funding carry", None, 4),
]


def camel(s: str) -> str:
    head, *rest = s.split("_")
    return head + "".join(w[:1].upper() + w[1:] for w in rest)


def js_name(name: str, entry: dict) -> str:
    if entry.get("kind") == "guide":
        return name
    if name in STREAM_OPS:
        return name[0].lower() + name[1:]
    return camel(name) if "_" in name else name


def js_literal(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, float) and not math.isfinite(v):
        return "NaN" if math.isnan(v) else ("Infinity" if v > 0 else "-Infinity")
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(js_literal(x) for x in v) + "]"
    return json.dumps(v, ensure_ascii=False)


def json_safe(v):
    """Non-finite floats as strings, since JSON has no Infinity or NaN."""
    if isinstance(v, float) and not math.isfinite(v):
        return js_literal(v)
    if isinstance(v, list):
        return [json_safe(x) for x in v]
    if isinstance(v, dict):
        return {k: json_safe(x) for k, x in v.items()}
    return v


def _balanced(text: str, start: int) -> int:
    """Index just past the parenthesis that closes the one at ``start``."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i + 1
    raise ValueError("unbalanced parentheses")


def _squash(s: str) -> str:
    s = re.sub(r"\s+", " ", s).replace("( ", "(").replace(" )", ")")
    return s.replace(",)", ")")


def _return_type(text: str, pos: int) -> tuple[str, str]:
    """The return type after a parameter list, and what ends it: ";" or "{".

    ``pos`` is just past the closing parenthesis. A ``{`` at the start of the
    type, or after ``|``, ``&``, ``<``, or ``,``, opens an object type; any
    other ``{`` at depth 0 opens the function body.
    """
    m = re.match(r"\s*:", text[pos:])
    if not m:
        rest = text[pos:].lstrip()
        return "", rest[:1]
    i, depth, out = pos + m.end(), 0, []
    while i < len(text):
        c = text[i]
        if depth == 0 and c == ";":
            return _squash("".join(out)).strip(), ";"
        if depth == 0 and c == "{":
            so_far = "".join(out).strip()
            if so_far and so_far[-1] not in "|&<,:":
                return _squash(so_far), "{"
        if c in "([{<":
            depth += 1
        elif c in ")]}>" and not (c == ">" and text[i - 1] == "="):
            depth -= 1
        out.append(c)
        i += 1
    return _squash("".join(out)).strip(), ";"


def ts_signatures(sources: dict[str, str], name: str) -> list[str]:
    """The exported TypeScript signatures of a function, or a class constructor."""
    sigs = []
    pattern = (rf"^export (?:async )?function\*? ?{re.escape(name)}\b"
               rf"|^export const {re.escape(name)}\b[^=]*= (?:async )?function\b")
    for text in sources.values():
        for m in re.finditer(pattern, text, re.M):
            paren = text.index("(", m.end())
            end = _balanced(text, paren)
            ret, kind = _return_type(text, end)
            sigs.append((kind, f"{name}{_squash(text[paren:end])}" + (f": {ret}" if ret else "")))
        m = re.search(rf"^export class {re.escape(name)}\b", text, re.M)
        if m:
            c = text.find("constructor(", m.end())
            if c != -1:
                end = _balanced(text, c + len("constructor"))
                sigs.append(("{", f"new {name}{_squash(text[c + len('constructor'):end])}"))
    # Overload declarations end in ";"; show those and not the implementation.
    overloads = [s for kind, s in sigs if kind == ";"]
    return overloads or [s for _, s in sigs]


def build() -> dict[Path, str]:
    sys.path.insert(0, str(ROOT))
    from screamer import docs as pydocs

    help_ = json.loads((DATA / "help.json").read_text())
    sources = {p.name: p.read_text() for p in JS_SRC.glob("*.ts")}

    names = {n: js_name(n, e) for n, e in help_.items() if n not in PYTHON_ONLY}
    entries = {}
    for name, jname in names.items():
        e = help_[name]
        out = {
            "name": jname, "pythonName": name, "kind": e.get("kind", "functor"),
            "title": e.get("title", jname), "short": e.get("short", ""),
            "topics": e.get("topics", []), "tags": e.get("tags", []),
            "index": e.get("index", True),
        }
        if "parameters" in e:
            params = []
            for p in e["parameters"]:
                q = {k: json_safe(v) for k, v in p.items()}
                q["name"] = camel(p["name"])
                if "default" in p:
                    q["defaultText"] = js_literal(p["default"])
                params.append(q)
            out["parameters"] = params
            opts = ", ".join(f"{q['name']}: {q['defaultText']}" if "defaultText" in q else q["name"]
                             for q in params)
            out["signatures"] = [f"{jname}({{ {opts} }})" if params else f"{jname}()"]
        elif out["kind"] != "guide":
            out["signatures"] = ts_signatures(sources, jname)
        for key, js_key in (("inputs", "inputs"), ("outputs", "outputs"),
                            ("output_names", "outputNames"), ("nan_policy", "nanPolicy"),
                            ("details", "details"), ("examples", "examples")):
            if key in e:
                out[js_key] = json_safe(e[key])
        out["seeAlso"] = [names[s] for s in e.get("see_also", []) if s in names]
        if e.get("doc_path"):
            out["url"] = f"{pydocs.DOCS_URL}{e['doc_path']}.html"
        entries[jname] = out

    data = {"entries": entries, "groups": json.loads((DATA / "topics.json").read_text()),
            "guide": pydocs.guide()}
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    ts = ("// Generated by devtools/wasm/gen_js_docs.py from screamer/data. Do not edit.\n"
          "// The documentation as one JSON string; docs.ts parses it on first use.\n"
          f"export const DOCS_JSON: string = {json.dumps(text, ensure_ascii=False)};\n")

    oracle = []
    for query, topic, limit in ORACLE_QUERIES:
        hits = [r["name"] for r in pydocs.search(query, topic=topic, limit=10_000)
                if r["name"] in names][:limit]
        oracle.append({"query": query, "topic": topic, "limit": limit,
                       "names": [names[h] for h in hits]})
    oracle_text = json.dumps(oracle, indent=1, ensure_ascii=False) + "\n"
    return {OUT_DATA: ts, OUT_ORACLE: oracle_text}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="write nothing; exit 1 when a generated file is stale")
    args = ap.parse_args(argv)
    stale = []
    for path, text in build().items():
        if path.exists() and path.read_text() == text:
            continue
        stale.append(path)
        if not args.check:
            path.write_text(text)
    for path in stale:
        print(f"{'stale' if args.check else 'wrote'}: {path.relative_to(ROOT)}")
    return 1 if args.check and stale else 0


if __name__ == "__main__":
    sys.exit(main())
