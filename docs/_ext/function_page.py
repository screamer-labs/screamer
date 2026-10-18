"""Render the standard header of every function page from its frontmatter.

Each ``docs/functions_*/<Name>.md`` page starts with a YAML frontmatter block
(summary, parameters, inputs, outputs, see also, tags). The same block feeds
``screamer/data/help.json``, Python ``help()`` and the JavaScript type hints.
This extension renders it on the HTML page, so the frontmatter is the one copy:

- search index entries for the title and tags, so a search for "ema" or
  "z-score" finds the function;
- the one-line summary and the Python and JavaScript signatures, under the
  page title;
- a parameter table and the outputs, after the Description section;
- See also and References lists at the end of the page.

The text is inserted while Sphinx reads the source; the files on disk are
unchanged.
"""
import json
import math
import os
import re
from pathlib import Path

import yaml

DOCS = Path(__file__).resolve().parent.parent
HELP_JSON = DOCS.parent / "screamer" / "data" / "help.json"
OPS_TS = DOCS.parent / "js" / "src" / "generated" / "ops.ts"

FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
H1_RE = re.compile(r"^# .*\n", re.M)
H2_RE = re.compile(r"^## (.+)$", re.M)
JS_SIG_RE = re.compile(r"^export function (\w+)\((.*)\): ScreamerOp \{$", re.M)
JS_PARAM_RE = re.compile(r"(\w+): [^=,]+?(?: = ((?:\[[^\]]*\])|[^,]+))?(?:,|$)")


def _js_signatures():
    if not OPS_TS.exists():
        return {}
    sigs = {}
    for name, params in JS_SIG_RE.findall(OPS_TS.read_text()):
        parts = []
        for pname, default in JS_PARAM_RE.findall(params):
            parts.append(f"{pname} = {default.strip()}" if default else pname)
        sigs[name] = f"{name}({', '.join(parts)})"
    return sigs


def _fmt_default(value):
    if isinstance(value, float) and math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if isinstance(value, str):
        return f'"{value}"'
    return repr(value)


def _cell(text):
    return " ".join(str(text).split()).replace("|", "\\|")


def _index_terms(fm):
    terms = []
    for raw in [fm.get("title")] + list(fm.get("tags") or []):
        if not raw:
            continue
        raw = str(raw).strip()
        for term in {raw, raw.replace("-", " ")}:
            term = term.replace(";", ",").replace("!", "")
            if term and term.lower() not in (t.lower() for t in terms):
                terms.append(term)
    return terms


def _is_row_op(name):
    try:
        import screamer
        from screamer import screamer_bindings as b
        cls = getattr(screamer, name, None)
        return isinstance(cls, type) and issubclass(cls, b.RowOp)
    except Exception:
        return False


def _io_line(fm):
    names = fm.get("output_names") or []
    n_in, n_out = fm.get("inputs"), fm.get("outputs")
    if _is_row_op(fm["name"]):
        return None
    parts = []
    if isinstance(n_in, int) and n_in > 1:
        parts.append(f"**Inputs:** {n_in} series, in the order of the Description.")
    if names:
        cols = ", ".join(f"`{n}`" for n in names)
        parts.append(f"**Outputs:** {len(names)} columns per step: {cols}.")
    elif isinstance(n_out, int) and n_out > 1:
        parts.append(f"**Outputs:** {n_out} columns per step, described above.")
    if fm.get("warmup"):
        parts.append(f"**Warmup:** {str(fm['warmup']).strip()}")
    return " ".join(parts) or None


def _parameter_table(fm):
    params = fm.get("parameters") or []
    if not params:
        return ""
    rows = ["| Parameter | Type | Default | Description |", "|---|---|---|---|"]
    for p in params:
        desc = (p.get("description") or "").strip()
        extra = []
        if p.get("enum"):
            extra.append("One of " + ", ".join(f"`{_fmt_default(v)}`" for v in p["enum"]) + ".")
        if "min" in p and "max" in p:
            extra.append(f"From {p['min']} to {p['max']}.")
        elif "min" in p:
            extra.append(f"At least {p['min']}.")
        elif "max" in p:
            extra.append(f"At most {p['max']}.")
        # A required parameter's frontmatter default is only the value the
        # live examples use; the binding has none.
        default = (f"`{_fmt_default(p['default'])}`"
                   if "default" in p and not p.get("required") else "required")
        rows.append(f"| `{p['name']}` | {_cell(p.get('type', ''))} | {default} "
                    f"| {_cell(' '.join([desc] + extra))} |")
    return "## Parameters\n\n" + "\n".join(rows) + "\n"


def _link(docname, target, help_map):
    entry = help_map.get(target)
    if not entry or not entry.get("doc_path"):
        return f"`{target}`"
    rel = os.path.relpath(entry["doc_path"], os.path.dirname(docname))
    short = _cell(entry.get("short", "")).rstrip(".")
    return f"[`{target}`]({rel}.md){': ' + short if short else ''}"


class _State:
    help_map = None
    js_sigs = None


def _source_read(app, docname, source):
    if not docname.startswith("functions_"):
        return
    text = source[0]
    m = FRONTMATTER_RE.match(text)
    if not m:
        return
    fm = yaml.safe_load(m.group(1)) or {}
    if "name" not in fm:
        return
    if _State.help_map is None:
        _State.help_map = json.loads(HELP_JSON.read_text()) if HELP_JSON.exists() else {}
        _State.js_sigs = _js_signatures()

    name = fm["name"]
    kind = fm.get("kind", "functor")
    body = text[m.end():]
    h1 = H1_RE.search(body)
    if not h1:
        return

    header = []
    terms = _index_terms(fm)
    if terms:
        header.append("".join(f"```{{index}} !single: {t}\n```\n" for t in terms))
    if kind == "functor":
        if fm.get("short"):
            header.append(fm["short"].strip() + "\n")
        py_sig = name + "(" + ", ".join(
            f"{p['name']}={_fmt_default(p['default'])}" if "default" in p else p["name"]
            for p in fm.get("parameters") or []) + ")"
        js_sig = _State.js_sigs.get(name)
        if js_sig:
            header.append(
                "::::{tab-set}\n:::{tab-item} Python\n```python\n" + py_sig + "\n```\n:::\n"
                ":::{tab-item} JavaScript\n```javascript\n" + js_sig + "\n```\n:::\n::::\n")
        else:
            header.append("```python\n" + py_sig + "\n```\n")
    body = body[:h1.end()] + "\n" + "\n".join(header) + "\n" + body[h1.end():]

    if kind == "functor":
        block = _parameter_table(fm)
        io = _io_line(fm)
        if io:
            block += ("\n" if block else "") + io + "\n"
        if block:
            headings = list(H2_RE.finditer(body))
            at = None
            for i, h in enumerate(headings):
                if h.group(1).strip() == "Description":
                    at = headings[i + 1].start() if i + 1 < len(headings) else len(body)
                    break
            if at is None:
                at = headings[0].start() if headings else len(body)
            marker = re.search(r"<!-- [A-Z_]+_START -->\n\Z", body[:at])
            if marker:
                at = marker.start()
            body = body[:at] + block + "\n" + body[at:]

    existing = {h.group(1).strip().lower() for h in H2_RE.finditer(body)}
    tail = []
    see_also = fm.get("see_also") or []
    if see_also and "see also" not in existing:
        tail.append("## See also\n\n" + "\n".join(
            f"- {_link(docname, t, _State.help_map)}" for t in see_also) + "\n")
    refs = fm.get("references") or []
    if refs and "references" not in existing:
        tail.append("## References\n\n" + "\n".join(f"- {r}" for r in refs) + "\n")
    if tail:
        body = body.rstrip("\n") + "\n\n" + "\n".join(tail)

    source[0] = text[:m.end()] + body


def setup(app):
    app.connect("source-read", _source_read)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
