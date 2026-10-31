"""Render the standard header of every function page from its frontmatter.

Each ``docs/functions_*/<Name>.md`` page starts with a YAML frontmatter block
(summary, parameters, inputs, outputs, see also, tags). The same block feeds
``screamer/data/help.json``, Python ``help()`` and the JavaScript type hints.
This extension renders it on the HTML page, so the frontmatter is the one copy:

- search index entries for the title and tags, so a search for "ema" or
  "z-score" finds the function;
- the one-line summary and the Python and JavaScript signatures, under the
  page title;
- the Examples section moved to directly after the header, so the reader
  sees the example before the Description, with each plot shown above its
  full source;
- a parameter table and the outputs, after the Description section;
- a "Used in the Guide" list of the Guide task pages whose code cells use
  the function, in the order of ``docs/guide.yml`` (the first five, then a
  count of the rest);
- See also and References lists at the end of the page.

The text is inserted while Sphinx reads the source; the files on disk are
unchanged.
"""
import ast
import functools
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
FENCE_RE = re.compile(r"^(`{3,}|~{3,})", re.M)
JS_SIG_RE = re.compile(r"^export function (\w+)\((.*)\): ScreamerOp \{$", re.M)
JS_PARAM_RE = re.compile(r"(\w+): [^=,]+?(?: = ((?:\[[^\]]*\])|[^,]+))?(?:,|$)")
START_MARKER_RE = re.compile(r"<!-- [A-Z_]+_START -->\n\Z")
PLOTLY_RE = re.compile(r"^( *)\.\. plotly::\s*$")
INCLUDE_SOURCE_RE = re.compile(r"^( *:include-source:\s*)(\S+)\s*$")
CODE_CELL_RE = re.compile(r"^```\{code-cell\}[^\n]*\n(.*?)^```", re.M | re.S)
FROM_IMPORT_RE = re.compile(
    r"^\s*from\s+screamer(?:\.\w+)*\s+import\s+(\([^)]*\)|[^\n]*)", re.M)
MODULE_IMPORT_RE = re.compile(r"^\s*import\s+screamer(?:\s+as\s+(\w+))?\s*$", re.M)
COMMENT_RE = re.compile(r"#[^\n]*")
# A function page lists at most this many Guide pages, then "and N more".
USED_IN_MAX = 5


def _js_signatures():
    if not OPS_TS.exists():
        return {}
    sigs = {}
    for name, params in JS_SIG_RE.findall(OPS_TS.read_text(encoding="utf-8")):
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


def _section_start(text, heading):
    """Start of a section, including a ``<!-- X_START -->`` marker above it."""
    marker = START_MARKER_RE.search(text[:heading.start()])
    return marker.start() if marker else heading.start()


def _headings(text, pos=0):
    """``## `` headings in ``text`` from ``pos`` on, skipping fenced code."""
    fenced, fence = [], None
    for m in FENCE_RE.finditer(text):
        line_end = text.find("\n", m.start())
        rest = text[m.end():line_end if line_end >= 0 else len(text)]
        if fence is None:
            fence = (m.group(1), m.start())
        elif m.group(1)[0] == fence[0][0] and len(m.group(1)) >= len(fence[0]) \
                and not rest.strip():
            fenced.append((fence[1], m.end()))
            fence = None
    if fence is not None:
        fenced.append((fence[1], len(text)))
    return [h for h in H2_RE.finditer(text, pos)
            if not any(a <= h.start() < b for a, b in fenced)]


def _plot_then_source(text):
    """Show each example plot first and its full source directly below it.

    Every ``.. plotly::`` directive in ``text`` with ``:include-source: True``
    or with no ``:include-source:`` option gets ``:include-source: False``,
    followed by a ``.. code-block:: python`` at the same indent holding the
    directive's source lines unchanged. A directive that sets
    ``:include-source: False`` and the text between directives are left as
    they are.
    """
    lines = text.split("\n")
    out = []
    i = 0
    while i < len(lines):
        m = PLOTLY_RE.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        indent = len(m.group(1))
        j = i + 1
        while j < len(lines) and (not lines[j].strip()
                                  or len(lines[j]) - len(lines[j].lstrip()) > indent):
            j += 1
        while j > i + 1 and not lines[j - 1].strip():
            j -= 1
        block = lines[i:j]
        opts = [k for k, line in enumerate(block)
                if INCLUDE_SOURCE_RE.match(line) and k < _first_blank(block)]
        if opts and INCLUDE_SOURCE_RE.match(block[opts[0]]).group(2) != "True":
            out.extend(block)
            i = j
            continue
        if opts:
            k = opts[0]
            block[k] = INCLUDE_SOURCE_RE.match(block[k]).group(1) + "False"
        else:
            body = next((line for line in block[1:] if line.strip()), "")
            pad = body[:len(body) - len(body.lstrip())] or " " * (indent + 4)
            block.insert(1, pad + ":include-source: False")
        source = block[_first_blank(block):]
        out.extend(block)
        out.extend(["", " " * indent + ".. code-block:: python"] + source)
        i = j
    return "\n".join(out)


def _first_blank(block):
    """Index of the blank line that ends a directive's options."""
    for k, line in enumerate(block):
        if not line.strip():
            return k
    return len(block)


def _examples_first(text):
    """Move the ``## Examples`` section in front of the first section.

    ``text`` holds the page's sections, from the first ``## `` heading on. The
    Examples section runs up to the next ``## `` heading; its plots are passed
    through ``_plot_then_source``. Text without an Examples section is returned
    unchanged.
    """
    headings = _headings(text)
    for i, h in enumerate(headings):
        if h.group(1).strip() == "Examples":
            break
    else:
        return text
    start = h.start()
    end = _section_start(text, headings[i + 1]) if i + 1 < len(headings) else len(text)
    first = _section_start(text, headings[0]) if i else start
    chunk = _plot_then_source(text[start:end].rstrip("\n")) + "\n\n"
    return text[:first] + chunk + text[first:start] + text[end:]


def screamer_names(code):
    """Names a code cell takes from screamer.

    A name counts when the cell imports it (``from screamer import Name``) or
    reads it from the module (``import screamer as sc`` then ``sc.Name``).
    Comments, strings and longer names that contain it do not count.
    """
    lines = ["" if line.lstrip().startswith(("%", "!")) else line
             for line in code.splitlines()]
    try:
        tree = ast.parse("\n".join(lines))
    except SyntaxError:
        return _screamer_names_re(code)
    names, aliases = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and (
                node.module or "").split(".")[0] == "screamer":
            names.update(a.name for a in node.names if a.name != "*")
        elif isinstance(node, ast.Import):
            aliases.update(a.asname or a.name for a in node.names if a.name == "screamer")
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id in aliases):
            names.add(node.attr)
    return names


def _screamer_names_re(code):
    """Line-based fallback for a cell that does not parse as Python."""
    code = COMMENT_RE.sub("", code)
    names = set()
    for m in FROM_IMPORT_RE.finditer(code):
        for part in m.group(1).strip("()").split(","):
            word = part.split(" as ")[0].strip()
            if word.isidentifier():
                names.add(word)
    for m in MODULE_IMPORT_RE.finditer(code):
        alias = m.group(1) or "screamer"
        names.update(re.findall(rf"(?<![\w.]){alias}\.(\w+)", code))
    return names


def guide_usage(docs):
    """Map each screamer name to the Guide task pages that use it.

    Reads the task pages listed in ``docs/guide.yml`` under ``docs`` and
    returns ``{name: [docname, ...]}`` with the pages in registry order.
    """
    spec = yaml.safe_load((docs / "guide.yml").read_text(encoding="utf-8")) or {}
    usage = {}
    for part in spec.get("parts") or []:
        for chapter in part.get("chapters") or []:
            for doc in chapter.get("tasks") or []:
                path = docs / f"{doc}.md"
                if not doc.startswith("guide/") or not path.exists():
                    continue
                text = path.read_text(encoding="utf-8")
                names = set()
                for cell in CODE_CELL_RE.findall(text):
                    names |= screamer_names(cell)
                for name in names:
                    usage.setdefault(name, []).append(doc)
    return usage


@functools.lru_cache(maxsize=None)
def _page_title(path):
    m = H1_RE.search(path.read_text(encoding="utf-8"))
    return m.group(0)[2:].strip() if m else path.stem


def _used_in(docname, name, usage):
    """The "Used in the Guide" section of a function page, or ``""``."""
    docs = usage.get(name)
    if not docs:
        return ""
    lines = []
    for doc in docs[:USED_IN_MAX]:
        rel = os.path.relpath(doc, os.path.dirname(docname))
        lines.append(f"- [{_page_title(DOCS / f'{doc}.md')}]({rel}.md)")
    if len(docs) > USED_IN_MAX:
        lines.append(f"- and {len(docs) - USED_IN_MAX} more")
    return "## Used in the Guide\n\n" + "\n".join(lines) + "\n"


class _State:
    help_map = None
    js_sigs = None
    usage = None


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
        _State.help_map = json.loads(HELP_JSON.read_text(encoding="utf-8")) if HELP_JSON.exists() else {}
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
            headings = _headings(body)
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

    first = next(iter(_headings(body, h1.end())), None)
    if first:
        at = _section_start(body, first)
        body = body[:at] + _examples_first(body[at:])

    if _State.usage is None:
        _State.usage = guide_usage(DOCS)
    existing = {h.group(1).strip().lower() for h in _headings(body)}
    tail = []
    used_in = _used_in(docname, name, _State.usage)
    if used_in and "used in the guide" not in existing:
        tail.append(used_in)
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


def _guide_outdated(app, env, added, changed, removed):
    """Function pages whose "Used in the Guide" list changed since the last build.

    The list depends on the Guide task pages, not on the function page, so
    Sphinx would not reread the page when a task page starts or stops using the
    function, or changes its title.
    """
    _page_title.cache_clear()
    _State.usage = guide_usage(DOCS)
    links = {name: [(doc, _page_title(DOCS / f"{doc}.md")) for doc in docs]
             for name, docs in _State.usage.items()}
    old = getattr(env, "screamer_guide_links", None)
    env.screamer_guide_links = links
    pages = [d for d in env.found_docs if d.startswith("functions_")]
    if old is None:  # an environment from before this list existed
        return pages
    names = {n for n in set(old) | set(links) if old.get(n) != links.get(n)}
    return [d for d in pages if d.rsplit("/", 1)[-1] in names]


def setup(app):
    app.connect("source-read", _source_read)
    app.connect("env-get-outdated", _guide_outdated)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
