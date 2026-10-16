"""Write llms.txt and llms-full.txt at the root of the HTML build.

``llms.txt`` (https://llmstxt.org) is an index of the site for language models:
the agent guide, the guides, every function grouped by topic, and the worked
examples, each linked to its page source as plain text. ``llms-full.txt``
holds the agent guide and the full documentation of every function in one
file. Both come from the installed screamer's built-in documentation
(``screamer.docs``) and from docs/examples.yml, so they always match the
pages built next to them.
"""
import os
from pathlib import Path

import yaml

# Guides listed in llms.txt, in order: (doc path, title, one-line note).
GUIDES = [
    ("agents", "Using screamer with AI agents", "how agents can query the documentation"),
    ("usage", "Using screamer", "construction, calling, shapes, warmup, chaining, pipelines"),
    ("polymorphic_api", "Polymorphic API", "the exact input-to-output contract for every container"),
    ("nan_and_warmup", "NaN and warmup", "missing-data policies and start policies"),
    ("streaming", "Streams in depth", "live, one value at a time, iterators, and state"),
    ("multistream", "Streams, values, and alignment", "aligning streams that tick at different times"),
    ("pipelines", "Pipelines", "reusable graphs of functions"),
    ("choosing_an_operator", "Choosing an operator", "which function for which job"),
    ("backtesting_best_practices", "Backtesting best practices", "costs, fills, and testing for luck"),
    ("portfolio_backtesting", "Portfolios and pairs", "many assets from one account"),
    ("labels", "Labels and training sets", "targets for supervised learning, paired with forecast_pairs"),
    ("microstructure", "Market microstructure", "order flow, impact, and toxicity"),
    ("conventions", "Conventions", "where screamer matches or differs from TA-Lib and pandas"),
    ("javascript", "JavaScript and the browser", "the WebAssembly package"),
]


def _base_url() -> str:
    url = os.environ.get("READTHEDOCS_CANONICAL_URL") or "https://screamer.readthedocs.io/en/latest/"
    return url if url.endswith("/") else url + "/"


def _source_url(base: str, srcdir: Path, doc: str) -> str:
    """The plain-text page source Sphinx publishes under _sources/."""
    for suffix in (".md", ".ipynb", ".rst"):
        if (srcdir / f"{doc}{suffix}").exists():
            return f"{base}_sources/{doc}{suffix}.txt"
    return f"{base}{doc}.html"


def _title(srcdir: Path, doc: str) -> str:
    """A recipe's or notebook's first '# ' heading."""
    import json
    import re
    md, nb = srcdir / f"{doc}.md", srcdir / f"{doc}.ipynb"
    if md.exists():
        text = md.read_text()
    else:
        text = "\n".join("".join(c["source"]) for c in json.loads(nb.read_text())["cells"]
                         if c["cell_type"] == "markdown")
    m = re.search(r"^# (.+)$", text, re.M)
    return m.group(1).strip() if m else doc


def build_llms_txt(srcdir: Path, base: str) -> str:
    from screamer import docs

    guide_short = ("Causal time-series functions for finance and signal processing, "
                   "implemented in C++, with the same results on stored arrays and on "
                   "a live feed. Python package `screamer`, JavaScript package "
                   "`@screamer-labs/screamer`.")
    lines = [
        "# screamer", "", f"> {guide_short}", "",
        "Before writing a calculation by hand, find the screamer function for it. "
        "With screamer installed, `python -m screamer.docs search <keywords>` and "
        "`python -m screamer.docs show <Name>` describe the installed version, and "
        "`pip install \"screamer[mcp]\"` adds an MCP server, `screamer-mcp`, with the "
        "same information. Function pages below link to their markdown source.",
        "",
        "## Guides", "",
        f"- [Agent guide]({base}llms-full.txt): how to use screamer, followed by the "
        "full documentation of every function, in one file",
    ]
    for doc, title, note in GUIDES:
        lines.append(f"- [{title}]({_source_url(base, srcdir, doc)}): {note}")
    lines.append("")

    registry = {}
    for g in docs.topics():
        lines += [f"## Functions: {g['name']}", "", g["desc"], ""]
        for t in g["topics"]:
            for name in t["functions"]:
                if name in registry:
                    continue
                entry = docs.show(name)
                registry[name] = entry
                path = docs._registry()[name].get("doc_path")
                url = _source_url(base, srcdir, path) if path else entry.get("url", "")
                lines.append(f"- [{name}]({url}): {entry.get('short', '')}")
        lines.append("")

    spec = yaml.safe_load((srcdir / "examples.yml").read_text())
    lines += ["## Examples", ""]
    for area in spec["areas"]:
        for ex in area["examples"]:
            lines.append(f"- [{area['title']}: {_title(srcdir, ex['doc'])}]"
                         f"({_source_url(base, srcdir, ex['doc'])}): {ex['text']}")
    lines.append("")
    return "\n".join(lines)


def build_llms_full_txt() -> str:
    from screamer import docs

    parts = [docs.guide().rstrip(), ""]
    seen = set()
    for g in docs.topics():
        parts += [f"# Functions: {g['name']}", "", g["desc"], ""]
        for t in g["topics"]:
            for name in t["functions"]:
                if name not in seen:
                    seen.add(name)
                    # One heading level down, under the group heading.
                    text = docs._format_show(docs.show(name))
                    parts += ["#" + text.rstrip(), ""]
    return "\n".join(parts) + "\n"


def _write(app, exception):
    if exception is not None or app.builder.format != "html":
        return
    srcdir, outdir, base = Path(app.srcdir), Path(app.outdir), _base_url()
    (outdir / "llms.txt").write_text(build_llms_txt(srcdir, base), encoding="utf-8")
    (outdir / "llms-full.txt").write_text(build_llms_full_txt(), encoding="utf-8")


def setup(app):
    app.connect("build-finished", _write)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
