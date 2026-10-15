"""Search and read screamer's built-in documentation.

The documentation of every function ships with the package, so it always
describes the installed version. It is meant for people at a prompt and for
coding agents alike::

    python -m screamer.docs topics
    python -m screamer.docs search "order flow imbalance"
    python -m screamer.docs show RollingStd
    python -m screamer.docs guide

Add ``--json`` for structured output. The same calls are available in Python
as :func:`topics`, :func:`search`, :func:`show`, and :func:`guide`, which return
plain dicts, lists, and strings.
"""
from __future__ import annotations

import argparse
import difflib
import inspect
import json
import re
import sys
from functools import lru_cache
from importlib import resources

__all__ = ["topics", "search", "show", "guide", "skill"]

DOCS_URL = "https://screamer.readthedocs.io/en/latest/"


@lru_cache(maxsize=None)
def _read(name: str) -> str:
    return resources.files("screamer").joinpath("data", name).read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def _registry() -> dict:
    return json.loads(_read("help.json"))


def _url(entry: dict) -> str | None:
    path = entry.get("doc_path")
    return f"{DOCS_URL}{path}.html" if path else None


def _summary(entry: dict) -> dict:
    return {"name": entry["name"], "kind": entry.get("kind", "functor"),
            "short": entry.get("short", ""), "topics": entry.get("topics", [])}


def topics() -> list[dict]:
    """The topic groups, each with its topics and the functions in each topic.

    Returns a list of groups in display order: ``{"slug", "name", "desc",
    "topics": [{"slug", "name", "desc", "functions": [name, ...]}]}``.
    """
    by_topic: dict[str, list[str]] = {}
    for name, entry in _registry().items():
        if entry.get("index", True):
            for t in entry.get("topics", []):
                by_topic.setdefault(t, []).append(name)
    groups = json.loads(_read("topics.json"))
    for g in groups:
        for t in g["topics"]:
            t["functions"] = sorted(by_topic.get(t["slug"], []), key=str.lower)
    return groups


_WORD_RE = re.compile(r"[a-z0-9]+")


def _words(text: str) -> list[str]:
    # A trailing "s" is dropped so that "bands" finds "band" and the reverse.
    return [w[:-1] if len(w) > 3 and w.endswith("s") else w
            for w in _WORD_RE.findall(text.lower())]


@lru_cache(maxsize=None)
def _index() -> list[tuple[dict, list[tuple[set, int]]]]:
    """Per entry, its fields as word sets with a weight each."""
    index = []
    for entry in _registry().values():
        name = entry["name"]
        # Split CamelCase and snake_case names into their words too.
        name_words = " ".join(re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", name))
        fields = [
            (set(_words(name)) | set(_words(name_words)), 6),
            (set(_words(" ".join(entry.get("tags", [])))), 4),
            (set(_words(entry.get("title", "") + " " + entry.get("short", ""))), 3),
            (set(_words(" ".join(entry.get("topics", [])))), 2),
            (set(_words(entry.get("details", ""))), 1),
        ]
        index.append((entry, fields))
    return index


def search(query: str, topic: str | None = None, limit: int = 10) -> list[dict]:
    """Functions that match ``query``, best match first.

    Each word of the query is looked up in a function's name, tags, title and
    summary, topics, and description, in that order of weight, and a word
    found in several of them counts for each. A function that matches more of
    the words ranks higher, and a name equal to the query ranks first. ``topic`` keeps only the functions in that topic slug (see
    :func:`topics`). Returns up to ``limit`` dicts with ``name``, ``kind``,
    ``short``, and ``topics``.
    """
    words = _words(query)
    exact = query.strip().lower()
    scored = []
    for entry, fields in _index():
        if topic is not None and topic not in entry.get("topics", []):
            continue
        hits, score = 0, 0
        for w in words:
            weight = sum(weight for ws, weight in fields if w in ws)
            hits += weight > 0
            score += weight
        if entry["name"].lower() == exact:
            score += 100
        if score:
            scored.append((-(hits * 10 + score), entry["name"].lower(), entry))
    scored.sort(key=lambda s: s[:2])
    return [_summary(e) for _, _, e in scored[:limit]]


def _signature(entry: dict) -> str | None:
    params = entry.get("parameters")
    if params is not None:
        args = ", ".join(f"{p['name']}={p['default']!r}" if "default" in p else p["name"]
                         for p in params)
        return f"{entry['name']}({args})"
    import screamer
    obj = getattr(screamer, entry["name"], None)
    if obj is None or not callable(obj):
        return None
    try:
        return f"{entry['name']}{inspect.signature(obj)}"
    except (TypeError, ValueError):
        return None


def show(name: str) -> dict:
    """The full documentation of one function.

    Returns the entry with ``name``, ``kind``, ``title``, ``short``,
    ``signature``, ``parameters`` (each with ``type``, ``default``, and where
    they apply ``min``, ``max``, ``enum``, and ``description``), ``inputs``,
    ``outputs``, ``nan_policy``, ``topics``, ``see_also``, ``details``
    (markdown), ``examples`` (``language``, ``caption``, ``code``), and ``url``.
    Fields that do not apply to a function are absent. An unknown name raises
    ``KeyError``, with the closest names in its message.
    """
    reg = _registry()
    entry = reg.get(name)
    if entry is None:
        lower = {k.lower(): k for k in reg}
        if name.lower() in lower:
            entry = reg[lower[name.lower()]]
        else:
            close = difflib.get_close_matches(name, list(reg), n=5, cutoff=0.5)
            hint = f"; did you mean {', '.join(close)}?" if close else ""
            raise KeyError(f"no function named {name!r}{hint}")
    out = {k: v for k, v in entry.items() if k not in ("doc_path", "index", "covers")}
    out.setdefault("kind", "functor")
    sig = _signature(entry)
    if sig:
        out["signature"] = sig
    url = _url(entry)
    if url:
        out["url"] = url
    return out


def guide() -> str:
    """How to use screamer, written for coding agents (markdown)."""
    return _read("agent_guide.md")


def skill() -> str:
    """The agent guide as an Agent Skill file (``SKILL.md``).

    Save it as ``<skills directory>/screamer/SKILL.md``, for example
    ``~/.claude/skills/screamer/SKILL.md`` for Claude Code.
    """
    head = (
        "---\n"
        "name: screamer\n"
        "description: Use when writing Python or JavaScript code that computes "
        "rolling statistics, technical indicators, filters, market microstructure "
        "measures, bars from ticks, backtests, or significance tests with the "
        "screamer library, or when choosing which screamer function to use.\n"
        "---\n\n"
    )
    return head + guide()


# --------------------------------------------------------------------------
# Text output for the command line
# --------------------------------------------------------------------------

def _format_topics(groups: list[dict]) -> str:
    lines = []
    for g in groups:
        lines += [f"{g['name']}", f"  {g['desc']}"]
        for t in g["topics"]:
            lines.append(f"  - {t['slug']}: {', '.join(t['functions'])}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _format_search(results: list[dict]) -> str:
    if not results:
        return "No matches.\n"
    width = max(len(r["name"]) for r in results)
    return "".join(f"{r['name']:<{width}}  {r['short']}\n" for r in results)


def _format_show(e: dict) -> str:
    lines = [f"# {e['name']}", "", e.get("short", ""), ""]
    if "signature" in e:
        lines += ["```python", e["signature"], "```", ""]
    if e.get("parameters"):
        lines.append("## Parameters")
        lines.append("")
        for p in e["parameters"]:
            extra = []
            if "type" in p:
                extra.append(p["type"])
            if "default" in p:
                extra.append(f"default {p['default']!r}")
            if "enum" in p:
                extra.append("one of " + ", ".join(repr(v) for v in p["enum"]))
            if "min" in p:
                extra.append(f"min {p['min']}")
            if "max" in p:
                extra.append(f"max {p['max']}")
            lines.append(f"- `{p['name']}` ({'; '.join(extra)}): {p.get('description', '')}")
        lines.append("")
    facts = []
    if "inputs" in e:
        facts.append(f"Inputs: {e['inputs']}")
    if "outputs" in e:
        names = e.get("output_names")
        facts.append(f"Outputs: {e['outputs']}" + (f" ({', '.join(names)})" if names else ""))
    if "nan_policy" in e:
        facts.append(f"NaN policy: {e['nan_policy']}")
    if e.get("topics"):
        facts.append(f"Topics: {', '.join(e['topics'])}")
    if e.get("see_also"):
        facts.append(f"See also: {', '.join(e['see_also'])}")
    if "url" in e:
        facts.append(f"Documentation: {e['url']}")
    lines += facts + [""]
    if e.get("details"):
        lines += [e["details"].strip(), ""]
    for ex in e.get("examples", []):
        lines += [f"## Example: {ex.get('caption') or e['name']}", "",
                  f"```{ex.get('language', 'python')}", ex["code"].rstrip(), "```", ""]
    return "\n".join(lines).rstrip() + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m screamer.docs",
        description="Search and read screamer's documentation (the installed version).")
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("topics", help="list the topic groups and their functions")
    p_search = sub.add_parser("search", help="find functions by keywords")
    p_search.add_argument("query", nargs="+", help="keywords, such as: order flow imbalance")
    p_search.add_argument("--topic", help="only functions in this topic slug")
    p_search.add_argument("--limit", type=int, default=10, help="most results to show")
    p_show = sub.add_parser("show", help="the full documentation of one function")
    p_show.add_argument("name", help="function name, such as RollingStd")
    sub.add_parser("guide", help="how to use screamer, written for coding agents")
    sub.add_parser("skill", help="the guide as an Agent Skill file (SKILL.md)")
    # Accept --json after the subcommand as well as before it.
    for p in sub.choices.values():
        p.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                       help="print JSON instead of text")
    args = ap.parse_args(argv)

    if args.command == "topics":
        data, text = topics(), _format_topics
    elif args.command == "search":
        data = search(" ".join(args.query), topic=args.topic, limit=args.limit)
        text = _format_search
    elif args.command == "show":
        try:
            data = show(args.name)
        except KeyError as exc:
            print(exc.args[0], file=sys.stderr)
            return 1
        text = _format_show
    elif args.command == "guide":
        data, text = guide(), str
    else:
        data, text = skill(), str
    sys.stdout.write(json.dumps(data, indent=2, ensure_ascii=False) + "\n"
                     if args.json else text(data))
    return 0


if __name__ == "__main__":
    sys.exit(main())
