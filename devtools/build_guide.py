#!/usr/bin/env python3
"""Build the Guide's landing pages and the home page contents.

Reads ``docs/guide.yml``, which lists the Guide's parts and chapters and the
pages each chapter holds, and writes:

- ``docs/guide/index.md``: the Guide landing page, with every part and chapter.
- ``docs/guide/<slug>/index.md``: one landing page per chapter. Only the block
  between ``<!-- tasks:start -->`` and ``<!-- tasks:end -->`` is generated (the
  numbered task links and the hidden toctree); text outside it is hand-written
  and kept. A missing landing page is created as a stub.
- ``docs/guide_toc.inc``: the same parts and chapters as reStructuredText, for
  the home page.

Chapters are numbered from 0 in registry order. The check fails when a notebook
or recipe is not listed in exactly one chapter, a listed page does not exist,
or a chapter landing's title differs from the registry.

Run after changing docs/guide.yml or adding a page, and commit the result:

    python devtools/build_guide.py

``--check`` writes nothing and exits 1 when a generated file is stale.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
SPEC = DOCS / "guide.yml"
GUIDE = DOCS / "guide"
TOC = DOCS / "guide_toc.inc"

START, END = "<!-- tasks:start -->", "<!-- tasks:end -->"
TITLE_RE = re.compile(r"^# (.+)$", re.M)
BLOCK_RE = re.compile(re.escape(START) + r".*?" + re.escape(END), re.S)


def load() -> dict:
    return yaml.safe_load(SPEC.read_text(encoding="utf-8"))


def chapters(spec: dict) -> list[dict]:
    return [c for p in spec["parts"] for c in p["chapters"]]


def page_title(doc: str) -> str:
    """First ``# `` heading of a page given as a doc path without suffix."""
    md = DOCS / f"{doc}.md"
    if md.exists():
        text = md.read_text(encoding="utf-8")
    elif (DOCS / f"{doc}.ipynb").exists():
        nb = json.loads((DOCS / f"{doc}.ipynb").read_text(encoding="utf-8"))
        text = "\n".join("".join(c["source"]) for c in nb["cells"]
                         if c["cell_type"] == "markdown")
    else:
        raise FileNotFoundError(f"{doc}: no such page")
    m = TITLE_RE.search(text)
    if not m:
        raise ValueError(f"{doc}: no '# Title' heading")
    return m.group(1).strip()


def expected_pages() -> set[str]:
    """Every notebook and non-stub recipe: each must be in one chapter."""
    pages = {f"notebooks/{p.stem}" for p in (DOCS / "notebooks").glob("*.ipynb")}
    for p in (DOCS / "recipes").glob("*.md"):
        if "orphan: true" not in p.read_text(encoding="utf-8"):
            pages.add(f"recipes/{p.stem}")
    return pages


def problems(spec: dict | None = None) -> list[str]:
    spec = spec or load()
    out = []
    listed: list[str] = []
    slugs = set()
    for c in chapters(spec):
        if c["slug"] in slugs:
            out.append(f"duplicate chapter slug {c['slug']}")
        slugs.add(c["slug"])
        docs = c.get("tasks", []) + c.get("deeper", [])
        listed += docs
        for d in docs:
            if not (DOCS / f"{d}.md").exists() and not (DOCS / f"{d}.ipynb").exists():
                out.append(f"listed page {d} does not exist")
            elif d in c.get("tasks", []):
                try:
                    page_title(d)
                except ValueError as e:
                    out.append(str(e))
        landing = GUIDE / c["slug"] / "index"
        if landing.with_suffix(".md").exists():
            try:
                if page_title(f"guide/{c['slug']}/index") != c["title"]:
                    out.append(f"{c['slug']}: landing title differs from guide.yml title")
            except ValueError as e:
                out.append(str(e))
    for d in sorted(expected_pages()):
        n = listed.count(d)
        if n != 1:
            out.append(f"{d} is listed {n} times in guide.yml; it must be in exactly one chapter")
    for d in sorted(set(listed)):
        if listed.count(d) > 1 and d not in expected_pages():
            out.append(f"{d} is listed {listed.count(d)} times in guide.yml")
    return list(dict.fromkeys(out))  # each problem once, in order


def block(c: dict, folder: Path) -> str:
    lines = [START, ""]
    for i, d in enumerate(c.get("tasks", []), 1):
        rel = os.path.relpath(DOCS / f"{d}.md", folder)
        lines.append(f"{i}. [{page_title(d)}]({rel})")
    if c.get("tasks"):
        lines.append("")
    lines += ["```{toctree}", ":hidden:", ""]
    lines += [f"/{d}" for d in c.get("tasks", []) + c.get("deeper", [])]
    lines += ["```", "", END]
    return "\n".join(lines)


def chapter_page(c: dict) -> str:
    path = GUIDE / c["slug"] / "index.md"
    new = block(c, path.parent)
    if path.exists():
        text = path.read_text(encoding="utf-8")
        if BLOCK_RE.search(text):
            return BLOCK_RE.sub(lambda _: new, text, count=1)
        return text.rstrip("\n") + "\n\n" + new + "\n"
    return f"# {c['title']}\n\n{c['line']}\n\n{new}\n"


def guide_index(spec: dict) -> str:
    lines = ["# Guide", "",
             "Each chapter answers a few tasks with a short, runnable example. Read in order, or jump to a chapter.",
             ""]
    n = 0
    for k, part in enumerate(spec["parts"]):
        lines += [f"## Part {k}: {part['title']}", ""]
        for c in part["chapters"]:
            lines.append(f"{n}. [{c['title']}]({c['slug']}/index.md): {c['line']}")
            n += 1
        lines.append("")
    lines += ["```{toctree}", ":hidden:", ""]
    lines += [f"{c['slug']}/index" for c in chapters(spec)]
    lines += ["```"]
    return "\n".join(lines) + "\n"


def toc_inc(spec: dict) -> str:
    lines = [".. Generated by devtools/build_guide.py from docs/guide.yml.",
             ".. Do not edit by hand.", ""]
    n = 0
    for k, part in enumerate(spec["parts"]):
        lines += [f"**Part {k}: {part['title']}**", ""]
        for c in part["chapters"]:
            lines.append(f"{n}. :doc:`{c['title']} <guide/{c['slug']}/index>`: {c['line']}")
            n += 1
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build() -> dict[Path, str]:
    spec = load()
    files = {GUIDE / "index.md": guide_index(spec), TOC: toc_inc(spec)}
    for c in chapters(spec):
        files[GUIDE / c["slug"] / "index.md"] = chapter_page(c)
    return files


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="write nothing; exit 1 when a generated file is stale or the registry is inconsistent")
    args = ap.parse_args(argv)
    bad = problems()
    for p in bad:
        print(f"problem: {p}")
    try:
        files = build()
    except (ValueError, FileNotFoundError) as e:
        if not bad:
            print(f"problem: {e}")
        return 1
    stale = []
    for path, text in files.items():
        if path.exists() and path.read_text(encoding="utf-8") == text:
            continue
        stale.append(path)
        if not args.check:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
    for path in stale:
        print(f"{'stale' if args.check else 'wrote'}: {path.relative_to(ROOT)}")
    return 1 if bad or (args.check and stale) else 0


if __name__ == "__main__":
    sys.exit(main())
