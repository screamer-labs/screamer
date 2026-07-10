#!/usr/bin/env python3
"""Write a flat `name, short description` index of every public function.

Functor descriptions come from the `short` field in screamer/data/help.json
(built from the reference-page frontmatter). The stream operators and the DAG
names have no such frontmatter, so their one-liner is taken from the first
sentence of the docstring. Output: docs/function_index.txt.

Run: poetry run python devtools/build_function_index.py
"""
import json
import re
from pathlib import Path

from screamer import streams, dag

ROOT = Path(__file__).resolve().parent.parent
HELP = json.loads((ROOT / "screamer" / "data" / "help.json").read_text())
OUT = ROOT / "docs" / "function_index.txt"

# Stream operators and DAG names, in a sensible reading order.
STREAM_NAMES = [
    "Stream", "merge", "merge_iter", "combine_latest", "combine_latest_iter",
    "replay", "dropna", "dropna_iter", "filter", "filter_iter",
    "select", "select_iter", "split", "resample",
]
DAG_NAMES = ["Input", "Dag", "Node"]


def first_sentence(obj):
    doc = (obj.__doc__ or "").strip()
    if not doc:
        return "(no docstring)"
    para = re.sub(r"\s+", " ", doc.split("\n\n")[0])
    m = re.match(r"(.+?\.)(\s|$)", para)
    return (m.group(1) if m else para).strip()


def main():
    lines = ["# screamer function index (name, short description)", "", "# Functors"]
    for name in sorted(HELP):
        lines.append(f"{name}, {HELP[name].get('short', '').strip()}")
    lines += ["", "# Stream operators"]
    for name in STREAM_NAMES:
        lines.append(f"{name}, {first_sentence(getattr(streams, name))}")
    lines += ["", "# Computational DAG"]
    for name in DAG_NAMES:
        lines.append(f"{name}, {first_sentence(getattr(dag, name))}")
    OUT.write_text("\n".join(lines) + "\n")
    n = len(HELP) + len(STREAM_NAMES) + len(DAG_NAMES)
    print(f"Wrote {OUT.relative_to(ROOT)} with {n} entries.")


if __name__ == "__main__":
    main()
