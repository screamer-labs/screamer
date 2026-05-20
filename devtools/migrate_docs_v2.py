#!/usr/bin/env python3
"""One-shot migration: restructure docs/functions_*/<Name>.md to the new layout.

Restructure rules (see docs/superpowers/specs/2026-05-20-help-json-structure-refactor-design.md):

- Post-HELP_END `## Usage Example*` section: move above HELP_END,
  demote to a `## Examples` parent with a `### Usage example` child.
- Pre-HELP_END code fences embedded in prose: extract, append under a
  `## Examples` section just before HELP_END, caption = enclosing H2 heading.
- If both shapes are present, combine into a single `## Examples` section.

The script is idempotent: running it twice on a converted file is a no-op.

After running, hand-tune captions in the flagged files (see run log).

Usage:
    poetry run python devtools/migrate_docs_v2.py                # all 146 files
    poetry run python devtools/migrate_docs_v2.py path/to/X.md   # specific files
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
HELP_END_MARKER = "<!-- HELP_END -->"
USAGE_H2_RE = re.compile(r"^## Usage[^\n]*\n", re.M)
PRE_HELP_FENCE_RE = re.compile(r"```[^\n]*\n.*?\n```", re.S)


def migrate_text(text: str) -> str:
    """Return the migrated markdown. Idempotent."""
    if HELP_END_MARKER not in text:
        return text

    pre, _, post = text.partition(HELP_END_MARKER)

    if re.search(r"^## Examples\s*$", pre, re.M):
        return text  # already migrated

    embedded, pre = _extract_embedded_fences(pre)
    usage_block, post = _extract_usage_block(post)

    pre_examples: list[tuple[str, str]] = list(embedded)
    if usage_block is not None:
        pre_examples.append(("Usage example", usage_block))

    if not pre_examples:
        return text

    new_pre = pre.rstrip() + "\n\n## Examples\n\n"
    for caption, fenced in pre_examples:
        new_pre += f"### {caption}\n\n{fenced.strip()}\n\n"

    return new_pre + HELP_END_MARKER + post


def _extract_embedded_fences(pre: str) -> tuple[list[tuple[str, str]], str]:
    """Find all fenced blocks in the pre-HELP_END region and remove them in place.

    Returns (list_of_(caption, fenced_block), pre_with_fences_removed).
    Caption = the enclosing `## H2` heading text, or "Usage example" if none.
    """
    extracted: list[tuple[str, str]] = []

    def _caption_for(match_start: int) -> str:
        # Walk backwards to find the nearest preceding `## H2` heading.
        h2_iter = list(re.finditer(r"^## (.+)$", pre[:match_start], re.M))
        if h2_iter:
            return h2_iter[-1].group(1).strip()
        return "Usage example"

    pieces: list[str] = []
    cursor = 0
    for m in PRE_HELP_FENCE_RE.finditer(pre):
        pieces.append(pre[cursor:m.start()])
        extracted.append((_caption_for(m.start()), m.group(0)))
        cursor = m.end()
        # Also swallow a trailing newline if present so we don't leave a blank line.
        if cursor < len(pre) and pre[cursor] == "\n":
            cursor += 1
    pieces.append(pre[cursor:])

    return extracted, "".join(pieces)


def _extract_usage_block(post: str) -> tuple[str | None, str]:
    """Find the first `## Usage*` section in `post` and return its inner code fence.

    Returns (fenced_block_or_None, post_with_section_removed).
    The "inner code fence" includes the triple-backtick lines so it can be
    inserted under a `### caption` heading verbatim.
    """
    m = USAGE_H2_RE.search(post)
    if not m:
        return None, post
    start = m.start()
    # Find the end of this section: next H2 or end of string.
    next_h2 = re.search(r"^## ", post[m.end():], re.M)
    end = m.end() + next_h2.start() if next_h2 else len(post)
    section_body = post[m.end():end]
    # The body should contain exactly one fenced code block. Extract it verbatim.
    fence_m = re.search(r"```[^\n]*\n.*?\n```", section_body, re.S)
    if not fence_m:
        # No code in the section; treat as nothing to migrate.
        return None, post
    fenced = fence_m.group(0)
    new_post = post[:start] + post[end:]
    return fenced, new_post


def main(argv: list[str]) -> int:
    if argv:
        paths = [Path(p) for p in argv]
    else:
        paths = sorted(DOCS.glob("functions_*/*.md"))
    changed = 0
    for path in paths:
        original = path.read_text()
        migrated = migrate_text(original)
        if migrated != original:
            path.write_text(migrated)
            changed += 1
            print(f"  migrated: {path.relative_to(ROOT)}")
        else:
            print(f"    no-op:  {path.relative_to(ROOT)}")
    print(f"\n{changed} file(s) migrated, {len(paths) - changed} no-op")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
