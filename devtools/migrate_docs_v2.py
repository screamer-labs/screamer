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


def migrate_text(text: str) -> str:
    """Return the migrated markdown. Idempotent."""
    # Implementation grows in Tasks 11–13. For now, return text unchanged
    # whenever the file is already in the target shape or has no code at all.
    pre, _, post = text.partition(HELP_END_MARKER)

    # Already migrated: has `## Examples` above HELP_END.
    has_examples_section = bool(re.search(r"^## Examples\s*$", pre, re.M))
    # No code anywhere.
    has_any_fence = "```" in text

    if has_examples_section or not has_any_fence:
        # Still might have a stray post-HELP_END `## Usage Example*`; that case
        # is handled in Task 11. For now, no-op.
        return text

    return text  # Other cases get real logic in Tasks 11–13.


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
