"""Load the canonical topic registry (docs/topics.yml).

Single source of truth mapping a topic slug -> display name + one-line
description, in display order. Consumed by build_help_registry.py (slug
validation), build_topic_pages.py (index generation), and the doc-coverage tests.
Renaming or reordering a topic is a one-line edit here.
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
TOPICS_YML = ROOT / "docs" / "topics.yml"


def load_topics():
    """Return an ordered dict: slug -> {"name": str, "desc": str} (display order)."""
    data = yaml.safe_load(TOPICS_YML.read_text())
    topics = data.get("topics") or {}
    for slug, entry in topics.items():
        if "name" not in entry or "desc" not in entry:
            raise ValueError(f"topics.yml: topic {slug!r} needs both 'name' and 'desc'")
    return topics


def topic_slugs():
    """The set of valid topic slugs."""
    return set(load_topics())
