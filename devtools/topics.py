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


def load_groups():
    """Return an ordered dict: group slug -> {"name", "desc", "topics": [slug,...]}."""
    data = yaml.safe_load(TOPICS_YML.read_text())
    groups = data.get("groups") or {}
    for slug, entry in groups.items():
        if "name" not in entry or "desc" not in entry or "topics" not in entry:
            raise ValueError(
                f"topics.yml: group {slug!r} needs 'name', 'desc', and 'topics'")
    return groups


def validate_groups():
    """Every topic must belong to exactly one group. Raises ValueError otherwise."""
    topics = set(load_topics())
    seen = {}
    for gslug, entry in load_groups().items():
        for tslug in entry["topics"]:
            if tslug not in topics:
                raise ValueError(f"group {gslug!r} lists unknown topic {tslug!r}")
            if tslug in seen:
                raise ValueError(
                    f"topic {tslug!r} is in two groups: {seen[tslug]!r} and {gslug!r}")
            seen[tslug] = gslug
    missing = topics - set(seen)
    if missing:
        raise ValueError(f"topics not assigned to any group: {sorted(missing)}")
