"""Guardrails so no public function ships undocumented or mis-tagged.

Every functor, stream operator, and DAG name must have a help.json entry (a doc
page, or a covered twin like the _iter variants). Every entry must declare at
least one topic, and every topic slug must exist in docs/topics.yml. These fail
the moment someone adds a function without wiring up its docs/topic.
"""
import json
from pathlib import Path

import pytest

import screamer
import screamer.screamer_bindings as _b
from devtools.topics import topic_slugs

HELP = json.loads((Path(screamer.__file__).parent / "data" / "help.json").read_text())

# Public names that are intentionally not part of the documented function surface.
_NOT_DOCUMENTED = {
    "EvalOp", "ScreamerBase",                          # abstract bases
    "LazyIterator", "LazyAsyncIterator", "AnextAwaitable",  # iterator plumbing
}


def _documentable_names():
    functors = {
        n for n in dir(screamer)
        if isinstance(getattr(screamer, n), type)
        and issubclass(getattr(screamer, n), _b.EvalOp)
        and getattr(screamer, n) not in (_b.EvalOp, _b.ScreamerBase)
    }
    names = functors | set(screamer.streams.__all__) | set(screamer.dag.__all__)
    return names - _NOT_DOCUMENTED


def test_every_public_function_is_documented():
    missing = sorted(_documentable_names() - set(HELP))
    assert not missing, (
        f"public functions with no help.json entry: {missing}. "
        "Add a docs/functions_*/<Name>.md page (or list the name under a page's "
        "`covers:`) with a `topics:` assignment.")


def test_every_help_entry_declares_at_least_one_topic():
    no_topic = sorted(n for n, e in HELP.items() if not e.get("topics"))
    assert not no_topic, f"help.json entries with no topics: {no_topic}"


def test_every_topic_slug_is_registered():
    valid = topic_slugs()
    bad = {n: [t for t in e.get("topics", []) if t not in valid]
           for n, e in HELP.items()}
    bad = {n: b for n, b in bad.items() if b}
    assert not bad, f"unknown topic slugs (add to docs/topics.yml or fix): {bad}"


def test_topics_registry_loads_in_display_order():
    from devtools.topics import load_topics
    topics = load_topics()
    assert len(topics) >= 15
    assert list(topics)[0] == "arithmetic" and list(topics)[-1] == "graphs"


@pytest.mark.parametrize("name", ["Add", "Sub", "Merge", "Pipeline", "RollingMean"])
def test_representative_names_present(name):
    assert name in HELP and HELP[name].get("topics")
