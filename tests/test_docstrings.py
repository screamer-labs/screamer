"""Every public operator has a help() text built from its documentation page."""
import json
from importlib import resources

import pytest

import screamer
from screamer._functor_params import _is_functor

HELP = json.loads(resources.files("screamer").joinpath("data/help.json").read_text())
FUNCTORS = sorted(n for n in dir(screamer)
                  if _is_functor(getattr(screamer, n)) and n in HELP)


def test_every_functor_is_covered():
    assert len(FUNCTORS) > 200


@pytest.mark.parametrize("name", FUNCTORS)
def test_docstring_from_help_registry(name):
    doc = getattr(screamer, name).__doc__
    entry = HELP[name]
    assert doc and doc.startswith(f"{name}(")
    assert entry["short"].strip() in doc
    for p in entry.get("parameters") or []:
        assert f"\n{p['name']}" in doc
    assert f"{entry['doc_path']}.html" in doc
    assert getattr(screamer, name).__module__ == "screamer"
