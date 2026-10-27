"""Function pages show the example first.

docs/_ext/function_page.py renders each function page's header from its
frontmatter at read time. It also moves the ``## Examples`` section to directly
after that header, so a reader sees the example before the Description.
"""
import importlib.util
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
EXT = REPO_ROOT / "docs" / "_ext" / "function_page.py"

# The release wheel-test CI job runs the suite against the installed wheel; skip
# there if the docs sources are absent.
pytestmark = pytest.mark.skipif(not EXT.exists(), reason="needs the docs sources")

FRONTMATTER = """---
name: Foo
title: Foo
short: Does foo to a series.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  description: Number of most recent samples in the window.
---

# `Foo`

"""

DESCRIPTION = "## Description\n\nFoo does foo.\n\n"
NAN = ("<!-- NAN_FOOTNOTE_START -->\n## NaN handling\n\n"
       "A NaN gives a NaN.\n<!-- NAN_FOOTNOTE_END -->\n\n")
EXAMPLES = "## Examples\n\n### Usage example\n\n```python\nFoo(5)([1.0])\n```\n\n"


def _module():
    spec = importlib.util.spec_from_file_location("function_page", EXT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _render(text):
    source = [text]
    _module()._source_read(None, "functions_test/Foo", source)
    return source[0].split("\n---\n", 1)[1]  # drop the frontmatter


def _pos(text, needle):
    assert text.count(needle) == 1, needle
    return text.index(needle)


@pytest.mark.parametrize("tail", [
    DESCRIPTION + NAN + EXAMPLES,
    DESCRIPTION + EXAMPLES + NAN,
])
def test_examples_follow_the_header(tail):
    out = _render(FRONTMATTER + tail)
    order = [_pos(out, s) for s in (
        "Does foo to a series.",       # header: summary
        "Foo(window_size=20)",         # header: signature
        "## Examples",
        "## Description",
        "## Parameters",
        "<!-- NAN_FOOTNOTE_START -->\n## NaN handling",
    )]
    assert order == sorted(order)
    # The example's own content travels with its heading.
    assert _pos(out, "## Examples") < _pos(out, "Foo(5)([1.0])") < _pos(out, "## Description")


def test_page_without_examples_is_unchanged():
    text = FRONTMATTER + DESCRIPTION + NAN
    mod = _module()
    assert mod._examples_first(text[text.index("## Description"):]) == \
        text[text.index("## Description"):]
    out = _render(text)
    order = [_pos(out, s) for s in (
        "Does foo to a series.", "## Description", "## Parameters", "## NaN handling")]
    assert order == sorted(order)
