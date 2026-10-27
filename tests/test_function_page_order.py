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


SOURCE = ["import numpy as np", "from screamer import Foo", "",
          "y = Foo(5)(np.arange(10.0))", "fig.show()"]


def _plotly(include_source):
    opt = f"    :include-source: {include_source}\n" if include_source else ""
    code = "".join(f"    {line}\n" if line else "\n" for line in SOURCE)
    return ".. plotly::\n" + opt + "\n" + code


def _fence(*parts):
    return "```{eval-rst}\n" + "\n".join(parts) + "```\n\n"


def _code_block_source(text):
    """Source lines of the first ``.. code-block:: python`` in ``text``."""
    after = text.split(".. code-block:: python\n\n", 1)[1]
    lines = []
    for line in after.split("\n"):
        if line and not line.startswith("    "):
            break
        lines.append(line[4:])
    while lines and not lines[-1]:
        lines.pop()
    return lines


@pytest.mark.parametrize("include_source", ["True", None])
def test_plot_comes_first_then_its_full_source(include_source):
    # A plot with no :include-source: option is treated like True.
    examples = "## Examples\n\n### Usage plot\n\n" + _fence(_plotly(include_source))
    out = _render(FRONTMATTER + DESCRIPTION + examples)
    section = out[out.index("## Examples"):out.index("## Description")]
    assert ":include-source: True" not in section
    assert _pos(section, ":include-source: False") < _pos(section, ".. code-block:: python")
    assert _code_block_source(section) == SOURCE
    # Both stay inside the one eval-rst fence, which still closes once.
    assert section.count("```{eval-rst}") == 1 and section.count("```\n") == 1


def test_every_plot_in_examples_is_split_and_prose_stays_in_place():
    examples = "## Examples\n\n" + _fence(_plotly("True"), "Some prose.\n", _plotly("True"))
    out = _render(FRONTMATTER + DESCRIPTION + examples)
    section = out[out.index("## Examples"):out.index("## Description")]
    assert section.count(":include-source: False") == 2
    assert section.count(".. code-block:: python") == 2
    first, second = section.split("Some prose.")
    assert _code_block_source(first) == SOURCE == _code_block_source(second)


@pytest.mark.parametrize("tail", [
    # A plot outside Examples keeps its inline source.
    "## Description\n\n" + _fence(_plotly("True")) + EXAMPLES,
    # A plot in Examples that already hides its source stays as it is.
    DESCRIPTION + "## Examples\n\n" + _fence(_plotly("False")),
])
def test_other_plotly_blocks_are_untouched(tail):
    out = _render(FRONTMATTER + tail)
    assert ".. code-block:: python" not in out
    assert out.count(".. plotly::") == 1
    for opt in (":include-source: True", ":include-source: False"):
        assert out.count(opt) == tail.count(opt)


def test_heading_inside_a_code_fence_is_not_a_section():
    fenced = "```text\n## Examples\nnot a heading\n```\n\n"
    out = _render(FRONTMATTER + DESCRIPTION + fenced + EXAMPLES)
    # The real Examples section moves first; the fenced copy stays in Description.
    assert out.index("## Examples\n\n### Usage example") < out.index("## Description")
    assert out.index("## Description") < out.index("```text\n## Examples")
