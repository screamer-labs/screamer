# tests/test_build_help_registry.py
import textwrap

from devtools.build_help_registry import parse_help_file_text


def _md(text: str) -> str:
    """Helper: dedent and trim leading newline for inline markdown strings."""
    return textwrap.dedent(text).lstrip("\n")


def test_parser_is_importable():
    assert callable(parse_help_file_text)


def test_minimal_file_no_examples():
    text = _md('''
        ---
        name: Foo
        title: Foo function
        short: A foo.
        inputs: 1
        outputs: 1
        parameters: []
        ---

        # `Foo`

        ## Description

        `Foo` does the foo thing.

        <!-- HELP_END -->

        ## Reference

        Sphinx-only stuff.
    ''')
    entry = parse_help_file_text(text)
    assert entry["name"] == "Foo"
    assert entry["title"] == "Foo function"
    assert entry["short"] == "A foo."
    assert entry["examples"] == []
    assert entry["details"].startswith("## Description")
    assert "`Foo` does the foo thing." in entry["details"]
    assert "Sphinx-only stuff." not in entry["details"]
    assert "body_markdown" not in entry


def test_returns_none_without_frontmatter():
    assert parse_help_file_text("just some markdown\n") is None


def test_returns_none_without_name_field():
    text = _md('''
        ---
        title: Untitled
        ---

        body
    ''')
    assert parse_help_file_text(text) is None


def test_single_plain_python_example():
    text = _md('''
        ---
        name: Foo
        ---

        # `Foo`

        ## Description

        Prose.

        ## Examples

        ### Basic usage

        ```python
        from screamer import Foo
        Foo()(arr)
        ```

        <!-- HELP_END -->
    ''')
    entry = parse_help_file_text(text)
    assert entry["examples"] == [
        {
            "language": "python",
            "caption": "Basic usage",
            "code": "from screamer import Foo\nFoo()(arr)",
        }
    ]
    assert "## Examples" not in entry["details"]
    assert "```" not in entry["details"]


def test_non_python_language_tag_preserved():
    text = _md('''
        ---
        name: Foo
        ---

        # `Foo`

        ## Examples

        ### Shell snippet

        ```bash
        echo hi
        ```

        <!-- HELP_END -->
    ''')
    entry = parse_help_file_text(text)
    assert entry["examples"][0]["language"] == "bash"
    assert entry["examples"][0]["code"] == "echo hi"


def test_plotly_eval_rst_is_unwrapped():
    text = _md('''
        ---
        name: Foo
        ---

        # `Foo`

        ## Examples

        ### Plotly demo

        ```{eval-rst}
        .. plotly::
            :include-source: True

            import numpy as np
            from screamer import Foo
            arr = np.arange(5)
            print(Foo()(arr))
        ```

        <!-- HELP_END -->
    ''')
    entry = parse_help_file_text(text)
    assert entry["examples"] == [
        {
            "language": "python",
            "caption": "Plotly demo",
            "code": (
                "import numpy as np\n"
                "from screamer import Foo\n"
                "arr = np.arange(5)\n"
                "print(Foo()(arr))"
            ),
        }
    ]


def test_eval_rst_without_plotly_directive_is_rejected():
    text = _md('''
        ---
        name: Foo
        ---

        # `Foo`

        ## Examples

        ### Other directive

        ```{eval-rst}
        .. note:: not supported here
        ```

        <!-- HELP_END -->
    ''')
    try:
        parse_help_file_text(text)
    except ValueError as e:
        assert "eval-rst" in str(e) or "plotly" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_multiple_examples_in_order():
    text = _md('''
        ---
        name: Foo
        ---

        # `Foo`

        ## Examples

        ### First

        ```python
        a = 1
        ```

        ### Second

        ```python
        b = 2
        ```

        ### Third

        ```python
        c = 3
        ```

        <!-- HELP_END -->
    ''')
    entry = parse_help_file_text(text)
    captions = [e["caption"] for e in entry["examples"]]
    codes = [e["code"] for e in entry["examples"]]
    assert captions == ["First", "Second", "Third"]
    assert codes == ["a = 1", "b = 2", "c = 3"]


def test_code_fence_in_details_is_rejected():
    text = _md('''
        ---
        name: Foo
        ---

        # `Foo`

        ## Description

        Inline example before Examples section:

        ```python
        oops()
        ```

        <!-- HELP_END -->
    ''')
    try:
        parse_help_file_text(text)
    except ValueError as e:
        assert "Examples" in str(e)
    else:
        raise AssertionError("expected ValueError")
