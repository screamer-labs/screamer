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
