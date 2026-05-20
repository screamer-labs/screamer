# tests/test_migrate_docs_v2.py
import textwrap

from devtools.migrate_docs_v2 import migrate_text


def _md(text: str) -> str:
    return textwrap.dedent(text).lstrip("\n")


def test_already_migrated_is_noop():
    """A file that already has `## Examples` above HELP_END and no other code is unchanged."""
    text = _md('''
        ---
        name: Foo
        ---

        # `Foo`

        ## Description

        Prose only.

        ## Examples

        ### Basic

        ```python
        Foo()
        ```

        <!-- HELP_END -->

        ## Reference

        Refs.
    ''')
    assert migrate_text(text) == text


def test_no_code_anywhere_is_noop():
    """A file with no code at all is unchanged."""
    text = _md('''
        ---
        name: Foo
        ---

        # `Foo`

        ## Description

        Prose only.

        <!-- HELP_END -->

        ## Reference

        Refs.
    ''')
    assert migrate_text(text) == text


def test_post_help_end_usage_example_is_moved():
    """The 85-file case: only a `## Usage Example*` section sits below HELP_END."""
    before = _md('''
        ---
        name: Foo
        ---

        # `Foo`

        ## Description

        Prose.

        <!-- HELP_END -->

        ## Usage Example and Plot

        ```{eval-rst}
        .. plotly::
            :include-source: True

            from screamer import Foo
            print(Foo()(arr))
        ```

        ## Reference

        Refs.
    ''')
    after = _md('''
        ---
        name: Foo
        ---

        # `Foo`

        ## Description

        Prose.

        ## Examples

        ### Usage example

        ```{eval-rst}
        .. plotly::
            :include-source: True

            from screamer import Foo
            print(Foo()(arr))
        ```

        <!-- HELP_END -->

        ## Reference

        Refs.
    ''')
    assert migrate_text(before) == after
    # Idempotency: applying migration to the result is a no-op.
    assert migrate_text(after) == after
