# tests/test_build_help_registry.py
from devtools.build_help_registry import parse_help_file_text


def test_parser_is_importable():
    assert callable(parse_help_file_text)
