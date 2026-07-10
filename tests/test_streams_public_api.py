import screamer
from screamer import streams


PUBLIC = [
    "merge", "combine_latest", "replay",
    "dropna", "filter", "select", "split",
]


def test_public_names_exported_from_package():
    for name in PUBLIC:
        assert hasattr(screamer, name), f"screamer.{name} missing"
        assert getattr(screamer, name) is getattr(streams, name)


def test_public_names_in_all():
    for name in PUBLIC:
        assert name in screamer.__all__


def test_internal_helpers_not_exported():
    # underscore helpers must stay private
    for name in ("_normalize_series", "_index_dtype_kind", "_make_merge_puller", "_run_chain"):
        assert not hasattr(screamer, name)
