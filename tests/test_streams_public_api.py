import screamer
from screamer import streams


PUBLIC = [
    "merge", "merge_iter", "combine_latest", "combine_latest_iter", "pace",
    "dropna", "filter", "split", "dropna_iter", "filter_iter",
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
    for name in ("_normalize_series", "_key_dtype_kind", "_make_merge_puller"):
        assert not hasattr(screamer, name)
