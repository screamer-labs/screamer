import screamer
from screamer import streams


# The five lowercase functions were removed from the public API (API collapse).
# Only the CamelCase class forms are public.
PUBLIC_CLASSES = [
    "Merge", "CombineLatest", "Dropna", "Select", "Resample",
    "replay", "Filter", "split",
]

# The lowercase functions are internal implementation details (not public).
PRIVATE_FUNCTIONS = [
    "merge", "combine_latest", "dropna", "select", "resample",
]


def test_public_names_exported_from_package():
    for name in PUBLIC_CLASSES:
        assert hasattr(screamer, name), f"screamer.{name} missing"
        assert getattr(screamer, name) is getattr(streams, name)


def test_public_names_in_all():
    for name in PUBLIC_CLASSES:
        assert name in screamer.__all__


def test_lowercase_functions_not_in_public_api():
    # The lowercase function forms are no longer exported from the top-level package.
    for name in PRIVATE_FUNCTIONS:
        assert not hasattr(screamer, name), f"screamer.{name} should not be public"
        assert name not in screamer.__all__


def test_internal_helpers_not_exported():
    # underscore helpers must stay private
    for name in ("_normalize_series", "_index_dtype_kind", "_make_merge_puller", "_run_chain"):
        assert not hasattr(screamer, name)
