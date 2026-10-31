"""Function pages link back to the Guide task pages that use them.

docs/_ext/function_page.py scans the code cells of the Guide task pages listed
in docs/guide.yml. A function counts as used on a page when a code cell imports
it from screamer or reads it from the screamer module; a name inside a longer
name, a comment or a string does not count.
"""
import importlib.util
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
EXT = REPO_ROOT / "docs" / "_ext" / "function_page.py"

# The release wheel-test CI job runs the suite against the installed wheel; skip
# there if the docs sources are absent.
pytestmark = pytest.mark.skipif(not EXT.exists(), reason="needs the docs sources")


def _module():
    spec = importlib.util.spec_from_file_location("function_page", EXT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _page(title, code):
    return ("---\nfile_format: mystnb\n---\n\n"
            f"# {title}\n\n```{{code-cell}} ipython3\n{code}\n```\n")


@pytest.fixture
def docs(tmp_path):
    (tmp_path / "guide" / "flow").mkdir(parents=True)
    (tmp_path / "guide.yml").write_text(
        "parts:\n- title: P\n  chapters:\n  - slug: flow\n    title: Flow\n    line: x\n"
        "    tasks:\n    - guide/flow/volume\n    - guide/flow/direction\n    deeper: []\n",
        encoding="utf-8")
    (tmp_path / "guide" / "flow" / "volume.md").write_text(_page(
        "Sign volume by trade direction",
        "from screamer import SignedVolume\n"
        "# Sign each trade, then sum the volume\n"
        "plt.title(\"Sign of the flow\")\n"
        "v = SignedVolume()(price, size)"), encoding="utf-8")
    (tmp_path / "guide" / "flow" / "direction.md").write_text(_page(
        "Take the direction of a slope",
        "from screamer import (RollingPoly1,\n    Sign)\n"
        "import screamer as sc\n"
        "d = Sign()(RollingPoly1(20)(x)[:, 1])\n"
        "z = sc.RollingZscore(20)(x)"), encoding="utf-8")
    return tmp_path


def test_whole_names_only(docs):
    usage = _module().guide_usage(docs)
    assert usage["SignedVolume"] == ["guide/flow/volume"]
    assert usage["Sign"] == ["guide/flow/direction"]
    assert usage["RollingPoly1"] == ["guide/flow/direction"]
    assert usage["RollingZscore"] == ["guide/flow/direction"]


def test_usage_follows_guide_order(docs):
    text = (docs / "guide" / "flow" / "volume.md").read_text(encoding="utf-8")
    (docs / "guide" / "flow" / "volume.md").write_text(
        text.replace("from screamer import SignedVolume",
                     "from screamer import SignedVolume, RollingPoly1"), encoding="utf-8")
    assert _module().guide_usage(docs)["RollingPoly1"] == [
        "guide/flow/volume", "guide/flow/direction"]


def test_page_lists_the_guide_pages(docs):
    mod = _module()
    mod.DOCS = docs
    source = ["---\nname: Sign\ntitle: Sign\nkind: functor\n---\n\n# `Sign`\n\n"
              "## Description\n\nThe sign.\n"]
    mod._source_read(None, "functions_math/Sign", source)
    out = source[0]
    assert "## Used in the Guide" in out
    assert "[Take the direction of a slope](../guide/flow/direction.md)" in out
    assert "volume.md" not in out


def test_unused_function_has_no_section(docs):
    mod = _module()
    mod.DOCS = docs
    source = ["---\nname: Exp\ntitle: Exp\nkind: functor\n---\n\n# `Exp`\n\n"
              "## Description\n\nThe exponential.\n"]
    mod._source_read(None, "functions_math/Exp", source)
    assert "Used in the Guide" not in source[0]


def test_changed_usage_rereads_the_function_page(docs):
    mod = _module()
    mod.DOCS = docs

    class Env:
        found_docs = {"functions_math/Sign", "functions_micro/SignedVolume",
                      "functions_rolling/RollingZscore", "guide/flow/volume"}

    env = Env()
    assert sorted(mod._guide_outdated(None, env, set(), set(), set())) == sorted(
        d for d in Env.found_docs if d.startswith("functions_"))
    assert mod._guide_outdated(None, env, set(), set(), set()) == []
    page = docs / "guide" / "flow" / "volume.md"
    page.write_text(page.read_text(encoding="utf-8").replace(
        "SignedVolume\n", "SignedVolume, RollingZscore\n"), encoding="utf-8")
    assert mod._guide_outdated(None, env, set(), set(), set()) == [
        "functions_rolling/RollingZscore"]
