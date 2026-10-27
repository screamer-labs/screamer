"""Gate for the Guide: registry, generated files, and task page shape.

devtools/build_guide.py builds the Guide landing pages and the home table of
contents from docs/guide.yml. These tests fail when a notebook or recipe is not
reachable from the Guide, a generated file is stale, or a task page has the
wrong shape.
"""
import copy
import importlib.util
import pathlib
import re
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "devtools" / "build_guide.py"
DOCS = REPO_ROOT / "docs"
SPEC = DOCS / "guide.yml"

# The release wheel-test CI job runs the suite against the installed wheel; skip
# there if the docs sources are absent.
pytestmark = pytest.mark.skipif(not SPEC.exists(), reason="needs the docs sources")


def _module():
    spec = importlib.util.spec_from_file_location("build_guide", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _chapters(mod):
    return [c for p in mod.load()["parts"] for c in p["chapters"]]


def _task_pages(mod):
    return [t for c in _chapters(mod) for t in c.get("tasks", []) if t.startswith("guide/")]


def test_guide_is_fresh():
    r = subprocess.run([sys.executable, str(SCRIPT), "--check"], cwd=REPO_ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0, (
        f"build_guide.py --check failed:\n{r.stdout}{r.stderr}\n"
        "Update docs/guide.yml if needed, re-run devtools/build_guide.py, and commit the result."
    )


def test_every_notebook_and_recipe_is_in_one_chapter():
    mod = _module()
    listed = [d for c in _chapters(mod) for d in c.get("tasks", []) + c.get("deeper", [])]
    expected = {f"notebooks/{p.stem}" for p in (DOCS / "notebooks").glob("*.ipynb")}
    for p in (DOCS / "recipes").glob("*.md"):
        if "orphan: true" not in p.read_text():
            expected.add(f"recipes/{p.stem}")
    for doc in expected:
        assert listed.count(doc) == 1, f"{doc} appears {listed.count(doc)} times in guide.yml"


def test_task_titles_match_pages():
    mod = _module()
    for doc in _task_pages(mod):
        assert mod.page_title(doc), doc
    for c in _chapters(mod):
        landing = DOCS / "guide" / c["slug"] / "index.md"
        assert mod.page_title(f"guide/{c['slug']}/index") == c["title"], landing


def test_task_page_shape():
    mod = _module()
    for doc in _task_pages(mod):
        text = (DOCS / f"{doc}.md").read_text()
        m = re.search(r"^## Go deeper\n(.*)", text, re.M | re.S)
        assert m and re.search(r"\]\([^)]+\)", m.group(1)), f"{doc}: no Go deeper links"
        front = re.match(r"---\n(.*?)\n---\n", text, re.S)
        if front and re.search(r"^exempt_shape:\s*true\s*$", front.group(1), re.M):
            continue  # a prose page, such as installation, has no example to run
        assert "file_format: mystnb" in text, doc
        cells = re.findall(r"```\{code-cell\}[^\n]*\n(.*?)```", text, re.S)
        assert cells, f"{doc}: no code-cell"
        lines = sum(len([l for l in c.splitlines() if l.strip() and not l.startswith(":")])
                    for c in cells)
        assert lines <= 35, f"{doc}: {lines} code lines"


def test_check_reports_unlisted_notebook():
    mod = _module()
    spec = copy.deepcopy(mod.load())
    dropped = "notebooks/01-quickstart-polymorphic-api"
    for c in (c for p in spec["parts"] for c in p["chapters"]):
        c["deeper"] = [d for d in c.get("deeper", []) if d != dropped]
    problems = mod.problems(spec)
    assert problems and any(dropped in p for p in problems)
