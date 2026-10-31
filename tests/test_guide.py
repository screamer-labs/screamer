"""Gate for the Guide: registry, generated files, and task page shape.

devtools/build_guide.py builds the Guide landing pages and the home table of
contents from docs/guide.yml. These tests fail when a notebook or recipe is not
reachable from the Guide, a generated file is stale, a task page has the wrong
shape, a function outside the allow-list is used on no Guide task page, or an
allow-listed function is used on one.
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
FUNCTION_PAGE = DOCS / "_ext" / "function_page.py"

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
        if "orphan: true" not in p.read_text(encoding="utf-8"):
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
        text = (DOCS / f"{doc}.md").read_text(encoding="utf-8")
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


def test_check_reports_unlisted_notebook(capsys):
    mod = _module()
    spec = copy.deepcopy(mod.load())
    dropped = "notebooks/01-quickstart-polymorphic-api"
    for c in (c for p in spec["parts"] for c in p["chapters"]):
        c["deeper"] = [d for d in c.get("deeper", []) if d != dropped]
    problems = mod.problems(spec)
    assert problems and any(dropped in p for p in problems)
    mod.load = lambda: spec  # main() reads the registry through load()
    assert mod.main(["--check"]) == 1
    assert dropped in capsys.readouterr().out


# Pages that moved in Phases 1 and 2. Each old URL must keep working through a
# stub.
MOVED = [
    "installation", "capabilities",
    "recipes/pandas-ohlcv-indicators", "recipes/migrating", "recipes/ticks-to-backtest",
    "recipes/volatility-sizing", "recipes/first-strategy",
    "recipes/pairs-trade", "recipes/cross-sectional-momentum", "recipes/funding-carry",
    "recipes/stops-and-exits", "recipes/multi-asset", "choosing_an_operator",
    # Phase 2
    "recipes/significance-test", "recipes/walk-forward", "recipes/parameter-grid",
    "recipes/ml-features", "recipes/live-loop", "recipes/strategy-to-live",
]

_REFRESH = re.compile(r'"http-equiv=refresh":\s*"\d+;\s*url=([^"]+)"')


def _stubs():
    for path in DOCS.rglob("*.md"):
        if "_build" in path.parts or "superpowers" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        front = re.match(r"---\n(.*?)\n---\n", text, re.S)
        if front and re.search(r"^orphan:\s*true\s*$", front.group(1), re.M):
            m = _REFRESH.search(front.group(1))
            if m:
                yield path, m.group(1)


def test_moved_pages_redirect():
    stubs = dict(_stubs())
    for doc in MOVED:
        assert DOCS / f"{doc}.md" in stubs, f"{doc}.md is not a redirect stub"
    for path, url in stubs.items():
        target = (path.parent / url.split("#")[0]).resolve()
        assert target.suffix == ".html", f"{path}: {url}"
        sources = [target.with_suffix(s) for s in (".md", ".ipynb", ".rst")]
        assert any(s.exists() for s in sources), f"{path.relative_to(DOCS)} -> {url}: no source page"
        assert target.with_suffix(".md") not in stubs, f"{path.relative_to(DOCS)}: redirect chain"


# Functions allowed on no Guide task page. The gate is a ratchet: a function
# outside these sets must appear in a Guide code cell, and a listed function that
# a page starts to use must leave the list, so the list only shrinks.
#
# Elementwise math and logic helpers: any page can use them, no task is about
# them. Cart2Polar and Polar2Cart are elementwise coordinate conversions and
# Identity passes its input through, so they belong here too.
MATH_HELPERS = {
    "Acos", "Add", "And", "Asin", "Atan", "Atan2", "Cart2Polar", "Ceil", "Cos",
    "Cube", "Elu", "Equal", "Erf", "Erfc", "Exp", "Floor", "GreaterEqual",
    "GreaterThan", "Hypot", "Identity", "IsFinite", "IsNan", "LessEqual",
    "LessThan", "Linear", "Linear2", "Mul", "Not", "NotEqual", "Or", "Polar2Cart",
    "Power", "Relu", "Selu", "Sigmoid", "Sin", "Softsign", "Sqrt", "Square", "Tanh",
    "Where",
}

# Functions still waiting for a Guide page (snapshot at the end of Phase 3).
# Remove a name when a page starts to use it; never add one.
NOT_YET_IN_GUIDE = {
    # functions_cross
    "CrossTopBottom",
    # functions_ew
    "EwAlpha", "EwCorr", "EwCov", "EwGarmanKlassVar", "EwGarmanKlassVol", "EwKurt",
    "EwParkinsonVar", "EwParkinsonVol", "EwResidualStd", "EwRms",
    "EwRogersSatchellVar", "EwRogersSatchellVol", "EwSkew", "EwSpread", "EwVar",
    "EwYangZhangVar", "EwYangZhangVol", "EwZscore",
    # functions_expanding
    "ExpandingKurt", "ExpandingMean", "ExpandingProd", "ExpandingSkew",
    "ExpandingSlope", "ExpandingStd", "ExpandingSum", "ExpandingVar",
    # functions_fin
    "BacktestL1TradesOrders", "BacktestPortfolioTarget", "BacktestReport",
    "BacktestStats", "BayesianRegression", "PortfolioReport",
    "RollingCov", "RollingHitRate", "RollingLinearRegression", "RollingPercentile",
    "RollingRank", "RollingResidualStd", "RollingSpread", "RollingTSF",
    "RollingUpsideDeviation", "backtest_trades",
    # functions_micro
    "BulkVolumeClassifier", "ContOFI", "EwKyleLambda", "LeeReadySign", "Propagator",
    "RollingOrderImbalance",
    # functions_misc
    "CumMax", "CumMin", "CumProd", "Detrend", "Diff2", "Last", "Momentum",
    # functions_rolling
    "ADOSC", "ADXR", "BOP", "DEMA", "DX", "ImpulseClip", "MinusDI",
    "MinusDM", "NATR", "PlusDM", "RollingArgmax", "RollingArgmin",
    "RollingGarmanKlassVar", "RollingGarmanKlassVol", "RollingIqr", "RollingKurt",
    "RollingMax", "RollingMin", "RollingMinMax", "RollingParkinsonVar",
    "RollingParkinsonVol", "RollingPoly2", "RollingQuantile", "RollingRange",
    "RollingRealizedKurt", "RollingRealizedSkew", "RollingRms",
    "RollingRogersSatchellVar", "RollingRogersSatchellVol", "RollingSeasonalMean",
    "RollingSeasonalStd", "RollingSkew", "RollingVar", "RollingYangZhangVar",
    "TEMA", "TRIMA", "TrueRange", "WMA",
    # functions_signal
    "ButterBandpass", "ButterBandstop", "ButterHighpass", "CycleAmplitude",
    "CycleFrequency", "CyclePhase", "Decycler", "HilbertPhasor", "MovingAverage",
    # functions_streams
    "Delay", "Dropna", "Filter", "Merge", "from_pandas", "reshape", "split",
}

NOT_IN_GUIDE = MATH_HELPERS | NOT_YET_IN_GUIDE


def _function_pages():
    """Name of every function page; pages of kind ``guide`` are prose."""
    import yaml
    names = set()
    for path in DOCS.glob("functions_*/*.md"):
        m = re.match(r"---\n(.*?)\n---\n", path.read_text(encoding="utf-8"), re.S)
        fm = yaml.safe_load(m.group(1)) if m else None
        if fm and "name" in fm and fm.get("kind") != "guide":
            names.add(fm["name"])
    return names


def _guide_usage():
    spec = importlib.util.spec_from_file_location("function_page", FUNCTION_PAGE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.guide_usage(DOCS)


def test_not_in_guide_names_are_functions():
    assert not MATH_HELPERS & NOT_YET_IN_GUIDE
    stale = sorted(NOT_IN_GUIDE - _function_pages())
    assert not stale, f"NOT_IN_GUIDE lists names without a function page: {stale}"


def test_guide_covers_functions():
    gap = sorted(_function_pages() - NOT_IN_GUIDE - set(_guide_usage()))
    assert not gap, (
        f"{len(gap)} functions are used on no Guide task page. Use each in a task "
        f"page's code cell:\n" + "\n".join(gap))


def test_allow_list_only_shrinks():
    covered = sorted(NOT_IN_GUIDE & set(_guide_usage()))
    assert not covered, (
        f"A Guide page now uses these functions; remove them from MATH_HELPERS or "
        f"NOT_YET_IN_GUIDE in tests/test_guide.py:\n" + "\n".join(covered))
