"""screamer.docs: the built-in documentation that agents and people query."""
import json
import re
import subprocess
import sys

import numpy as np
import pytest

import screamer
from screamer import docs


def _registry():
    return json.loads(docs._read("help.json"))


def test_every_entry_can_be_shown_and_found():
    for name, entry in _registry().items():
        shown = docs.show(name)
        assert shown["name"] == name
        assert shown["short"] == entry.get("short", "")
        # Searching a function's own name finds it first.
        assert docs.search(name, limit=1)[0]["name"] == name, name


def test_show_adds_signature_and_url():
    e = docs.show("RollingRSI")
    assert e["signature"] == "RollingRSI(window_size=14, method='wilder', start_policy='strict')"
    assert e["url"] == docs.DOCS_URL + "functions_rolling/RollingRSI.html"
    assert {p["name"] for p in e["parameters"]} == {"window_size", "method", "start_policy"}
    # A plain function has no parameter schema; its signature comes from Python.
    assert docs.show("position_test")["signature"].startswith("position_test(engine, target, *market")


def test_show_unknown_name_suggests():
    with pytest.raises(KeyError, match="RollingMean"):
        docs.show("RollingMaen")
    assert docs.show("rollingmean")["name"] == "RollingMean"


def test_search_ranks_by_meaning():
    assert docs.search("bollinger")[0]["name"] == "BollingerBands"
    assert docs.search("exponential moving average")[0]["name"] == "EwMean"
    assert {r["name"] for r in docs.search("p-value randomization", limit=2)} == {
        "position_test", "strategy_test"}
    assert all("significance" in r["topics"]
               for r in docs.search("test", topic="significance"))
    assert docs.search("zzqxv") == []


def test_topics_cover_every_indexed_function():
    groups = docs.topics()
    listed = {f for g in groups for t in g["topics"] for f in t["functions"]}
    indexed = {n for n, e in _registry().items() if e.get("index", True) and e.get("topics")}
    assert listed == indexed
    for g in groups:
        assert g["name"] and g["desc"] and g["topics"]


def test_skill_is_the_guide_with_frontmatter():
    s = docs.skill()
    assert s.startswith("---\nname: screamer\ndescription: ")
    assert s.endswith(docs.guide())


@pytest.mark.parametrize("args", [
    ["topics"], ["search", "order", "flow"], ["show", "RollingStd"], ["guide"], ["skill"],
    ["--json", "show", "RollingStd"], ["search", "kalman", "--json"],
])
def test_command_line(args):
    r = subprocess.run([sys.executable, "-m", "screamer.docs", *args],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip()
    if "--json" in args:
        json.loads(r.stdout)


def test_command_line_unknown_name_fails():
    r = subprocess.run([sys.executable, "-m", "screamer.docs", "show", "NoSuchThing"],
                       capture_output=True, text=True)
    assert r.returncode == 1 and "no function named" in r.stderr


def test_guide_python_examples_run():
    """The guide's Python examples run, so the guide follows the API."""
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.standard_normal(300) * 0.01))
    open_ = np.r_[close[0], close[:-1]]
    namespace = {
        "prices": close, "close": close, "open_": open_,
        "high": np.maximum(open_, close) * 1.001, "low": np.minimum(open_, close) * 0.999,
        "target": np.sign(np.r_[0.0, np.diff(close)]),
        "array_a": close, "array_b": open_,
    }
    blocks = re.findall(r"```python\n(.*?)```", docs.guide(), re.S)
    assert len(blocks) == 3
    for code in blocks:
        exec(compile(code, "<agent guide>", "exec"), namespace)
