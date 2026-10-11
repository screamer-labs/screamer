"""Objects with unusual class metadata must raise, never crash the process.

Each case runs in a subprocess, so a segmentation fault fails the test instead
of killing pytest.
"""
import subprocess
import sys
import textwrap

import pytest

# Class headers that define a class M whose type metadata is hostile.
HEADERS = {
    "module_none": "class M:\n    __module__ = None\n",
    "module_int": "class M:\n    __module__ = 5\n",
    "module_bytes": "class M:\n    __module__ = b'x'\n",
    "module_surrogate": "class M:\n    __module__ = '\\udc80'\n",
    "module_raises": (
        "class Meta(type):\n"
        "    @property\n"
        "    def __module__(cls):\n"
        "        raise RuntimeError('boom')\n"
        "class M(metaclass=Meta):\n"
        "    pass\n"
    ),
}

CALLS = [
    "s.RollingMean(2)(M())",
    "s.Add()(M(), M())",
    "s.Add()([1.0, 2.0], M())",
    "s.RollingCorr(3)(M(), [1.0])",
    "s.RollingMinMax(3)(M())",
    "s.CrossRank()(M())",
    "s.PortfolioReport()(M())",
]


def _run(header, call):
    code = "import screamer as s\n" + header + textwrap.dedent(f"""
        try:
            {call}
        except Exception as e:
            print(type(e).__name__)
            print(e)
        else:
            print("no error")
    """)
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert p.returncode == 0, f"rc={p.returncode}\n{p.stderr}"
    return p.stdout.splitlines()


@pytest.mark.parametrize("call", CALLS)
@pytest.mark.parametrize("case", sorted(HEADERS))
def test_hostile_module_raises_type_error(case, call):
    out = _run(HEADERS[case], call)
    assert out[0] in ("TypeError", "ValueError"), out
    if out[0] == "TypeError" and "Rank" not in call and "Portfolio" not in call:
        # Operator dispatch names the offending type.
        assert "M" in out[1], out


@pytest.mark.parametrize("case", sorted(HEADERS))
def test_rejected_str_subclass_with_hostile_module(case):
    # A str subclass is rejected by name before any routing.
    header = HEADERS[case].replace("class M:", "class M(str):").replace(
        "class M(metaclass=Meta):", "class M(str, metaclass=Meta):")
    for call in ["s.RollingMean(2)(M('a'))", "s.Add()(M('a'), 1.0)", "s.RollingMinMax(3)(M('a'))"]:
        out = _run(header, call)
        assert out[0] == "TypeError" and " a M " in out[1], (call, out)


def test_mixed_inputs_name_their_types():
    out = _run("", "s.Add()(1.0, [1.0])")
    assert out == ["TypeError", "Add: cannot combine inputs of types (float, list); "
                   "pass all numbers, all arrays, or all sequences"]


# nanobind v2.9.2 formats __module__ unchecked when it builds its own messages
# and when a Python class subclasses a bound type. Fixed upstream in v2.13.0
# (commits 7f6ad04, 0a58515); these pass once the pinned tag is bumped.
@pytest.mark.xfail(strict=True, reason="nanobind < 2.13 crashes on a non-str __module__")
@pytest.mark.parametrize("call", [
    "s.RollingMean(2)(M(), M())",                                   # overload error message
    "type('X', (s.RollingMean,), {'__module__': None})(2)",          # subclass a bound type
])
def test_nanobind_internal_hostile_module(call):
    out = _run(HEADERS["module_none"], call)
    assert out[0] in ("TypeError", "ValueError", "no error"), out
