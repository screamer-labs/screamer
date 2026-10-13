"""Task 31 fix round 1: settings documented as finite refuse an infinity and
NaN, and a setting of the wrong kind names the operator and the setting.
Mirrors js/test/setting_kinds.test.ts."""
import math

import numpy as np
import pytest

import screamer as s


@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
def test_finite_settings_refuse_infinity_and_nan(bad):
    got = {math.inf: "inf", -math.inf: "-inf"}.get(bad, "nan")
    with pytest.raises(ValueError, match=f"^BollingerBands: num_std must be finite, got {got}$"):
        s.BollingerBands(20, bad)
    with pytest.raises(ValueError, match=f"^SchmittTrigger: lower must be finite, got {got}$"):
        s.SchmittTrigger(bad, 3.0)
    with pytest.raises(ValueError, match=f"^SchmittTrigger: upper must be finite, got {got}$"):
        s.SchmittTrigger(-3.0, bad)
    s.BollingerBands(20, 0.0)


def test_wrong_kinds_name_the_operator_and_the_setting():
    cases = [
        (lambda: s.RollingMean(2.5), "RollingMean: window_size must be an integer, got 2.5"),
        (lambda: s.RollingMean(np.float64(3.0)), "RollingMean: window_size must be an integer, got 3.0"),
        (lambda: s.BollingerBands(20, "2"), "BollingerBands: num_std must be a number, got str"),
        (lambda: s.RollingMean(20, 3), "RollingMean: start_policy must be a string, got int"),
        (lambda: s.RollingMean(None), "RollingMean: window_size must be an integer, got None"),
    ]
    for call, msg in cases:
        with pytest.raises(TypeError) as e:
            call()
        assert str(e.value) == msg
    # More arguments than settings keep nanobind's message.
    with pytest.raises(TypeError, match="incompatible function arguments"):
        s.RollingMean(3, "strict", 1)


# ---- N1: a setting that is not finite: one message form ------------------

import json as _json                                                  # noqa: E402
import pathlib as _pathlib                                            # noqa: E402

from screamer import screamer_bindings as _b                          # noqa: E402

_ROOT = _pathlib.Path(__file__).resolve().parent.parent
_ORACLE = _ROOT / "wasm" / "smoke" / "null_oracle.json"
_HELP = _ROOT / "screamer" / "data" / "help.json"
_FLOATS = ("float", "float|null", "float|list[float]", "float[]")
_EW = {"com", "span", "halflife", "alpha"}
_TEXT = {math.inf: "inf", -math.inf: "-inf"}


def _float_settings():
    types = {op: {p["name"]: p["type"] for p in e.get("parameters", [])}
             for op, e in _json.loads(_HELP.read_text()).items()}
    out = []
    for entry in _json.loads(_ORACLE.read_text()):
        for name in entry["accepts_bool"]:
            kind = types[entry["name"]][name]
            if kind in _FLOATS:
                base = dict(entry["base"])
                if name in _EW:                 # one decay setting at a time
                    base = {k: v for k, v in base.items() if k not in _EW}
                out.append((entry["name"], name, base, kind == "float[]"))
    return out


FLOAT_SETTINGS = _float_settings()


@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan], ids=["inf", "-inf", "nan"])
def test_a_setting_that_is_not_finite_has_one_message(bad):
    assert len(FLOAT_SETTINGS) > 250
    refused = 0
    for op, name, base, vector in FLOAT_SETTINGS:
        value = [bad] if vector else bad
        expected = _b._finite_setting_error(op, [name], [[bad]])
        try:
            getattr(s, op)(**dict(base, **{name: value}))
        except ValueError as e:
            refused += 1
            assert expected, f"{op}({name}={bad}) is refused but finite_settings.h lets it pass: {e}"
            assert str(e) == expected, (op, name)
            assert str(e).startswith(f"{op}: {name} must be finite")
            assert str(e).endswith(f", got {_TEXT.get(bad, 'nan')}")
        else:
            assert not expected, f"{op}({name}={bad}) is taken but finite_settings.h refuses it"
    assert refused > 200


def test_a_list_for_one_number_is_a_wrong_kind():
    with pytest.raises(TypeError, match=r"^RollingMean: window_size must be an integer, got list$"):
        s.RollingMean([3])
    with pytest.raises(TypeError, match=r"^BollingerBands: num_std must be a number, got ndarray$"):
        s.BollingerBands(20, np.array([2.0]))


def test_required_settings_show_no_default():
    assert s.Hold.__doc__.splitlines()[0] == "Hold(n, release=0.0)"
    assert s.SchmittTrigger.__doc__.splitlines()[0] == "SchmittTrigger(lower, upper, initial=0.0)"
    assert "n : int, required" in s.Hold.__doc__
