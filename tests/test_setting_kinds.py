"""Task 31 fix round 1: settings documented as finite refuse an infinity and
NaN, and a setting of the wrong kind names the operator and the setting.
Mirrors js/test/setting_kinds.test.ts."""
import math

import numpy as np
import pytest

import screamer as s


@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
def test_finite_settings_refuse_infinity_and_nan(bad):
    with pytest.raises(ValueError, match="num_std must be a non-negative finite number"):
        s.BollingerBands(20, bad)
    with pytest.raises(ValueError, match="lower and upper must be finite"):
        s.SchmittTrigger(bad, 3.0)
    with pytest.raises(ValueError, match="lower and upper must be finite"):
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
