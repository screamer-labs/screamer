"""G25: a bool is not a number setting, in Python as in JavaScript.

Every number, integer and vector setting of every operator refuses True and
NumPy's True with TypeError, and so do the stream settings. The JavaScript
side reads the same table from wasm/smoke/null_oracle.json (accepts_bool).
"""
import json
import pathlib

import numpy as np
import pytest

import screamer as s

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURE = REPO_ROOT / "wasm" / "smoke" / "null_oracle.json"

pytestmark = pytest.mark.skipif(not FIXTURE.exists(), reason="needs the source tree")


def _settings():
    out = []
    for entry in json.loads(FIXTURE.read_text()):
        for name in entry["accepts_bool"]:
            out.append((entry["name"], entry["base"], name))
    return out


SETTINGS = _settings()


@pytest.mark.parametrize("flag", [True, np.True_], ids=["bool", "numpy.bool"])
def test_every_operator_setting_refuses_a_bool(flag):
    assert len(SETTINGS) > 450
    for op, base, name in SETTINGS:
        with pytest.raises(TypeError):
            getattr(s, op)(**dict(base, **{name: flag}))


def test_a_vector_setting_refuses_a_bool_element():
    for taps in ([True, 0.5], np.array([True, False])):
        with pytest.raises(TypeError, match="got an element of type"):
            s.MovingAverage(taps)
    with pytest.raises(TypeError):
        s.BacktestPortfolioTarget(fee=[0.1, True])
    s.MovingAverage([0.5, 0.5])
    s.BacktestPortfolioTarget(fee=[0.1, 0.2])


def test_numbers_still_convert():
    assert repr(s.RollingMean(np.int64(3))) == "RollingMean(window_size=np.int64(3))"
    s.BollingerBands(3, np.float32(2.0))
    s.BollingerBands(3, 2)
    s.EwMean(alpha=None, span=3)


@pytest.mark.parametrize("flag", [True, np.True_], ids=["bool", "numpy.bool"])
def test_stream_settings_refuse_a_bool(flag):
    v, k = np.arange(5.0), np.arange(5)
    x = s.Input("x")
    calls = [
        lambda: s.Resample(count=flag)(v),
        lambda: s.Resample(freq=flag)((v, k)),
        lambda: s.Resample(freq=2, origin=flag)((v, k)),
        lambda: s.Resample((v, k), (v, k), threshold=flag),
        lambda: s.Resample(freq=1, fill="nan", max_fill=flag)((v, k)),
        lambda: s.Delay(flag),
        lambda: s.Select(flag),
        lambda: s.split(v, [0, 1, 0, 1, 0], n=flag),
        lambda: s.forecast_pairs(v, v, count=flag),
        lambda: s.forecast_pairs((v, k), (v, k), duration=flag),
        lambda: s.Pipeline([x], [x], max_pending=flag),
    ]
    for call in calls:
        with pytest.raises(TypeError):
            call()
