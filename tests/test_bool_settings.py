"""G25: a bool is not a number setting, in Python as in JavaScript.

Every number, integer, string and vector setting of every operator refuses
True, NumPy's True and a 0-d bool array with TypeError
"<Op>: <setting> must be <a number | an integer | ...>, got bool", and so do
the stream settings and the number arguments of the validation and backtest
functions. The JavaScript side reads the same table, with the messages, from
wasm/smoke/null_oracle.json (accepts_bool, bool_message).
"""
import importlib.resources
import json
import pathlib

import numpy as np
import pytest

import screamer as s

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURE = REPO_ROOT / "wasm" / "smoke" / "null_oracle.json"
HELP = importlib.resources.files("screamer").joinpath("data/help.json")   # the installed package

pytestmark = pytest.mark.skipif(not FIXTURE.exists(), reason="needs the source tree")

FLAGS = [True, np.True_, np.array(True), np.array(False)]
FLAG_IDS = ["bool", "numpy.bool", "0-d bool array", "0-d false array"]
WANTS = {"int": "an integer", "float": "a number", "float|null": "a number", "str": "a string",
         "float|list[float]": "a number or a sequence of numbers",
         "float[]": "a number or a sequence of numbers"}


def _settings():
    types = {op: {p["name"]: p["type"] for p in e.get("parameters", [])}
             for op, e in json.loads(HELP.read_text()).items()}
    out = []
    for entry in json.loads(FIXTURE.read_text()):
        for name in entry["accepts_bool"]:
            out.append((entry["name"], entry["base"], name, types[entry["name"]][name]))
    return out


SETTINGS = _settings()


@pytest.mark.parametrize("flag", FLAGS, ids=FLAG_IDS)
def test_every_operator_setting_refuses_a_bool(flag):
    assert len(SETTINGS) > 450
    for op, base, name, kind in SETTINGS:
        msg = f"{op}: {name} must be {WANTS[kind]}, got bool"
        with pytest.raises(TypeError) as e:
            getattr(s, op)(**dict(base, **{name: flag}))
        assert str(e.value) == msg


def test_the_oracle_holds_the_messages():
    for entry in json.loads(FIXTURE.read_text()):
        for name, message in entry["bool_message"].items():
            assert message.startswith(f"{entry['name']}: {name} must be ")
            assert message.endswith(", got bool")


def test_positional_settings_name_the_setting():
    with pytest.raises(TypeError, match=r"^RollingMean: window_size must be an integer, got bool$"):
        s.RollingMean(True)
    with pytest.raises(TypeError, match=r"^BollingerBands: num_std must be a number, got bool$"):
        s.BollingerBands(20, np.array(True))
    with pytest.raises(TypeError, match=r"^RollingMean: window_size must be an integer, got str$"):
        s.RollingMean("3")
    with pytest.raises(TypeError, match=r"^RollingMean: window_size must be an integer, got 2.5$"):
        s.RollingMean(2.5)


def test_a_vector_setting_refuses_a_bool_element():
    msg = r"^MovingAverage: taps must be a number or a sequence of numbers, got bool$"
    for taps in ([True, 0.5], np.array([True, False])):
        with pytest.raises(TypeError, match=msg):
            s.MovingAverage(taps)
    with pytest.raises(TypeError, match=r"^BacktestPortfolioTarget: fee must be a number or a "
                                        r"sequence of numbers, got bool$"):
        s.BacktestPortfolioTarget(fee=[0.1, True])
    s.MovingAverage([0.5, 0.5])
    s.BacktestPortfolioTarget(fee=[0.1, 0.2])


def test_numbers_still_convert():
    assert repr(s.RollingMean(np.int64(3))) == "RollingMean(window_size=np.int64(3))"
    s.RollingMean(np.array(3))
    s.BollingerBands(3, np.float32(2.0))
    s.BollingerBands(3, np.array(2.0))
    s.BollingerBands(3, 2)
    s.EwMean(alpha=None, span=3)


@pytest.mark.parametrize("flag", FLAGS, ids=FLAG_IDS)
def test_stream_settings_refuse_a_bool(flag):
    v, k = np.arange(5.0), np.arange(5)
    x = s.Input("x")
    calls = [
        ("resample: count must be an integer", lambda: s.Resample(count=flag)(v)),
        ("resample: freq must be an integer", lambda: s.Resample(freq=flag)((v, k))),
        ("resample: origin must be an integer", lambda: s.Resample(freq=2, origin=flag)((v, k))),
        ("resample: threshold must be a number", lambda: s.Resample((v, k), (v, k), threshold=flag)),
        ("resample: max_fill must be an integer",
         lambda: s.Resample(freq=1, fill="nan", max_fill=flag)((v, k))),
        ("Delay: duration must be an integer", lambda: s.Delay(flag)),
        ("select: columns must be an integer", lambda: s.Select(flag)),
        ("split: n must be an integer", lambda: s.split(v, [0, 1, 0, 1, 0], n=flag)),
        ("forecast_pairs: count must be an integer", lambda: s.forecast_pairs(v, v, count=flag)),
        ("forecast_pairs: duration must be an integer",
         lambda: s.forecast_pairs((v, k), (v, k), duration=flag)),
        ("Pipeline: max_pending must be an integer", lambda: s.Pipeline([x], [x], max_pending=flag)),
    ]
    for msg, call in calls:
        with pytest.raises(TypeError, match=f"^{msg}, got bool$"):
            call()


@pytest.mark.parametrize("flag", FLAGS, ids=FLAG_IDS)
def test_function_number_arguments_refuse_a_bool(flag):
    pnl = np.random.default_rng(0).standard_normal(50)
    x = np.sign(pnl)
    p = np.cumsum(pnl) + 100
    out = s.BacktestPriceTarget()(x, p)
    calls = [
        ("probabilistic_sharpe: benchmark must be a number", lambda: s.probabilistic_sharpe(pnl, flag)),
        ("circular_shift: n must be an integer", lambda: s.circular_shift(x, flag)),
        ("circular_shift: seed must be an integer", lambda: s.circular_shift(x, 3, seed=flag)),
        ("circular_shift: min_shift must be an integer",
         lambda: s.circular_shift(x, 3, min_shift=flag)),
        ("segment_shuffle: n must be an integer", lambda: s.segment_shuffle(x, flag)),
        ("permute_returns: n must be an integer", lambda: s.permute_returns(p, flag)),
        ("permute_bars: seed must be an integer", lambda: s.permute_bars(p, p + 1, p - 1, p, 2, seed=flag)),
        ("position_test: n must be an integer",
         lambda: s.position_test(s.BacktestPriceTarget(), x, p, n=flag)),
        ("position_test: periods_per_year must be a number",
         lambda: s.position_test(s.BacktestPriceTarget(), x, p, n=5, periods_per_year=flag)),
        ("walk_forward: train must be an integer",
         lambda: s.walk_forward(s.BacktestPriceTarget(), x, p, train=flag, test=5)),
        ("backtest_report: periods_per_year must be a number",
         lambda: s.backtest_report(out, periods_per_year=flag)),
    ]
    for msg, call in calls:
        with pytest.raises(TypeError, match=f"^{msg}, got bool$"):
            call()
