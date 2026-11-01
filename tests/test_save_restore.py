"""Save and load of operators: settings, state and the save file.

An operator saved after any number of events and loaded back continues bit for
bit: feeding A, saving, loading and feeding B gives what feeding A then B
gives. The registry test runs every functor of help.json at three save points
(at construction, mid warm-up and after two windows), through JSON and CBOR.
Every functor must be saveable (``test_every_functor_saveable``).

The golden files in ``tests/data/state_golden`` fix each operator's state
layout per state version, so a layout change without a version bump fails.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

import numpy as np
import pytest

import screamer
from screamer import Input, Pipeline
from screamer import screamer_bindings as _b
from tests._oracle_fresh import _same as _oracle_same, run_generator
from tests.regime_helpers import assert_bitwise
from tests import test_batch_stream_bitwise as _sweep
from tests.save_cases import (FUNCTOR_NAMES, HELP, REST, case, feed,
                              kwargs_for, splits)

ROOT = pathlib.Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "tests" / "data" / "state_golden"


def _same(a, b):
    """Two JSON-native trees are equal, NaN equal to NaN and -0.0 apart from 0.0."""
    return json.dumps(a) == json.dumps(b)


# ---- Continuation ------------------------------------------------------------

@pytest.mark.parametrize("split", ["0", "3", "2w"])
@pytest.mark.parametrize("name", FUNCTOR_NAMES)
def test_continuation(name, split):
    make, inputs = case(name)
    k = splits(name, kwargs_for(name))[("0", "3", "2w").index(split)]
    a = make()
    feed(a, inputs, 0, k)
    text = a.dumps()
    blob = a.dumps(format="cbor")
    assert isinstance(text, str) and isinstance(blob, bytes)
    b = screamer.loads(text)
    c = screamer.loads(blob)
    assert type(b).__name__ == name and type(c).__name__ == name
    assert b.dumps() == text and c.dumps() == text
    end = k + REST
    want = feed(a, inputs, k, end)
    assert_bitwise(feed(b, inputs, k, end), want)
    assert_bitwise(feed(c, inputs, k, end), want)


@pytest.mark.parametrize("name", FUNCTOR_NAMES)
def test_settings_match_help(name):
    make, _ = case(name)
    op = make()
    settings = op.settings()
    assert list(settings) == [p["name"] for p in HELP[name].get("parameters") or []]
    again = screamer.from_settings({"type": name, "settings": settings})
    assert type(again).__name__ == name
    assert _same(again.settings(), settings)


@pytest.mark.parametrize("name", FUNCTOR_NAMES)
def test_state_is_reset_state(name):
    make, inputs = case(name)
    op = make()
    fresh = op.state()
    feed(op, inputs, 0, 10)
    op.reset()
    assert _same(op.state(), fresh)


@pytest.mark.parametrize("name", FUNCTOR_NAMES)
def test_from_settings_is_fresh(name):
    make, inputs = case(name)
    op = make()
    feed(op, inputs, 0, 10)
    built = screamer.from_settings(json.loads(op.dumps())["object"])
    assert _same(built.state(), make().state())


def test_save_and_load_files(tmp_path):
    x = np.linspace(0.0, 1.0, 40)
    a = screamer.RollingMean(5)
    for v in x[:20]:
        a(v)
    for path in (tmp_path / "f.json", tmp_path / "f.cbor"):
        a.save(path)
        b = screamer.load(path)
        assert b.dumps() == a.dumps()
    assert (tmp_path / "f.json").read_text(encoding="utf-8") == a.dumps()
    assert (tmp_path / "f.cbor").read_bytes() == a.dumps(format="cbor")


def test_file_layout():
    a = screamer.RollingMean(3)
    for v in (1.0, 2.0, float("nan"), 4.0):
        a(v)
    doc = json.loads(a.dumps())
    assert list(doc) == ["screamer", "library_version", "object"]
    assert doc["screamer"] == "save/1" and doc["library_version"] == screamer.__version__
    node = doc["object"]
    assert list(node) == ["type", "settings", "state"]
    assert node["type"] == "RollingMean"
    assert node["settings"] == {"window_size": 3, "start_policy": "strict"}
    assert node["state"]["version"] == 1
    assert node["state"]["values"]["sum"] == 7.0


def test_ewmean_keeps_given_smoothing():
    assert screamer.EwMean(span=10).settings() == {
        "com": None, "span": 10.0, "halflife": None, "alpha": None}
    assert screamer.EwMean(halflife=3).settings()["halflife"] == 3.0
    b = screamer.from_settings({"type": "EwMean", "settings": screamer.EwMean(alpha=0.25).settings()})
    assert b.settings() == {"com": None, "span": None, "halflife": None, "alpha": 0.25}
    assert repr(b) == "EwMean(alpha=0.25)"


# ---- Edited and bad files ----------------------------------------------------

def _rolling_mean_20():
    a = screamer.RollingMean(20)
    for v in np.arange(30.0):
        a(v)
    return a


def test_edited_setting_refuses_or_fresh():
    text = _rolling_mean_20().dumps()
    edited = text.replace('"window_size": 20', '"window_size": 50')
    assert edited != text
    with pytest.raises(screamer.StateMismatch, match="window_size|buffer"):
        screamer.loads(edited)
    with pytest.warns(UserWarning, match="RollingMean"):
        b = screamer.loads(edited, on_mismatch="fresh")
    x = np.arange(100.0)
    fresh = screamer.RollingMean(50)
    assert_bitwise([b(v) for v in x], [fresh(v) for v in x])


def _edit(text, fn):
    doc = json.loads(text)
    fn(doc)
    return json.dumps(doc)


def _unknown_type(d):
    d["object"]["type"] = "NoSuchOperator"


def _unknown_field(d):
    d["object"]["state"]["values"]["extra"] = 1


def _missing_field(d):
    del d["object"]["state"]["values"]["sum"]


def _string_for_number(d):
    d["object"]["state"]["values"]["sum"] = "a lot"


def _wrong_length(d):
    d["object"]["state"]["values"]["buffer"] = [1.0, 2.0]


# (edit of a saved RollingMean(20), or a whole bad file; the path the message names)
STATE_EDITS = [
    (_unknown_field, "RollingMean.extra"),
    (_missing_field, "RollingMean.sum"),
    (_string_for_number, "RollingMean.sum"),
    (_wrong_length, "RollingMean.buffer"),
]
BAD_FILES = [(_unknown_type, "object.type")] + STATE_EDITS + [
    ("{not json", "save JSON"),
    (b"\xa1\x61", "CBOR"),
    ('{"screamer": "save/2", "object": {}}', "save/2"),
]


def _ids(cases):
    return [c.__name__.lstrip("_") if callable(c) else repr(c)[:12] for c, _ in cases]


@pytest.mark.parametrize("bad, where", BAD_FILES, ids=_ids(BAD_FILES))
def test_bad_files_named_errors(bad, where):
    text = _edit(_rolling_mean_20().dumps(), bad) if callable(bad) else bad
    with pytest.raises(screamer.SaveError, match=re.escape(where)):
        screamer.loads(text)


def test_constructor_runtime_error_is_a_save_error(monkeypatch):
    """A setting the constructor refuses with a RuntimeError (Butter of an
    order too high to design raises one) refuses the file with SaveError."""
    class Refusing(screamer.RollingMean):
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Denominator coefficients are complex.")
    text = _rolling_mean_20().dumps()
    monkeypatch.setattr(screamer, "RollingMean", Refusing)
    with pytest.raises(screamer.SaveError, match=re.escape("RollingMean.settings: Denominator")):
        screamer.loads(text)


def test_json_file_that_is_not_utf8_refused():
    data = _rolling_mean_20().dumps().encode("utf-8").replace(b'"RollingMean"', b'"Rolling\xf3Mean"')
    with pytest.raises(screamer.SaveError, match="save JSON: invalid UTF-8 at byte"):
        screamer.loads(data)


@pytest.mark.parametrize("bad, where", STATE_EDITS, ids=_ids(STATE_EDITS))
def test_failed_set_state_leaves_object_unchanged(bad, where):
    a = _rolling_mean_20()
    before = a.state()
    with pytest.raises(screamer.StateMismatch, match=re.escape(where)):
        a.set_state(json.loads(_edit(a.dumps(), bad))["object"]["state"])
    assert _same(a.state(), before)
    b = _rolling_mean_20()
    x = np.arange(30.0, 60.0)
    assert_bitwise([a(v) for v in x], [b(v) for v in x])


def test_bad_state_invariants_refuse():
    a = _rolling_mean_20()
    state = a.state()
    state["values"]["count"] = 21
    with pytest.raises(screamer.StateMismatch, match="RollingMean"):
        a.set_state(state)
    q = screamer.RollingQuantile(4, 0.5)
    for v in (1.0, 2.0, 3.0):
        q(v)
    state = q.state()
    state["values"]["window"][1] = 9.0
    with pytest.raises(screamer.StateMismatch, match="RollingQuantile"):
        q.set_state(state)
    m = screamer.RollingMax(4)
    for v in (3.0, 1.0, 2.0):
        m(v)
    state = m.state()
    state["values"]["values"] = state["values"]["values"][::-1]
    with pytest.raises(screamer.StateMismatch, match="RollingMax"):
        m.set_state(state)


def test_rolling_mean_index_must_equal_count_in_warm_up():
    a = screamer.RollingMean(5)
    for v in (1.0, 2.0):
        a(v)
    state = a.state()
    state["values"]["index"] = 4
    with pytest.raises(screamer.SaveError, match="RollingMean: index must equal count"):
        a.set_state(state)


def test_rolling_max_index_not_negative():
    m = screamer.RollingMax(4)
    state = m.state()
    state["values"]["index"] = -1
    with pytest.raises(screamer.SaveError, match="RollingMax: index must not be negative"):
        m.set_state(state)


def _fed(op, values):
    for v in values:
        op(v)
    return op


@pytest.mark.parametrize("make, edit, message", [
    (lambda: _fed(screamer.RollingVar(4), (1.0, 2.0)),
     lambda v: v["window"].update(index=3), "RollingVar.window: index must equal count"),
    (lambda: _fed(screamer.RollingVar(4, start_policy="zero"), (1.0,)),
     lambda v: v["window"].update(count=1, index=1), "count must equal window_size under the zero"),
    (lambda: _fed(screamer.RollingSum(4), (1.0, 2.0)),
     lambda v: v.update(count=5), "RollingSum: count must be at most window_size"),
    (lambda: _fed(screamer.RollingRange(4), (1.0, 2.0)),
     lambda v: v["min"].update(index=3, indices=[1, 2]), "min and max must have seen the same"),
    (lambda: _fed(screamer.RollingMedian(4), (1.0, 2.0, 3.0)),
     lambda v: v.update(high=[9.0]), "RollingMedian: low and high must hold the values"),
    (lambda: _fed(screamer.RollingIqr(4), (1.0, 2.0, 3.0)),
     lambda v: v["window"].__setitem__(1, 7.0), "RollingIqr: tree must hold the values"),
    (lambda: _fed(screamer.Hampel(4), (1.0, 2.0)),
     lambda v: v["window"].update(last=4), "Hampel.window: last must lie in"),
    (lambda: _fed(screamer.ImpulseClip(4, start_policy="zero"), (1.0, 2.0)),
     lambda v: v.update(have_prev=False), "ImpulseClip: the windows must be full and have_prev"),
    (lambda: _fed(screamer.ImpulseClip(4, start_policy="zero"), (1.0, 2.0)),
     lambda v: v["values"].update(count=1, index=1), "ImpulseClip: the windows must be full"),
    (lambda: _fed(screamer.ImpulseClip(4), (1.0, 2.0)),
     lambda v: v.update(have_prev=False), "ImpulseClip: have_prev must be true once values"),
    (lambda: _fed(screamer.RollingPermutationEntropy(10, 3), np.arange(6.0)),
     lambda v: v["counts"].__setitem__(0, 9), "counts must count the codes"),
    (lambda: _fed(screamer.RollingBipowerVar(4), (1.0, -2.0, 3.0)),
     lambda v: v.update(square_count=0), "square_count must equal the nonzero flags"),
    (lambda: _fed(screamer.RollingHMM(10), (1.0, 2.0)),
     lambda v: v.update(fitted=True), "a fitted model must have a full window"),
    (lambda: _fed(screamer.RollingSeasonalMean(3, 4), (1.0, 2.0)),
     lambda v: v.update(slots=v["slots"][:2]), r"RollingSeasonalMean.slots: expected 3 objects, got 2"),
    (lambda: _fed(screamer.RollingSeasonalMean(3, 4), (1.0, 2.0)),
     lambda v: v["slots"][1].update(count=5), r"RollingSeasonalMean.slots\[1\]: count must be at most"),
])
def test_rolling_state_checks(make, edit, message):
    """A hand edit that breaks a state invariant raises a SaveError naming the
    field's path, and leaves the operator unchanged."""
    op = make()
    state = op.state()
    before = op.dumps()
    edit(state["values"])
    with pytest.raises(screamer.SaveError, match=message):
        op.set_state(state)
    assert op.dumps() == before


def _fed2(op, pairs):
    for x, y in pairs:
        op(x, y)
    return op


_WALK = tuple(100.0 + np.cumsum(np.sin(np.arange(60.0))))


@pytest.mark.parametrize("make, edit, message", [
    (lambda: _fed(screamer.RollingHMM(10), (1.0, 2.0)),
     lambda v: v.update(var=[1.0, -1.0]), "RollingHMM: var must not be negative"),
    (lambda: _fed(screamer.Butter(2, 0.1, start_policy="zero"), (1.0, 2.0)),
     lambda v: v["iir"].update(started=False), "Butter.iir: a filter with start_policy zero"),
    (lambda: _fed(screamer.Butter(2, 0.1), (1.0,)),
     lambda v: v["iir"].update(z=[0.0]), r"Butter.iir.z: expected 2 values, got 1"),
    (lambda: _fed(screamer.MovingAverage([0.2, 0.3, 0.5]), (1.0,)),
     lambda v: v["fir"].update(index=2), "MovingAverage.fir: index must equal n_seen"),
    (lambda: _fed(screamer.FracDiff(0.4, 5), (1.0, 2.0)),
     lambda v: v["fir"].update(n_seen=9), "FracDiff.fir: n_seen must be at most"),
    (lambda: _fed(screamer.KalmanFilter(0.01, 1.0), (1.0,)),
     lambda v: v.update(diffuse_pending=True), "KalmanFilter: diffuse_pending needs"),
    (lambda: _fed(screamer.KalmanFilter(0.01, 1.0), (1.0,)),
     lambda v: v.update(P=-1.0), "KalmanFilter: P must not be negative"),
    (lambda: _fed(screamer.SchmittTrigger(-1.0, 1.0), (2.0,)),
     lambda v: v.update(output=0.5), "SchmittTrigger: output must be"),
    (lambda: _fed(screamer.Hold(3), (1.0,)),
     lambda v: v.update(remaining=3), "Hold: remaining must lie in"),
    (lambda: _fed(screamer.Hold(1, cooldown=2), (1.0,)),
     lambda v: v.update(cool=3), "Hold: cool must lie in"),
    (lambda: _fed(screamer.CusumFilter(1.0), (0.5,)),
     lambda v: v.update(pos=2.0), "CusumFilter: pos must lie in"),
    (lambda: _fed(screamer.DominantCycle(), _WALK),
     lambda v: v["engine"].update(n=-1), "DominantCycle.engine: n must not be negative"),
    (lambda: _fed(screamer.DominantCycle(), _WALK),
     lambda v: v["engine"]["i1"].update(values=[0.0] * 7),
     r"DominantCycle.engine.i1.values: expected 8 values, got 7"),
    (lambda: _fed(screamer.CycleSine(), _WALK),
     lambda v: v["dc"]["smooth"].update(values=[0.0] * 51),
     r"CycleSine.dc.smooth.values: expected 52 values, got 51"),
    (lambda: _fed(screamer.TrendMode(), _WALK),
     lambda v: v["dc"].update(side=2), "TrendMode.dc: side and prev_side"),
    (lambda: _fed(screamer.TrendMode(), _WALK),
     lambda v: v.update(days_in_trend=-1), "TrendMode: days_in_trend must not be negative"),
    (lambda: _fed(screamer.InstantaneousTrendline(), _WALK),
     lambda v: v["trendline"].update(count=51), "InstantaneousTrendline.trendline: count must lie"),
    (lambda: _fed2(screamer.EwCov(span=5), ((1.0, 2.0), (2.0, 1.0))),
     lambda v: v["sums"].update(w=-1.0), "EwCov.sums: w must not be negative"),
    (lambda: _fed2(screamer.EwBeta(span=5), ((1.0, 2.0), (2.0, 1.0))),
     lambda v: v.update(sum_w2=-1.0), "EwBeta: sum_w2 must not be negative"),
])
def test_ew_signal_state_checks(make, edit, message):
    """The state checks of the ew and signal operators and their blocks: a
    hand edit that breaks one raises a SaveError naming the path and leaves
    the operator unchanged."""
    op = make()
    state = op.state()
    before = op.dumps()
    edit(state["values"])
    with pytest.raises(screamer.SaveError, match=message):
        op.set_state(state)
    assert op.dumps() == before


def test_cycle_rings_continue_after_load():
    """The Hilbert and sinewave rings save newest first and load from slot 0,
    so a loaded TrendMode continues as the original does, mid ring."""
    x = np.asarray(_WALK)
    a = _fed(screamer.TrendMode(), x[:45])
    b = screamer.loads(a.dumps())
    assert b.dumps() == a.dumps()
    np.testing.assert_array_equal([b(v) for v in x[45:]], [a(v) for v in x[45:]])


def test_rolling_hmm_zero_var_after_inf_continues():
    """An infinite input leaves the fit floor NaN, so a constant group's var
    is 0; that state saves, loads and continues bit for bit."""
    x = np.concatenate([np.ones(30), [np.inf], np.ones(30), np.arange(20.0)])
    a = _fed(screamer.RollingHMM(20), x[:61])
    assert 0.0 in a.state()["values"]["var"]
    b = screamer.loads(a.dumps())
    assert b.dumps() == a.dumps()
    np.testing.assert_array_equal([b(v) for v in x[61:]], [a(v) for v in x[61:]])


# ---- Backtest engines mid position ---------------------------------------------

_T = 160
_rng = np.random.default_rng(11)
_P = 100.0 + np.cumsum(_rng.normal(0.0, 0.6, _T))
_OPEN = np.concatenate([[_P[0]], _P[:-1]])
_HIGH = np.maximum(_OPEN, _P) + 0.3
_LOW = np.minimum(_OPEN, _P) - 0.3
_TGT = np.round(3.0 * np.sin(np.arange(_T) / 7.0))
_SIZE = 2.0 + (np.arange(_T) % 5)
_CHUNK = _P[(np.arange(_T) // 8) * 8]              # quotes re-priced every 8 events
_MY_BID, _MY_ASK = _CHUNK - 0.4, _CHUNK + 0.4
_ONES = np.ones(_T)


def _engine_out(op, inputs):
    """An engine's (T, 4) output, fed by events."""
    return feed(op, inputs, 0, _T)


_PRICE_OUT = _engine_out(screamer.BacktestPriceTarget(fee=1e-4), [_TGT, _P])


def _portfolio_inputs(fields):
    """K = 3 assets: shifted copies of the walk, asset 1 unquoted on t 60..75."""
    cols = [np.stack([x, x * 1.01 + 1.0, x * 0.99 - 1.0], axis=1) for x in fields]
    for c in cols[1:]:
        c[60:76, 1] = np.nan
    return cols


_PF_OUT = feed(screamer.BacktestPortfolioTarget(size="units"),
               _portfolio_inputs([_TGT, _P]), 0, _T).reshape(_T, 3, 4)


def _pos(v):
    return v["account"]["position"] if "account" in v else v["core"]["account"]["position"]


def _finite(xs):
    return any(x == x for x in xs)


# (name, make, inputs, holds): `holds(values)` tells when the state is mid run.
_MID = [
    ("BacktestPriceTarget",
     lambda: screamer.BacktestPriceTarget(fee=1e-4, stop_loss=0.05, trailing_stop=0.02,
                                          size="notional", initial_capital=1e3),
     [_TGT * 100.0, _P], lambda v: _pos(v) != 0 and v["exits"]["entry"] == v["exits"]["entry"]),
    ("BacktestOHLCTarget", lambda: screamer.BacktestOHLCTarget(delay=3, taker_fee=1e-4),
     [_TGT, _OPEN, _HIGH, _LOW, _P],
     lambda v: _pos(v) != 0 and _finite(v["queue"]) and v["head"] != 0),
    ("BacktestOHLCVTarget",
     lambda: screamer.BacktestOHLCVTarget(max_participation=0.5, delay=2, trailing_stop=0.03),
     [_TGT, _OPEN, _HIGH, _LOW, _P, _SIZE], lambda v: _pos(v) != 0 and _finite(v["queue"])),
    ("BacktestL1Target", lambda: screamer.BacktestL1Target(stop_loss=0.02),
     [_TGT, _P - 0.05, _P + 0.05, _SIZE, _SIZE], lambda v: _pos(v) != 0),
    ("BacktestTradesTarget", lambda: screamer.BacktestTradesTarget(max_participation=0.5),
     [_TGT, _P, _SIZE], lambda v: _pos(v) != 0 and v["working"] == v["working"]),
    ("BacktestOHLCOrders", lambda: screamer.BacktestOHLCOrders(maker_fee=-1e-4),
     [_MY_BID, _ONES, _MY_ASK, _ONES, _OPEN, _HIGH, _LOW, _P], lambda v: _pos(v) != 0),
    ("BacktestTradesOrders", lambda: screamer.BacktestTradesOrders(participation_ratio=0.5),
     [_MY_BID, _ONES, _MY_ASK, _ONES, _P, _SIZE],
     lambda v: _pos(v) != 0 and v["bid_prev"] == v["bid_prev"]),
    ("BacktestL1Orders", lambda: screamer.BacktestL1Orders(fill="touch"),
     [_MY_BID, _ONES, _MY_ASK, _ONES, _P - 0.05, _P + 0.05, _SIZE, _SIZE],
     lambda v: _pos(v) != 0 and v["bid_prev"] == v["bid_prev"]),
    ("BacktestL1TradesOrders", lambda: screamer.BacktestL1TradesOrders(),
     [_MY_BID, _ONES, _MY_ASK, _ONES, _P - 0.05, _P + 0.05, _SIZE, _SIZE, _P, _SIZE],
     lambda v: _pos(v) != 0 and v["bid_prev"] == v["bid_prev"]),
    ("BacktestCarry", lambda: screamer.BacktestCarry(),
     [*_PRICE_OUT.T, _P, np.full(_T, 1e-4)], lambda v: v["prev_position"] != 0),
    ("BacktestReport", lambda: screamer.BacktestReport(252.0), list(_PRICE_OUT.T),
     lambda v: v["has_prev_pos"] and v["prev_pos"] != 0),
    ("BacktestStats", lambda: screamer.BacktestStats(252.0), list(_PRICE_OUT.T),
     lambda v: v["tracker"]["open"] and v["stats"]["trades"] > 0),
    ("BacktestPortfolioTarget",
     lambda: screamer.BacktestPortfolioTarget(fee=[1e-4, 2e-4, 0.0], size="units"),
     _portfolio_inputs([_TGT, _P]),
     lambda v: any(a["position"] != 0 for a in v["accounts"])),
    ("BacktestPortfolioOHLCTarget",
     lambda: screamer.BacktestPortfolioOHLCTarget(size="units", delay=2, fill_at="open"),
     _portfolio_inputs([_TGT, _OPEN, _HIGH, _LOW, _P]),
     lambda v: _finite(v["pending"]) and _finite(v["queue"])
     and any(a["position"] != 0 for a in v["accounts"])),
    ("PortfolioReport", lambda: screamer.PortfolioReport(), [_PF_OUT],
     lambda v: any(p != 0 for p in v["prev_positions"])),
]


@pytest.mark.parametrize("name, make, inputs, holds", _MID, ids=[m[0] for m in _MID])
def test_backtest_mid_position(name, make, inputs, holds):
    """A backtest engine saved mid run (holding a position, with a resting
    order or a delayed target where it has them) loads and continues bit for
    bit, through JSON and CBOR; the infinite position caps of its settings
    round-trip."""
    a = make()
    for k in range(1, _T - 20):
        feed(a, inputs, k - 1, k)
        if k >= 10 and holds(json.loads(a.dumps())["object"]["state"]["values"]):
            break
    else:
        pytest.fail(f"{name}: the run never reached the state to save")
    text, blob = a.dumps(), a.dumps(format="cbor")
    settings = json.loads(text)["object"]["settings"]
    if "max_position" in settings:
        assert settings["max_position"] == np.inf and settings["min_position"] == -np.inf
    if "max_gross" in settings:
        assert settings["max_gross"] == np.inf and settings["max_net"] == np.inf
    want = feed(a, inputs, k, _T)
    for data in (text, blob):
        b = screamer.loads(data)
        assert b.dumps() == text
        assert_bitwise(feed(b, inputs, k, _T), want)
    # The continuation equals the uninterrupted run.
    assert_bitwise(want, feed(make(), inputs, 0, _T)[k:])

def _mid(name, k=40):
    """An engine of the mid-position cases fed k events."""
    _, make, inputs, _ = next(m for m in _MID if m[0] == name)
    op = make()
    feed(op, inputs, 0, k)
    return op


@pytest.mark.parametrize("make, edit, message", [
    (lambda: _mid("BacktestPriceTarget"),
     lambda v: v["account"].update(has_prev=False), "BacktestPriceTarget.account: an account"),
    (lambda: _mid("BacktestPriceTarget"),
     lambda v: v["exits"].update(blocked=2.0), "BacktestPriceTarget.exits: blocked must be"),
    (lambda: _mid("BacktestPriceTarget"),
     lambda v: v["sizer"].update(has=False), "BacktestPriceTarget.sizer: value must be set"),
    (lambda: _mid("BacktestOHLCTarget"),
     lambda v: v.update(head=3), "BacktestOHLCTarget: head must be inside"),
    (lambda: _mid("BacktestOHLCTarget"),
     lambda v: v.update(queue=[0.0]), r"BacktestOHLCTarget.queue: expected 3 values, got 1"),
    (lambda: _mid("BacktestL1Orders"),
     lambda v: v.update(bid_locked=True, bid_prev=np.nan), "BacktestL1Orders: a locked bid"),
    (lambda: _mid("BacktestStats"),
     lambda v: v["stats"].update(wins=v["stats"]["trades"] + 1), "BacktestStats.stats: wins"),
    (lambda: _mid("BacktestStats"),
     lambda v: v["tracker"].update(open=not v["tracker"]["open"]), "BacktestStats.tracker: open"),
    (lambda: _mid("BacktestReport"),
     lambda v: v["report"].update(worst_dd=1.0), "BacktestReport.report: worst_dd"),
    (lambda: _fed(screamer.LiquidationDistance(5.0), ()),
     lambda v: v.update(side=2), "LiquidationDistance: side must be"),
    (lambda: _mid("BacktestPortfolioOHLCTarget"),
     lambda v: v.update(width=2), "BacktestPortfolioOHLCTarget.accounts: expected 2 objects, got 3"),
    (lambda: _mid("BacktestPortfolioTarget"),
     lambda v: v.update(width=4), "BacktestPortfolioTarget: width must fit"),
    (lambda: _mid("BacktestPortfolioOHLCTarget"),
     lambda v: v.update(head=2), "BacktestPortfolioOHLCTarget: head must be inside"),
    (lambda: _mid("BacktestPortfolioOHLCTarget"),
     lambda v: v.update(pending=[0.0]), r"BacktestPortfolioOHLCTarget.pending: expected 3 values"),
])
def test_backtest_state_checks(make, edit, message):
    """The state checks of the backtest engines and their blocks: a hand edit
    that breaks one raises a SaveError naming the path and leaves the engine
    unchanged."""
    op = make()
    state = op.state()
    before = op.dumps()
    edit(state["values"])
    with pytest.raises(screamer.SaveError, match=message):
        op.set_state(state)
    assert op.dumps() == before


def test_portfolio_loads_into_a_fresh_engine():
    """A portfolio engine's state of K assets loads into a fresh engine (no
    width yet), and a state of no assets resets a running one."""
    a = _mid("BacktestPortfolioOHLCTarget")
    b = screamer.BacktestPortfolioOHLCTarget(size="units", delay=2, fill_at="open")
    fresh = b.state()
    b.set_state(a.state())
    assert b.dumps() == a.dumps()
    a.set_state(fresh)
    assert _same(a.state(), fresh)

def _fed_rows(op, rows):
    for r in rows:
        op(*r)
    return op


_RETS = tuple(np.diff(np.log(_WALK)))
_PAIRS = tuple(zip(_WALK[1:], _WALK[:-1]))


@pytest.mark.parametrize("make, edit, message", [
    (lambda: _fed(screamer.RollingCVaR(5), _RETS[:8]),
     lambda v: v["tree"].update(keys=[k + 1.0 for k in v["tree"]["keys"]]),
     "RollingCVaR: tree must hold the values of the window"),
    (lambda: _fed(screamer.RollingMaxDrawdown(5), _WALK[:3]),
     lambda v: v.update(index=1), "RollingMaxDrawdown: index must equal count"),
    (lambda: _fed(screamer.RollingSortino(5), _RETS[:3]),
     lambda v: v.update(count=6), "RollingSortino: count must lie in"),
    (lambda: _fed(screamer.RollingHitRate(5), _RETS[:3]),
     lambda v: v.update(n_seen=6), "RollingHitRate: n_seen must lie in"),
    (lambda: _fed(screamer.RollingDownsideDeviation(5, start_policy="zero"), _RETS[:3]),
     lambda v: v.update(n=3), "RollingDownsideDeviation: n must equal window_size"),
    (lambda: _fed(screamer.BandSignal(), (2.5, 1.0)),
     lambda v: v.update(pos=0.5), "BandSignal: pos must be"),
    (lambda: _fed2(screamer.KalmanRegression(), _PAIRS[:3]),
     lambda v: v.update(n=-1), "KalmanRegression: n must not be negative"),
    (lambda: _fed(screamer.RollingADF(10, 1), _WALK[:8]),
     lambda v: v.update(seen=v["seen"] + 1), "RollingADF: count must follow from seen"),
    (lambda: _fed(screamer.RollingADF(10, 1), _WALK[:6]),
     lambda v: v.update(head=0), "RollingADF: head must equal count"),
    (lambda: _fed_rows(screamer.RollingMultiRegression(8, 1), _PAIRS[:5]),
     lambda v: v.update(head=1), "RollingMultiRegression: head must equal n"),
    (lambda: _fed_rows(screamer.ExpandingMultiRegression(1), _PAIRS[:5]),
     lambda v: v.update(fitted=True, n=1), "ExpandingMultiRegression: a fit needs n_min"),
    (lambda: _fed_rows(screamer.EwMultiRegression(1, span=5.0), _PAIRS[:5]),
     lambda v: v.update(c=[0.0]), r"EwMultiRegression.c: expected 3 values, got 1"),
    (lambda: _fed(screamer.BayesianChangepoint(max_run=4), _RETS[:10]),
     lambda v: v.update(len=6), "BayesianChangepoint: len must be at most"),
    (lambda: _fed(screamer.TripleBarrier(5, 0.01, 0.01), _WALK[:3]),
     lambda v: v.update(cursor=1), "TripleBarrier: cursor must be 0 during the warm-up"),
    (lambda: _fed(screamer.TripleBarrier(5, 0.01, 0.01), _WALK[:3]),
     lambda v: v.update(age=[0, 9, 0]), "TripleBarrier: age must lie in"),
])
def test_fin_state_checks(make, edit, message):
    """The state checks of the return, ratio, regression and label
    operators: a hand edit that breaks one raises a SaveError naming the path
    and leaves the operator unchanged."""
    op = make()
    state = op.state()
    before = op.dumps()
    edit(state["values"])
    with pytest.raises(screamer.SaveError, match=message):
        op.set_state(state)
    assert op.dumps() == before


_BARS = tuple((w * 1.01, w * 0.99, w) for w in _WALK)
_OHLC = tuple((w * 1.002, w * 1.01, w * 0.99, w) for w in _WALK)


def _edge_ring_inf(v):
    v["ring"][1] = float("inf")


@pytest.mark.parametrize("make, edit, message", [
    (lambda: _fed(screamer.Diff2(), _WALK[:1]),
     lambda v: v.update(full1=False, full2=True), "Diff2: the second delay fills after the first"),
    (lambda: _fed(screamer.Lag(3), _WALK[:2]),
     lambda v: v["delay"].update(index=0), "Lag.delay: index must equal count"),
    (lambda: _fed(screamer.Momentum(3), _WALK[:2]),
     lambda v: v["delay"].update(count=4), "Momentum.delay: count must be at most"),
    (lambda: _fed(screamer.First(), _WALK[:2]),
     lambda v: v.update(has_value=False), "First: first_value must be set exactly when has_value"),
    (lambda: _fed(screamer.TickRuleSign(), _WALK[:3]),
     lambda v: v.update(last_sign=0.5), "TickRuleSign: last_sign must be 1, -1 or NaN"),
    (lambda: _fed(screamer.TickRuleSign(), _WALK[:3]),
     lambda v: v.update(have_prev=False), "TickRuleSign: a sign needs a previous price"),
    (lambda: _fed2(screamer.LeeReadySign(), _PAIRS[:3]),
     lambda v: v.update(last_sign=2.0), "LeeReadySign: last_sign must be 1, -1 or NaN"),
    (lambda: _fed(screamer.BulkVolumeClassifier(5), _RETS[:3]),
     lambda v: v.update(n=2), "BulkVolumeClassifier: n must count the samples"),
    (lambda: _fed(screamer.BulkVolumeClassifier(5, start_policy="zero"), _RETS[:3]),
     lambda v: v.update(n=3), "BulkVolumeClassifier: n must count the samples"),
    (lambda: _fed(screamer.BulkVolumeClassifier(5), _RETS[:3]),
     lambda v: v["sum_yy"].update(count=2, index=2), "BulkVolumeClassifier: the window sums"),
    (lambda: _fed(screamer.RollSpread(5), _WALK[:6]),
     lambda v: v.update(have_price=False), "RollSpread: a price change needs a price"),
    (lambda: _fed(screamer.RollSpread(5), _WALK[:6]),
     lambda v: v.update(n=1), "RollSpread: n must count the samples"),
    (lambda: _fed_rows(screamer.CorwinSchultzSpread(5), _BARS[:3]),
     lambda v: v.update(have_prev=False), "CorwinSchultzSpread: the window needs a previous bar"),
    (lambda: _fed_rows(screamer.AbdiRanaldoSpread(5), _BARS[:3]),
     lambda v: v.update(have_prev=False), "AbdiRanaldoSpread: the window needs a previous bar"),
    (lambda: _fed_rows(screamer.EdgeSpread(5), _OHLC[:3]),
     lambda v: v.update(pos=0), "EdgeSpread: pos must equal bars"),
    (lambda: _fed_rows(screamer.EdgeSpread(5), _OHLC[:3]),
     lambda v: v.update(bars=6), "EdgeSpread: bars must be at most window_size"),
    (lambda: _fed_rows(screamer.EdgeSpread(5), _OHLC[:3]),
     lambda v: v.update(counts=[3] * 7), "EdgeSpread: counts must be in 0..bars - 1"),
    (lambda: _fed_rows(screamer.EdgeSpread(5), _OHLC[:3]),
     _edge_ring_inf, "EdgeSpread: ring values must be finite or NaN"),
    (lambda: _fed_rows(screamer.EdgeSpread(5), _OHLC[:3]),
     lambda v: v.update(ring=v["ring"][:8]), r"EdgeSpread.ring: expected 12 values, got 8"),
    (lambda: _fed(screamer.Propagator(5), _RETS[:3]),
     lambda v: v.update(index=1), "Propagator: index must equal count"),
    (lambda: _fed2(screamer.VPIN(1.0, 5), ((0.6, 0.6),) * 3),
     lambda v: v.update(filled=6), "VPIN: filled must be in 0..n_buckets"),
    (lambda: _fed2(screamer.VPIN(1.0, 5), ((0.6, 0.6),) * 3),
     lambda v: v.update(head=0), "VPIN: head must equal filled"),
    (lambda: _fed2(screamer.VPIN(1.0, 5), ((0.2, 0.1),)),
     lambda v: v.update(cur_buy=1e9), "VPIN: the open bucket must be at most bucket_volume"),
    (lambda: _fed2(screamer.VPIN(1.0, 5), ((0.2, 0.1),)),
     lambda v: v.update(cur_sell=-0.5), "VPIN: the open bucket must not be negative"),
    (lambda: _fed2(screamer.VPIN(1.0, 5), ((0.2, 0.1),)),
     lambda v: v.update(cur_buy=float("nan")), "VPIN: the open bucket must not be negative"),
    (lambda: _fed2(screamer.VPIN(1.0, 5), ((0.6, 0.6),) * 3),
     lambda v: v["imbalances"].__setitem__(0, 2.0), "VPIN: imbalances must be in 0..bucket_volume"),
    (lambda: _fed2(screamer.VPIN(1.0, 5), ((0.6, 0.6),) * 3),
     lambda v: v.update(sum=float("inf")), "VPIN: sum must be finite"),
    (lambda: _fed_rows(screamer.ContOFI(), _OHLC[:2]),
     lambda v: v.update(prev_bid=float("nan")), "ContOFI: the previous quote must not be NaN"),
    (lambda: _fed_rows(screamer.EdgeSpread(5), _OHLC[:3]),
     lambda v: v["sums"].__setitem__(0, float("nan")), "EdgeSpread: sums must be finite"),
    (lambda: _fed2(screamer.RealizedSpread(3), _PAIRS[:2]),
     lambda v: v.update(count=4), "RealizedSpread: count must be in 0..lag"),
])
def test_misc_micro_state_checks(make, edit, message):
    """The state checks of the lag, sign, spread and flow operators: a hand
    edit that breaks one raises a SaveError naming the path and leaves the
    operator unchanged."""
    op = make()
    state = op.state()
    before = op.dumps()
    edit(state["values"])
    with pytest.raises(screamer.SaveError, match=message):
        op.set_state(state)
    assert op.dumps() == before


def test_edge_spread_continues_after_an_outlier_resum():
    """An EdgeSpread saved around an outlier bar's early recompute loads and
    continues bit for bit: the sums carry their rounding."""
    rng = np.random.default_rng(7)
    c = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 120)))
    o = c * np.exp(rng.normal(0.0, 0.002, 120))
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0.0, 0.004, 120)))
    lo = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0.0, 0.004, 120)))
    h[33] *= 1.5     # an outlier bar; its second pair leaves at bar 43
    o[50] = np.nan   # a missing open
    full = screamer.EdgeSpread(10)(o, h, lo, c)
    for k in (43, 44, 46, 55):
        a = screamer.EdgeSpread(10)
        head = a.resume(o[:k], h[:k], lo[:k], c[:k])
        if k == 44:
            assert a.state()["values"]["since"] == 0   # the early recompute
        for data in (a.dumps(), a.dumps(format="cbor")):
            b = screamer.loads(data)
            assert b.dumps() == a.dumps()
            tail = b.resume(o[k:], h[k:], lo[k:], c[k:])
            assert_bitwise(np.concatenate([head, tail]), full)


def test_triple_barrier_ring_continues_after_load():
    """A TripleBarrier saved with open labels in its ring loads and resolves
    them as the original does."""
    x = np.asarray(_WALK)
    a = _fed(screamer.TripleBarrierOHLC(5, 0.005, 0.005), ())
    for v in x[:12]:
        a(v, v + 0.4, v - 0.4, v)
    assert any(a.state()["values"]["open"])
    b = screamer.loads(a.dumps())
    assert b.dumps() == a.dumps()
    np.testing.assert_array_equal([b(v, v + 0.4, v - 0.4, v) for v in x[12:]],
                                  [a(v, v + 0.4, v - 0.4, v) for v in x[12:]])

def test_renamed_alias_saves_its_own_type():
    """ExpandingSum is CumSum under another name, and saves under its own."""
    a = _fed(screamer.ExpandingSum(), (1.0, 2.0))
    doc = json.loads(a.dumps())
    assert doc["object"]["type"] == "ExpandingSum"
    b = screamer.loads(a.dumps())
    assert type(b).__name__ == "ExpandingSum" and b(3.0) == 6.0
    assert json.loads(_fed(screamer.CumSum(), (1.0,)).dumps())["object"]["type"] == "CumSum"
    p = screamer.ExpandingMax()
    p.resume(np.array([[1.0, 5.0], [3.0, 2.0]]))
    assert type(screamer.loads(p.dumps())).__name__ == "ExpandingMax"
    state = a.state()
    state["values"]["sum"] = "x"
    with pytest.raises(screamer.SaveError, match="ExpandingSum.sum"):
        a.set_state(state)


def test_rolling_quantile_fresh_value_by_value_equals_batch():
    """A fresh RollingQuantile starts with an empty (NaN) window, so a -1.0
    in the data stays in the window (it started filled with -1 before)."""
    x = [-1.0, 5.0, 7.0]
    q = screamer.RollingQuantile(3, 0.5)
    got = [q(v) for v in x]
    assert_bitwise(got, screamer.RollingQuantile(3, 0.5)(np.array(x)))
    assert_bitwise(got, [np.nan, np.nan, 5.0])


def test_unknown_setting_named():
    with pytest.raises(screamer.SaveError, match="RollingMean.settings.window"):
        screamer.from_settings({"type": "RollingMean", "settings": {"window": 3}})


def test_invalid_setting_named():
    with pytest.raises(screamer.SaveError, match="RollingMean.settings"):
        screamer.from_settings({"type": "RollingMean", "settings": {"window_size": 0}})


def test_state_version_mismatch():
    text = _rolling_mean_20().dumps().replace('"version": 1', '"version": 99')
    with pytest.raises(screamer.StateVersionError, match=r"RollingMean.*99.*\b1\b"):
        screamer.loads(text)
    with pytest.warns(UserWarning, match="RollingMean"):
        b = screamer.loads(text, on_mismatch="fresh")
    assert _same(b.state(), screamer.RollingMean(20).state())


def test_on_mismatch_checked():
    with pytest.raises(ValueError, match="on_mismatch"):
        screamer.loads(_rolling_mean_20().dumps(), on_mismatch="ignore")


def test_errors_are_save_errors():
    for cls in (screamer.StateVersionError, screamer.NotSaveable, screamer.StateMismatch):
        assert issubclass(cls, screamer.SaveError)
    assert issubclass(screamer.SaveError, ValueError)


@pytest.mark.parametrize("name", FUNCTOR_NAMES)
def test_every_functor_saveable(name):
    """Every functor of help.json saves and loads. A new operator declares
    kTypeName, kStateVersion, settings(Ar&) and state(Ar&); CONTRIBUTING.md
    ("Save and restore") says how."""
    op = case(name)[0]()
    assert op._saveable() and op._save_type() == name, (
        f"{name} cannot be saved: give it kTypeName, kStateVersion, settings(Ar&) and "
        f"state(Ar&) (CONTRIBUTING.md, \"Save and restore\"), then run "
        f"devtools/gen_state_golden.py {name}")


def test_not_saveable_message():
    """A subclass that renames an operator would save as its parent, so it
    refuses, naming itself."""
    class MyMean(screamer.RollingMean):
        pass

    with pytest.raises(screamer.NotSaveable, match="MyMean: saves as RollingMean"):
        MyMean(3).dumps()


# ---- Nested operators --------------------------------------------------------

def test_nested_op_roundtrip():
    rng = np.random.default_rng(3)
    labels = np.repeat(np.arange(8.0), 9)
    x = rng.standard_normal(labels.size)
    a = screamer.Segment(screamer.RollingMean(5))
    for t in range(40):
        a(labels[t], x[t])
    text = a.dumps()
    node = json.loads(text)["object"]
    assert node["settings"] == {
        "op": {"type": "RollingMean", "settings": {"window_size": 5, "start_policy": "strict"}}}
    assert node["state"]["values"]["op"]["version"] == 1
    loaded = [screamer.loads(text), screamer.loads(a.dumps(format="cbor"))]
    want = [a(labels[t], x[t]) for t in range(40, labels.size)]
    for b in loaded:
        assert b.dumps() == text
        assert_bitwise([b(labels[t], x[t]) for t in range(40, labels.size)], want)


def _deep_segment(depth):
    """Segment(Segment(...RollingMean(3))) with `depth` Segments, fed a few
    rows (the label of each level, then the value)."""
    op = screamer.RollingMean(3)
    for _ in range(depth):
        op = screamer.Segment(op)
    for t in range(5):
        op(*([0.0] * depth), float(t))
    return op


def test_deep_nesting_loads_once_per_level():
    """A load walks each nested operator once per pass. Each level used to
    load its operator three times, so 12 levels took 0.8 s and 16 (the most
    Segment's input limit allows) about a minute."""
    import time
    a = _deep_segment(16)
    text = a.dumps()
    start = time.perf_counter()
    b = screamer.loads(text)
    assert time.perf_counter() - start < 1.0
    assert b.dumps() == text


def test_deep_nesting_failed_load_leaves_state_unchanged():
    """A check that fails in the innermost operator, after every outer level
    has read its fields, undoes the whole load."""
    a = _deep_segment(4)
    before = a.state()
    state = json.loads(json.dumps(before))
    inner = state
    for _ in range(4):
        inner["values"]["started"] = False
        inner["values"]["label"] = 7.0
        inner = inner["values"]["op"]
    inner["values"]["count"] = 99
    with pytest.raises(screamer.StateMismatch, match=re.escape("Segment.op.op.op.op")):
        a.set_state(state)
    assert _same(a.state(), before)


def test_nested_op_refuses_per_column_state():
    """A nested operator keeps one state; a "columns" state in its place
    would never run, so it raises instead of loading into a panel."""
    labels = np.zeros(3)
    x = np.array([4.0, 5.0, 6.0])
    a = screamer.Segment(screamer.RollingMean(2))
    for t in range(2):
        a(labels[t], x[t])
    state = a.state()
    state["values"]["op"] = {"version": 1, "columns": [state["values"]["op"]["values"]]}
    with pytest.raises(screamer.StateMismatch,
                       match=re.escape("Segment.op.columns: a nested operator holds no per-column state")):
        a.set_state(state)
    assert a(labels[2], x[2]) == 5.5


def test_operator_nesting_is_capped():
    """Settings nested deeper than serial::kMaxOpDepth raise SaveError
    before any operator is built, in Python and in the C++ registry."""
    assert _b._serial_max_op_depth == 32
    node = {"type": "RollingMean", "settings": {"window_size": 3}}
    for _ in range(33):
        node = {"type": "Segment", "settings": {"op": node}}
    where = "Segment" + ".settings.op" * 32 + ": operators nested more than 32 deep"
    with pytest.raises(screamer.SaveError, match=re.escape(where)):
        screamer.from_settings(node)
    with pytest.raises(screamer.SaveError, match=re.escape(where)):
        _b._serial_make_op("Segment", json.dumps(node["settings"]))


def test_int_settings_out_of_range_as_python_words_them():
    """Every integer setting at -1 and at 0 (the oracle's "int_errors", which
    Python wrote) gives the same SaveError through the C++ registry, which
    JavaScript uses."""
    cases = json.loads((ROOT / "wasm" / "smoke" / "save_oracle.json").read_text("utf-8"))["int_errors"]
    assert len(cases) > 300
    for c in cases:
        try:
            _b._serial_make_op(c["name"], json.dumps(c["settings"]))
            got = None
        except screamer.SaveError as e:
            got = str(e)
        assert got == c["message"], c


# ---- Files ---------------------------------------------------------------------

def test_loads_takes_any_bytes_object():
    a = _rolling_mean_20()
    blob = a.dumps(format="cbor")
    for data in (bytearray(blob), memoryview(blob), bytearray(a.dumps().encode("utf-8"))):
        assert screamer.loads(data).dumps() == a.dumps()


def test_utf8_bom_and_other_files():
    a = _rolling_mean_20()
    assert screamer.loads(b"\xef\xbb\xbf" + a.dumps().encode("utf-8")).dumps() == a.dumps()
    for data in ("[1, 2]", b"", b"\xef\xbb\xbf[1]", b"\x00"):
        with pytest.raises(screamer.SaveError, match="neither a JSON object nor CBOR"):
            screamer.loads(data)


def test_set_state_takes_only_a_state():
    a = _rolling_mean_20()
    for bad, kind in (([1], "an array"), ("x", "a string"), (None, "null"), (3, "a number")):
        with pytest.raises(screamer.StateMismatch, match=re.escape(
                'RollingMean: expected a state {"version": N, "values": {...}}, got ' + kind)):
            a.set_state(bad)


@pytest.mark.parametrize("name", ["m.json", "m.cbor"])
def test_failed_save_keeps_the_previous_file(tmp_path, monkeypatch, name):
    """A save writes a new file beside the target and renames it over the
    target, so a write that fails leaves the previous file whole."""
    import os
    path = tmp_path / name
    _rolling_mean_20().save(path)
    before = path.read_bytes()

    def fail(src, dst):
        raise OSError("disk full")
    monkeypatch.setattr(os, "replace", fail)
    x = Input("x")
    pipe = Pipeline([x], [screamer.RollingMean(3)(x)])
    for obj in (screamer.RollingMean(7), pipe, pipe.live()):
        with pytest.raises(OSError, match="disk full"):
            obj.save(path)
        assert path.read_bytes() == before
        assert sorted(p.name for p in tmp_path.iterdir()) == [name]
    monkeypatch.undo()
    screamer.RollingMean(7).save(path)
    assert screamer.load(path).settings()["window_size"] == 7
    assert sorted(p.name for p in tmp_path.iterdir()) == [name]


@pytest.mark.skipif(sys.platform == "win32", reason="symbolic links and modes need POSIX")
def test_save_follows_links_and_keeps_permissions(tmp_path):
    """A save through a symbolic link replaces the file the link points to
    and keeps the link; a replaced file keeps its permissions."""
    import os
    import stat
    target = tmp_path / "real.json"
    _rolling_mean_20().save(target)
    os.chmod(target, 0o600)
    (tmp_path / "link.json").symlink_to("real.json")
    screamer.RollingMean(7).save(tmp_path / "link.json")
    assert (tmp_path / "link.json").is_symlink()
    assert stat.S_IMODE(os.stat(target).st_mode) == 0o600
    assert screamer.load(target).settings()["window_size"] == 7
    (tmp_path / "dangling.json").symlink_to("new.json")
    _rolling_mean_20().save(tmp_path / "dangling.json")
    assert (tmp_path / "dangling.json").is_symlink()
    assert screamer.load(tmp_path / "new.json").dumps() == _rolling_mean_20().dumps()
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "dangling.json", "link.json", "new.json", "real.json"]


def test_fresh_warning_points_at_the_caller(tmp_path):
    a = _rolling_mean_20()
    d = json.loads(a.dumps())
    d["object"]["settings"]["window_size"] = 5
    path = tmp_path / "m.json"
    path.write_text(json.dumps(d), encoding="utf-8")
    for load in (lambda: screamer.loads(json.dumps(d), on_mismatch="fresh"),
                 lambda: screamer.load(path, on_mismatch="fresh")):
        with pytest.warns(UserWarning) as rec:
            load()
        assert rec[0].filename == __file__


# ---- Golden files ------------------------------------------------------------

def _golden_cases():
    return sorted(p.name[:-len(".case.json")] for p in GOLDEN.glob("*.case.json"))


def _with_library_version(text):
    return re.sub(r'"library_version": "[^"]*"',
                  f'"library_version": "{screamer.__version__}"', text, count=1)


@pytest.mark.parametrize("stem", _golden_cases())
def test_golden(stem):
    name, version = re.fullmatch(r"(\w+)\.v(\d+)", stem).groups()
    scenario = json.loads((GOLDEN / f"{stem}.case.json").read_text(encoding="utf-8"))
    text = _with_library_version((GOLDEN / f"{stem}.json").read_text(encoding="utf-8"))
    prefix = [np.asarray(x, dtype=np.float64) for x in scenario["prefix"]]
    rest = [np.asarray(x, dtype=np.float64) for x in scenario["rest"]]
    current = screamer.from_settings(json.loads(text)["object"]).state()["version"]
    if int(version) != current:
        with pytest.raises(screamer.StateVersionError):
            screamer.loads(text)
        return
    a = screamer.from_settings(json.loads(text)["object"])
    feed(a, prefix, 0, len(prefix[0]))
    assert a.dumps() == text, (
        f"{name}: the state layout differs from the golden of state version {version}; "
        "bump kStateVersion and add a golden with devtools/gen_state_golden.py")
    b = screamer.loads(text)
    assert b.dumps() == text
    want = np.asarray(scenario["rest_output"], dtype=np.float64)
    assert_bitwise(feed(b, rest, 0, len(rest[0])).reshape(want.shape), want)


def test_goldens_cover_saveable():
    """Every saveable operator has a golden of its current state version."""
    missing = []
    for name in FUNCTOR_NAMES:
        version = case(name)[0]().state()["version"]
        if not (GOLDEN / f"{name}.v{version}.json").exists():
            missing.append(f"{name}.v{version}")
    assert not missing, f"no golden for {missing}; run devtools/gen_state_golden.py"


# ---- The C++ registry and the JavaScript files ---------------------------------

def _settings_value(v):
    """A constructor argument of tests/save_cases.py as a settings value."""
    if isinstance(v, _b.EvalOp):
        return {"type": type(v).__name__, "settings": v.settings()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_settings_value(x) for x in v]
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v)
    return v


@pytest.mark.parametrize("name", FUNCTOR_NAMES)
def test_make_op_builds_every_functor(name):
    """serial::make_op (the JavaScript loader's builder) builds each functor
    as its Python class does: the same outputs, and the same file."""
    make, inputs = case(name)
    settings = {k: _settings_value(v) for k, v in kwargs_for(name).items()}
    built = _b._serial_make_op(name, json.dumps(settings))
    assert_bitwise(np.asarray(built(*inputs)), np.asarray(make()(*inputs)))
    assert built._dumps(False) == make().dumps()


@pytest.mark.parametrize("settings, where, error", [
    ('{"window": 3}', "RollingMean.settings.window: unknown parameter", screamer.SaveError),
    ('{"window_size": 2.5}', "RollingMean.settings.window_size: expected an integer", screamer.SaveError),
    ('{"window_size": "3"}', "RollingMean.settings.window_size: expected an integer", screamer.SaveError),
    ('{"window_size": 0}', "RollingMean.settings: ", screamer.SaveError),
    ('[]', "RollingMean.settings: expected an object, got an array", screamer.SaveError),
])
def test_make_op_errors_named(settings, where, error):
    with pytest.raises(error, match=re.escape(where)):
        _b._serial_make_op("RollingMean", settings)


def test_make_op_unknown_type_and_nested():
    with pytest.raises(screamer.SaveError, match=re.escape("object.type: unknown type 'Nope'")):
        _b._serial_make_op("Nope", "{}")
    with pytest.raises(screamer.SaveError,
                       match=re.escape("Segment.settings.op.type: unknown type 'Nope'")):
        _b._serial_make_op("Segment", '{"op": {"type": "Nope", "settings": {}}}')
    with pytest.raises(screamer.SaveError, match=re.escape("Segment.settings.op: missing required")):
        _b._serial_make_op("Segment", "{}")
    seg = _b._serial_make_op("Segment", '{"op": {"type": "RollingMean", "settings": {"window_size": 5}}}')
    assert seg._dumps(False) == screamer.Segment(screamer.RollingMean(5)).dumps()


@pytest.fixture
def rename_window():
    """A test row of the rename table: RollingMean's window_size was 'window'."""
    _b._serial_test_rename("RollingMean", "window", "window_size")
    yield
    _b._serial_test_rename_clear()


def test_settings_compat_renames(rename_window):
    a = screamer.from_settings({"type": "RollingMean", "settings": {"window": 7}})
    assert a.settings() == {"window_size": 7, "start_policy": "strict"}
    old = screamer.RollingMean(7).dumps().replace('"window_size"', '"window"')
    assert screamer.loads(old).dumps() == screamer.RollingMean(7).dumps()
    built = _b._serial_make_op("RollingMean", '{"window": 7}')
    assert built._dumps(False) == screamer.RollingMean(7).dumps()
    for both in ({"window": 7, "window_size": 7},):
        with pytest.raises(screamer.SaveError, match="RollingMean.settings.window: renamed to window_size"):
            screamer.from_settings({"type": "RollingMean", "settings": both})
        with pytest.raises(screamer.SaveError, match="RollingMean.settings.window: renamed to window_size"):
            _b._serial_make_op("RollingMean", json.dumps(both))


def test_settings_compat_table_shipped_empty():
    with pytest.raises(screamer.SaveError, match="RollingMean.settings.window: unknown parameter"):
        screamer.from_settings({"type": "RollingMean", "settings": {"window": 7}})


def test_op_registry_fresh():
    gen = ROOT / "devtools" / "gen_op_registry.py"
    committed = (ROOT / "src" / "screamer" / "serial" / "op_registry.cpp").read_text(encoding="utf-8")
    assert run_generator(gen, ROOT) == committed, (
        "src/screamer/serial/op_registry.cpp is stale: run python devtools/gen_op_registry.py")


def _file_tree(text):
    doc = json.loads(text)
    doc["library_version"] = "*"
    return doc


def test_js_oracle_fresh():
    """wasm/smoke/save_oracle.json is what devtools/wasm/gen_save_oracle.py
    writes today. The numbers compare to 1e-9 (tests/_oracle_fresh.py) and
    the library version is ignored."""
    gen = ROOT / "devtools" / "wasm" / "gen_save_oracle.py"
    fixture = ROOT / "wasm" / "smoke" / "save_oracle.json"
    now = json.loads(run_generator(gen, ROOT))
    committed = json.loads(fixture.read_text(encoding="utf-8"))

    def comparable(doc):
        for c in doc["cases"]:
            c["json_text"] = _file_tree(c["json_text"])
            c["cbor_hex"] = _file_tree(_b._serial_cbor_decode(bytes.fromhex(c["cbor_hex"])))
        return doc

    diff = _oracle_same(comparable(now), comparable(committed))
    assert diff is None, (f"wasm/smoke/save_oracle.json is stale ({diff}): run python "
                          "devtools/wasm/gen_save_oracle.py and commit the result")
    assert [c["name"] for c in now["cases"]] == FUNCTOR_NAMES


def test_from_settings_takes_python_values():
    """Settings built in Python (NumPy integers, an operator object where a
    {type, settings} node belongs) are read as before, without a JSON pass."""
    a = screamer.from_settings({"type": "RollingMean", "settings": {"window_size": np.int64(5)}})
    assert a.settings()["window_size"] == 5
    with pytest.raises(screamer.SaveError, match=re.escape("Segment.settings.op: expected")):
        screamer.from_settings({"type": "Segment", "settings": {"op": screamer.RollingMean(3)}})


def test_nested_setting_errors_name_the_path():
    node = {"type": "Segment", "settings": {"op": {"type": "RollingMean", "settings": {"window": 3}}}}
    where = "Segment.settings.op.settings.window: unknown parameter"
    with pytest.raises(screamer.SaveError, match=re.escape(where)):
        screamer.from_settings(node)
    with pytest.raises(screamer.SaveError, match=re.escape(where)):
        _b._serial_make_op("Segment", json.dumps(node["settings"]))
    bad = '{"op": {"type": "RollingMean", "settings": {"window_size": 0}}}'
    with pytest.raises(screamer.SaveError, match=re.escape("Segment.settings.op.settings: ")):
        _b._serial_make_op("Segment", bad)


def _js_fixture():
    return json.loads((ROOT / "wasm" / "smoke" / "save_js.json").read_text(encoding="utf-8"))


_VERSION_KEY = b"\x6flibrary_version"   # CBOR: a text string of 15 bytes


def _cbor_with_library_version(blob, version):
    """``blob`` (a CBOR save file) with its library_version set to ``version``."""
    at = blob.index(_VERSION_KEY) + len(_VERSION_KEY)
    n = blob[at] - 0x60                      # a text string shorter than 24 bytes
    assert 0 <= n < 24 and len(version) < 24
    return blob[:at] + bytes([0x60 + len(version)]) + version.encode() + blob[at + 1 + n:]


def _check_js_file(c):
    """One file JavaScript wrote: Python loads it and writes the same bytes,
    in JSON and CBOR, but for the library version, which is this library's."""
    text, blob = c["json_text"], bytes.fromhex(c["cbor_hex"])
    want_text = _with_library_version(text)
    want_blob = _cbor_with_library_version(blob, screamer.__version__)
    a, b = screamer.loads(text), screamer.loads(blob)
    assert a.dumps() == want_text and b.dumps() == want_text
    assert a.dumps(format="cbor") == want_blob and b.dumps(format="cbor") == want_blob
    return a


# Operators whose JavaScript and Python outputs round apart by more (the TOL
# table of js/test/save_restore.test.ts says why).
_JS_TOL = {"BayesianRegression": 1e-8}


def test_js_files_load_byte_for_byte():
    """The files JavaScript wrote (js/scripts/gen_save_js.ts) load in Python
    and write back the same bytes, in JSON and CBOR, and continue as
    Python's own file does (to 1e-9: D4, see js/test/save_restore.test.ts)."""
    fixture = _js_fixture()
    assert [c["name"] for c in fixture] == FUNCTOR_NAMES, (
        "wasm/smoke/save_js.json is stale: run node --import tsx js/scripts/gen_save_js.ts")
    for c in fixture:
        a = _check_js_file(c)
        make, inputs = case(c["name"])
        k = splits(c["name"], kwargs_for(c["name"]))[2]
        ref = make()
        feed(ref, inputs, 0, k)
        tol = _JS_TOL.get(c["name"], 1e-9)
        np.testing.assert_allclose(feed(a, inputs, k, k + REST), feed(ref, inputs, k, k + REST),
                                   rtol=tol, atol=tol, equal_nan=True)


def test_js_files_of_another_library_version():
    """A fixture written by another library version still compares byte for
    byte (the version is the one field that differs)."""
    for c in _js_fixture():
        old = {"json_text": re.sub(r'"library_version": "[^"]*"', '"library_version": "0.0.1"',
                                   c["json_text"], count=1),
               "cbor_hex": _cbor_with_library_version(bytes.fromhex(c["cbor_hex"]), "0.0.1").hex()}
        assert '"0.0.1"' in old["json_text"] and b"0.0.1" in bytes.fromhex(old["cbor_hex"])
        _check_js_file(old)


# ---- Calls with one event on an object holding per-column state ---------------

def _panel_inputs(name, inputs):
    """Inputs of a (T, K) resume: two columns per input."""
    if any(np.ndim(x) != 1 for x in inputs):
        return None
    return [np.stack([x, x[::-1]], axis=1) for x in inputs]


@pytest.mark.parametrize("name", FUNCTOR_NAMES)
def test_event_calls_on_a_panel_raise(name):
    """After a resume on a (T, K) array every call with one event, and every
    stream, raises (decision 9) and keeps the per-column state."""
    if name in _sweep.BLOCK_OPS:
        pytest.skip(f"{name} continues its report on resume; it keeps no per-column state")
    make, inputs = case(name)
    panel = _panel_inputs(name, inputs)
    op = make()
    try:
        op.resume(*(panel or []))
    except (TypeError, ValueError):
        pytest.skip(f"{name} keeps no per-column state")
    if panel is None:
        pytest.skip(f"{name} keeps no per-column state")
    k = panel[0].shape[1]
    ev = [float(x[0, 0]) for x in panel]
    calls = [lambda: op(*ev), lambda: op(*[iter([v]) for v in ev])]
    if len(ev) > 1:
        calls += [lambda: op(tuple(ev)), lambda: op(iter([tuple(ev)]))]

    async def agen(v):
        yield v
    calls.append(lambda: op(*[agen(v) for v in ev]))
    for call in calls:
        with pytest.raises(ValueError, match=f"per-column state for {k} columns"):
            call()
    with pytest.raises(ValueError, match=f"holds {k} columns, got {k + 1}"):
        op.resume(*[np.concatenate([x, x[:, :1]], axis=1) for x in panel])
    op.resume(*panel)    # the columns are still there
