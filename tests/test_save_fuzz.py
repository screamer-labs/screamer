"""Fuzzed save files: edited or corrupted files load correctly or raise.

Each case saves an operator (or a live session) part way through its data,
then edits the file at random: a field deleted or added, a value of another
type, a number changed, an array made longer or shorter, a byte flipped or
dropped, the file cut short. JSON and CBOR files are both edited.

A load of the edited file must do one of two things:

* raise ``SaveError`` (a ``ValueError``), or
* give an object whose ``settings()`` and ``state()`` describe it fully: an
  object built from them with ``from_settings`` and ``set_state`` writes the
  same file and continues with the same outputs.

A failed ``set_state`` must leave the operator unchanged. A crash or a hang
(the watchdog stops the run after a minute) fails the run. The edits come
from a seeded loop, which is the path CI runs; Hypothesis is not a
dependency, and the Hypothesis branch at the end runs only where it is
installed.
"""
from __future__ import annotations

import faulthandler
import json
import random
import zlib

import numpy as np
import pytest

import screamer
from screamer import screamer_bindings as _b
from tests import save_cases
from tests.test_save_pipeline import CASES as SESSION_CASES, _events

try:
    from hypothesis import HealthCheck, given, settings as hyp_settings, strategies as st
except ImportError:            # the seeded loop below
    given = None

# Operators with every kind of state: rings, deques, order statistics, trees
# of nested operators, filters, models, labels with open entries, volume
# buckets, accounts with open orders, a row operator and a report of K assets.
OPERATORS = [
    "RollingMean", "EwMean", "RollingMax", "RollingQuantile", "RollingMedian",
    "Segment", "TripleBarrier", "TripleBarrierOHLC", "VPIN", "EdgeSpread",
    "RollingCorr", "RollingHMM", "KalmanFilter", "MAMA", "Butter", "CusumFilter",
    "FracDiff", "BayesianChangepoint", "RollingSeasonalZscore",
    "BacktestOHLCTarget", "BacktestL1Orders", "BacktestPortfolioTarget",
    "PortfolioReport",
]
PANEL = "RollingMean panel"
SESSIONS = ["Delay into CombineLatest", "Resample freq into CombineLatest",
            "operator of two streams"]
CASE_NAMES = OPERATORS + [PANEL] + [f"session: {s}" for s in SESSIONS]

# 2,000 edits in all.
PER_CASE = -(-2000 // len(CASE_NAMES))

# Values put in place of another value.
OTHER_VALUES = [None, True, False, "text", "", [], {}, 0, 1, -1, 1.5, [1.0], {"a": 1},
                [[1.0]], float("nan")]
# Numbers put in place of a number. A setting gets no integer above 1000 that
# fits a C++ int: a window of 2**31 - 1 is valid and allocates 16 GB, and a
# filter of a high order takes hours to design, as the constructors would.
NUMBERS = [0, 1, 2, 3, -1, -2, 0.5, -0.0, 1e-300, float("nan"), float("inf"), float("-inf"),
           1e308, -1e308, 10**6, 2**53 + 1, 2**63 - 1, 2**64, -2**63, -2**64]
STATE_NUMBERS = NUMBERS + [2**31 - 1, 2**32, 10**12]


# ---- The saved files ---------------------------------------------------------

def _rest_slice(inputs, start, stop):
    return [x[start:stop] for x in inputs]


class _OperatorCase:
    """An operator saved after part of its inputs; the rest continues it."""

    def __init__(self, name):
        self.panel = False
        if name != PANEL:
            make, inputs = save_cases.case(name)
            split = max(save_cases.splits(name, save_cases.kwargs_for(name)))
        else:
            x = np.stack([save_cases.case("RollingMean")[1][0] + k for k in range(3)], axis=1)
            make, inputs, split = (lambda: screamer.RollingMean(5)), [x], 30
            self.panel = True
        op = make()
        if self.panel:
            op.resume(x[:split])
        else:
            save_cases.feed(op, inputs, 0, split)
        self.text = op.dumps()
        self.blob = op.dumps(format="cbor")
        self.rest = _rest_slice(inputs, split, split + 12)

    def run(self, op):
        """The outputs of the rest, or the error it raised."""
        try:
            if self.panel:
                return np.asarray(op.resume(self.rest[0]))
            return save_cases.feed(op, self.rest, 0, len(self.rest[0]))
        except Exception as e:                       # compared with the twin's
            return (type(e), str(e))


class _SessionCase:
    """A live session saved with events waiting; the rest continues it."""

    def __init__(self, name):
        pipe, feeds = SESSION_CASES[name]
        first, self.rest = _events(feeds, "blocked")
        s = pipe.live()
        for k, t, v in first:
            s.push(k, t, v)
        self.text = s.dumps()
        self.blob = s.dumps(format="cbor")
        self.rest = self.rest[:40]

    def run(self, session):
        try:
            for k, t, v in self.rest:
                session.push(k, t, v)
            session.flush()
            return session.result()
        except Exception as e:
            return (type(e), str(e))


_CASES: dict = {}


def _case(name):
    if name not in _CASES:
        _CASES[name] = (_SessionCase(name[len("session: "):]) if name.startswith("session: ")
                        else _OperatorCase(name))
    return _CASES[name]


# ---- Edits -------------------------------------------------------------------

def _places(tree, path=()):
    """Every (container, key, path) below ``tree``."""
    items = tree.items() if isinstance(tree, dict) else enumerate(tree)
    for k, v in items:
        yield tree, k, path + (k,)
        if isinstance(v, (dict, list)):
            yield from _places(v, path + (k,))


def _edit_tree(tree, rng):
    """Edit one place of the parsed file in place; a description of the edit."""
    places = list(_places(tree))
    kind = rng.choice(["delete", "add", "retype", "number", "number", "length"])
    if kind == "delete":
        dicts = [(c, k, p) for c, k, p in places if isinstance(c, dict)]
        c, k, p = rng.choice(dicts)
        del c[k]
        return f"delete {p}"
    if kind == "add":
        dicts = [tree] + [c[k] for c, k, _ in places if isinstance(c[k], dict)]
        d = rng.choice(dicts)
        d["extra_field"] = rng.choice(OTHER_VALUES)
        return "add extra_field"
    if kind == "retype":
        c, k, p = rng.choice(places)
        c[k] = rng.choice(OTHER_VALUES)
        return f"retype {p} -> {c[k]!r}"
    if kind == "number":
        numbers = [(c, k, p) for c, k, p in places
                   if isinstance(c[k], (int, float)) and not isinstance(c[k], bool)]
        if not numbers:
            return _edit_tree(tree, rng)
        c, k, p = rng.choice(numbers)
        pool = NUMBERS if "settings" in p else STATE_NUMBERS
        old = c[k]
        c[k] = rng.choice(pool + [old + 1, old - 1, old * 2])
        if "settings" in p and isinstance(c[k], int) and 1000 < abs(c[k]) < 2**31:
            c[k] = 1000         # Butter(order=10**6) takes hours to design
        return f"number {p}: {old!r} -> {c[k]!r}"
    lists = [(c, k, p) for c, k, p in places if isinstance(c[k], list)]
    if not lists:
        return _edit_tree(tree, rng)
    c, k, p = rng.choice(lists)
    a = c[k]
    how = rng.choice(["drop", "append", "double", "empty", "long"])
    if how == "drop" and a:
        a.pop(rng.randrange(len(a)))
    elif how == "append":
        a.append(a[-1] if a and rng.random() < 0.7 else rng.choice(OTHER_VALUES))
    elif how == "double":
        a.extend(list(a))
    elif how == "empty":
        a.clear()
    else:
        a.extend([a[-1] if a else 0.0] * 500)
    return f"length {p}: {how}"


def _edit_bytes(data, rng):
    data = bytearray(data)
    how = rng.choice(["flip", "drop", "truncate"])
    i = rng.randrange(len(data))
    if how == "flip":
        data[i] ^= 1 << rng.randrange(8)
    elif how == "drop":
        del data[i]
    else:
        data = data[:i]
    return bytes(data), f"{how} byte {i}"


def _edited_file(case, rng):
    """An edited copy of the case's file: ``str`` or ``bytes``, and what was done."""
    fmt = rng.choice(["json", "cbor"])
    if rng.random() < 0.6:
        tree = json.loads(case.text)
        what = _edit_tree(tree, rng)
        text = json.dumps(tree, indent=1)
        if fmt == "json":
            return text, f"json {what}"
        return _b._serial_cbor_encode(text), f"cbor {what}"
    data = case.text.encode("utf-8") if fmt == "json" else case.blob
    data, what = _edit_bytes(data, rng)
    return data, f"{fmt} {what}"


# ---- Checks ------------------------------------------------------------------

def _same(a, b):
    """Outputs equal bit for bit, with every NaN counted equal (so -0.0 and
    0.0 differ), or the same error."""
    if isinstance(a, tuple) and a and isinstance(a[0], type):
        return a == b
    if isinstance(a, tuple):
        return isinstance(b, tuple) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(b, tuple) and b and isinstance(b[0], type):
        return False
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind == "f":
        a, b = a.astype(np.float64), b.astype(np.float64)
        a[np.isnan(a)] = np.nan
        b[np.isnan(b)] = np.nan
        return bool(np.array_equal(a.view(np.uint64), b.view(np.uint64)))
    return bool(np.array_equal(a, b))


def _load(data):
    """The loaded object, or None when the load raised a SaveError."""
    try:
        return screamer.loads(data)
    except screamer.SaveError:
        return None


def _twin(case, loaded):
    """An object built from the loaded one's settings and state."""
    if isinstance(case, _SessionCase):
        return screamer.loads(loaded.dumps())
    twin = screamer.from_settings({"type": type(loaded).__name__,
                                   "settings": loaded.settings()})
    twin.set_state(loaded.state())
    return twin


def _check_load(case, data, what):
    loaded = _load(data)
    if loaded is None:
        return
    twin = _twin(case, loaded)
    assert twin.dumps() == loaded.dumps(), f"{what}: the twin writes another file"
    assert _same(case.run(loaded), case.run(twin)), f"{what}: the twin continues differently"


def _check_set_state(case, rng):
    """Edit the state tree and load it into a saved operator: it loads and
    describes itself, or raises and keeps its old state."""
    op = screamer.loads(case.text)
    before = op.dumps()
    tree = op.state()
    what = _edit_tree(tree, rng)
    try:
        op.set_state(tree)
    except screamer.SaveError:
        assert op.dumps() == before, f"set_state {what}: a failed load changed the operator"
        assert _same(case.run(op), case.run(screamer.loads(before))), what
        return
    twin = _twin(case, op)
    assert twin.dumps() == op.dumps(), f"set_state {what}: the twin writes another file"
    assert _same(case.run(op), case.run(twin)), f"set_state {what}: the twin continues differently"


def _one(name, rng):
    case = _case(name)
    data, what = _edited_file(case, rng)
    _check_load(case, data, f"{name}: {what}")
    if isinstance(case, _OperatorCase) and rng.random() < 0.3:
        _check_set_state(case, rng)


@pytest.fixture(autouse=True)
def _watchdog():
    """Stop the run with a traceback when one case hangs."""
    faulthandler.dump_traceback_later(60, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


def test_cases_save_rich_states():
    """The saved files hold real state: each case loads and continues."""
    for name in CASE_NAMES:
        case = _case(name)
        loaded = screamer.loads(case.text)
        assert loaded.dumps() == case.text
        assert screamer.loads(case.blob).dumps() == case.text
        out = case.run(loaded)
        assert not (isinstance(out, tuple) and out and isinstance(out[0], type)), (name, out)


if given is not None:
    @pytest.mark.parametrize("name", CASE_NAMES)
    @hyp_settings(max_examples=PER_CASE, deadline=None, derandomize=True,
                  suppress_health_check=list(HealthCheck))
    @given(rng=st.randoms(use_true_random=False))
    def test_edited_files_load_or_raise(name, rng):
        _one(name, rng)
else:
    @pytest.mark.parametrize("name", CASE_NAMES)
    def test_edited_files_load_or_raise(name):
        rng = random.Random(zlib.crc32(name.encode("utf-8")))
        for _ in range(PER_CASE):
            _one(name, rng)


def test_triple_barrier_long_price_array_refused():
    """A ring longer than the horizon is refused before its entries are read."""
    case = _case("TripleBarrier")
    tree = json.loads(case.text)
    values = tree["object"]["state"]["values"]
    values["price"] = values["price"] * 1000
    with pytest.raises(screamer.SaveError, match="TripleBarrier"):
        screamer.loads(json.dumps(tree))
