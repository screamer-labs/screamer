"""Operators and inputs for the save and restore tests.

``continuation_cases()`` lists every functor of the registry (help.json) with
a constructor and inputs: the arguments ``tests/test_batch_stream_bitwise.py``
builds it with, and that file's related random walks (NaN in the second half)
in the input order that gives the most informative output. The save tests,
the golden generator (``devtools/gen_state_golden.py``) and the JavaScript
oracle run on these cases.

``KNOWN_UNSAVEABLE`` lists the functors without save support yet. Their tests
are ``xfail(strict=True)``, so an operator that gains support fails the suite
until it leaves the set; the set is empty once every operator is migrated.
"""
from __future__ import annotations

import functools
import importlib.resources
import json

import numpy as np

import screamer
from tests import test_batch_stream_bitwise as _sweep

HELP: dict[str, dict] = json.loads(
    importlib.resources.files("screamer").joinpath("data/help.json").read_text(encoding="utf-8"))

FUNCTOR_NAMES = sorted(n for n, e in HELP.items() if e.get("kind", "functor") == "functor")

KNOWN_UNSAVEABLE: set[str] = {
    'AbdiRanaldoSpread', 'AmihudIlliquidity', 'BandSignal', 'BayesianChangepoint',
    'BayesianRegression', 'BulkVolumeClassifier', 'ContOFI', 'CorwinSchultzSpread',
    'Detrend', 'Diff', 'Diff2', 'Drawdown', 'EdgeSpread', 'EffectiveSpread',
    'EwKyleLambda', 'EwMultiRegression', 'ExpandingMultiRegression', 'First',
    'HawkesIntensity', 'KalmanRegression', 'Lag', 'Last', 'LeeReadySign', 'LogReturn',
    'MaxDrawdown', 'MicroPrice', 'Momentum', 'OFI', 'Propagator', 'QueueImbalance',
    'ROC', 'ROCP', 'ROCR', 'RealizedSpread', 'Return', 'RollSpread', 'RollingADF',
    'RollingAlpha', 'RollingBeta', 'RollingCVaR', 'RollingCalmar', 'RollingCorr',
    'RollingCov', 'RollingDownsideDeviation', 'RollingHitRate', 'RollingInfoRatio',
    'RollingKyleLambda', 'RollingLinearRegression', 'RollingMaxDrawdown',
    'RollingMultiRegression', 'RollingOmega', 'RollingOrderImbalance',
    'RollingResidualStd', 'RollingSharpe', 'RollingSortino', 'RollingSpread',
    'RollingTLSBeta', 'RollingUpsideDeviation', 'SignedVolume', 'TickRuleSign',
    'TripleBarrier', 'TripleBarrierOHLC', 'VPIN'
}

# Arguments that differ from the sweep's: Segment runs an operator that can be
# saved.
_KWARGS = {
    "Segment": lambda: {"op": screamer.RollingMean(5)},
}

# Operators whose first input is a segment label: runs of 12 rows, so a
# segment spans a save point and the nested state continues.
_SEGMENT_LABELS = {"Segment"}

# Events after the last split point.
REST = 50


def window(name: str, kwargs: dict) -> int | None:
    """The operator's window length, or None when it has none."""
    for p in HELP[name].get("parameters") or []:
        if p["name"] in ("window_size", "window"):
            return int(kwargs.get(p["name"], p.get("default")))
    return None


def splits(name: str, kwargs: dict) -> tuple[int, int, int]:
    """Save points: at construction, mid warm-up and after two windows."""
    w = window(name, kwargs)
    return (0, 3, 2 * w if w else 50)


def kwargs_for(name: str) -> dict:
    if name in _KWARGS:
        return _KWARGS[name]()
    return dict(_sweep.default_kwargs(name) or {})


@functools.lru_cache(maxsize=None)
def case(name: str):
    """(make, inputs) for ``name``: ``make()`` builds a fresh instance and
    ``inputs`` holds one array per input, time first. A row operator's inputs
    are (T, K) arrays, PortfolioReport's one (T, assets, 4) array."""
    def make():
        return getattr(screamer, name)(**kwargs_for(name))

    t = max(splits(name, kwargs_for(name))) + REST
    if name in _sweep.BLOCK_OPS:
        x = np.stack([np.stack(_sweep.series(7 * a, 1e3, t=t)[:4], axis=1) for a in range(3)],
                     axis=1)
        return make, [x]
    if name in _sweep.ROW_OPS:
        probe = [np.stack(_sweep.series(k + 50, 0.0), axis=1) for k in range(_sweep.MAX_INPUTS)]
        n = next(m for m in range(1, _sweep.MAX_INPUTS + 1) if _calls(make, probe[:m]))
        return make, [np.stack(_sweep.series(31 * i, 1e3, t=t), axis=1) for i in range(n)]
    if name in _SEGMENT_LABELS:
        x = _sweep.series(0, 1e3, t=t)
        return make, [np.repeat(np.arange(t // 12 + 1.0), 12)[:t], x[3]]
    found = _sweep.inputs_for(make)
    assert found is not None, f"{name}: no array call with up to {_sweep.MAX_INPUTS} inputs"
    n, layout = found
    return make, _sweep.series(0, 1e3, layout, t=t)[:n]


def _calls(make, arrays) -> bool:
    try:
        make()(*arrays)
        return True
    except Exception:
        return False


def continuation_cases():
    """Yield ``(name, make, inputs)`` for every functor in the registry."""
    for name in FUNCTOR_NAMES:
        make, inputs = case(name)
        yield name, make, inputs


def feed(op, inputs, start: int, stop: int) -> np.ndarray:
    """Feed events ``start:stop`` one at a time; the outputs, one row per event."""
    rows = []
    for t in range(start, stop):
        event = [float(x[t]) if np.ndim(x) == 1 else x[t] for x in inputs]
        rows.append(np.asarray(op(*event), dtype=np.float64).ravel())
    return np.stack(rows) if rows else np.empty((0, 0))
