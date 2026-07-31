"""Every operator must have an independent reference, or a written reason it cannot.

The rest of the suite checks that an operator agrees with *itself*: batch equals
stream, the three regimes agree, `reset()` is clean, the declared NaN and start
policies hold, the shapes are right. None of that can catch an operator that
computes the wrong quantity consistently, and two did:

* `EwSkew` and `EwKurt` divide their result by roughly `n_eff`, so the value
  shrinks as the window lengthens. On exponential data (true skew 2.0),
  `EwSkew(span=100)` returns 0.019. Every self-consistency test passed.
* Their baseline comparison, the one test class that could have caught it, was
  disabled in a code comment reading `# EwSkew, EwKurt: todo baselines`, and CI
  stayed green.

A disabled correctness check must be as loud as a failing one. This file makes
the gap explicit and stops it growing: a new operator needs a baseline, or an
entry here saying why not. The dict can shrink and never silently grow.

`BASELINE_GAP` is not a list of broken operators. It is the list of operators
whose numbers nothing independently verifies.
"""
from __future__ import annotations

import json
from pathlib import Path

from devtools import baselines, get_baselines

HELP_JSON = Path(__file__).resolve().parent.parent / "screamer" / "data" / "help.json"

# An execution simulator: the fill model is screamer's own, so there is no
# external library computing the same thing. These want hand-written
# known-answer tests instead, which they have in tests/test_backtest.py.
_NO_EXTERNAL = "execution simulator; no external library implements this fill model"

# No reference written yet. This is the worklist.
_TODO = "no reference implementation written yet"

BASELINE_GAP: dict[str, str] = {
    "AmihudIlliquidity": _TODO,
    "BacktestL1Orders": _NO_EXTERNAL,
    "BacktestL1Target": _NO_EXTERNAL,
    "BacktestL1TradesOrders": _NO_EXTERNAL,
    "BacktestOHLCOrders": _NO_EXTERNAL,
    "BacktestOHLCTarget": _NO_EXTERNAL,
    "BacktestPriceTarget": _NO_EXTERNAL,
    "BacktestReport": _NO_EXTERNAL,
    "BacktestTradesOrders": _NO_EXTERNAL,
    "BacktestTradesTarget": _NO_EXTERNAL,
    "BayesianRegression": _TODO,
    "BollingerBands": _TODO,
    "BulkVolumeClassifier": _TODO,
    "ContOFI": _TODO,
    "CycleAmplitude": _TODO,
    "CycleFrequency": _TODO,
    "CyclePhase": _TODO,
    "CycleSine": _TODO,
    "DominantCycle": _TODO,
    "DonchianChannels": _TODO,
    "EffectiveSpread": _TODO,
    "EwKyleLambda": _TODO,
    "Hampel": _TODO,
    "HilbertPhasor": _TODO,
    "ImpulseClip": _TODO,
    "InstantaneousTrendline": _TODO,
    "KeltnerChannels": _TODO,
    "LeeReadySign": _TODO,
    "MicroPrice": _TODO,
    "OFI": _TODO,
    "Propagator": _TODO,
    "QueueImbalance": _TODO,
    "RealizedSpread": _TODO,
    "RollSpread": _TODO,
    "RollingAlpha": _TODO,
    "RollingCVaR": _TODO,
    "RollingCalmar": _TODO,
    "RollingDownsideDeviation": _TODO,
    "RollingHurst": _TODO,
    "RollingInfoRatio": _TODO,
    "RollingKyleLambda": _TODO,
    "RollingLinearRegression": _TODO,
    "RollingMaxDrawdown": _TODO,
    "RollingMinMax": _TODO,
    "RollingOmega": _TODO,
    "RollingOrderImbalance": _TODO,
    "RollingPercentile": _TODO,
    "RollingRank": _TODO,
    "RollingResidualStd": _TODO,
    "RollingSharpe": _TODO,
    "RollingSortino": _TODO,
    "RollingSpread": _TODO,
    "RollingTSF": _TODO,
    "SignedVolume": _TODO,
    "TickRuleSign": _TODO,
    "TrendMode": _TODO,
    "VPIN": _TODO,
}


def _functors() -> dict[str, dict]:
    return {
        name: entry
        for name, entry in json.loads(HELP_JSON.read_text()).items()
        if entry.get("kind", "functor") == "functor"
    }


def test_every_functor_has_a_baseline_or_a_stated_reason():
    """A new operator cannot land with neither."""
    unaccounted = sorted(
        name for name in _functors()
        if not get_baselines(name) and name not in BASELINE_GAP
    )
    assert not unaccounted, (
        f"operators with no independent reference and no stated reason: {unaccounted}. "
        "Add devtools/baselines/<Name>.py with a class <Name>_<lib>, or add the name "
        "to BASELINE_GAP with the reason none can be written."
    )


def test_gap_entries_carry_a_reason():
    blank = sorted(name for name, reason in BASELINE_GAP.items() if not reason.strip())
    assert not blank, f"BASELINE_GAP entries with no reason: {blank}"


def test_gap_entries_are_live():
    """An entry naming something that is not a functor is stale."""
    stale = sorted(set(BASELINE_GAP) - set(_functors()))
    assert not stale, f"BASELINE_GAP names non-functors: {stale}. Remove them."


def test_gap_shrinks_when_a_baseline_lands():
    """An operator that now has a baseline must come off the list.

    Without this the list would keep its entry forever and stop reflecting
    reality, which is how `# todo baselines` survived.
    """
    resolved = sorted(name for name in BASELINE_GAP if get_baselines(name))
    assert not resolved, (
        f"these now have a baseline and must be removed from BASELINE_GAP: {resolved}"
    )


def test_optional_dependency_skips_are_visible():
    """A baseline skipped for a missing library is reported, not silent.

    `devtools/baselines/__init__.py` skips a baseline whose optional dependency
    (TA-Lib, say) is not installed, so that one missing C library cannot take
    down collection for the whole suite. This asserts the skip is recorded, so
    `get_baselines` returning nothing is explainable rather than mysterious.
    """
    skipped = baselines.missing_dependencies()
    assert isinstance(skipped, dict)
    for module_name, reason in skipped.items():
        assert reason, f"{module_name} was skipped with no recorded reason"
