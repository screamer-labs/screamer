"""Compliance tests for the documented ``nan_policy`` contract.

Every function whose docs page declares a ``nan_policy`` field must honor that
policy at runtime. This file enumerates ``screamer/data/help.json`` and runs
per-policy assertions on each function. The canonical contract is documented
in ``docs/nan_policy.md``.

The original symptom that motivated this test suite: leading-NaN inputs
(typically warmup NaNs produced by an upstream EW function) silently poisoned
the running-sum state of downstream rolling/EW functions, making them emit
all-NaN forever.

Two independent properties are checked per function:

1. **No sticky NaN** (:func:`test_no_sticky_nan`). Three leading NaN inputs
   followed by ~500 finite samples; the last output must be finite. This
   catches the original poisoning bug.

2. **NaN propagates at the NaN index** (:func:`test_nan_at_nan_index`). For
   single mid-stream NaN, the output at that same index must be NaN. This
   catches functions that silently drop NaN samples without emitting the
   NaN-in-NaN-out marker that the policy guarantees.

Functions known to violate either property are listed in the corresponding
``KNOWN_*`` set below and marked ``xfail(strict=True)``. When a function is
fixed, removing it from the set is mandatory -- ``strict=True`` causes the
test suite to fail loudly if a formerly-broken function starts passing
without an update to this list.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

import screamer

HELP_JSON = Path(__file__).resolve().parent.parent / "screamer" / "data" / "help.json"

# Functions whose runtime behavior does not yet match their declared
# ``nan_policy``: leading-NaN inputs poison running state forever. Each
# entry here is a bug to fix. Remove from the set as fixes land;
# ``strict=True`` on the xfail mark will fail the build if a formerly-broken
# function starts passing without an update.
KNOWN_STICKY_NAN: set[str] = {
    # Composite smoothers/bands that use running-sum primitives internally.
    "BollingerBands",
    "DEMA",
    "Detrend",
    "HullMA",
    "KAMA",
    "KeltnerChannels",
    "MACD",
    "TEMA",
    "TRIMA",
    "TRIX",
    "WMA",
    # Rolling family with running-sum state.
    "RollingAlpha",
    "RollingBeta",
    "RollingCalmar",
    "RollingCorr",
    "RollingCov",
    "RollingInfoRatio",
    "RollingKurt",
    "RollingLinearRegression",
    "RollingMad",
    "RollingMean",
    "RollingOU",
    "RollingPoly1",
    "RollingPoly2",
    "RollingResidualStd",
    "RollingRms",
    "RollingSharpe",
    "RollingSkew",
    "RollingSortino",
    "RollingSpread",
    "RollingStd",
    "RollingSum",
    "RollingTSF",
    "RollingVar",
    "RollingZscore",
    # EW family with running-sum state.
    "EwBeta",
    "EwCorr",
    "EwCov",
    "EwGarmanKlassVar",
    "EwGarmanKlassVol",
    "EwKurt",
    "EwMean",
    "EwParkinsonVar",
    "EwParkinsonVol",
    "EwRms",
    "EwRogersSatchellVar",
    "EwRogersSatchellVol",
    "EwSkew",
    "EwStd",
    "EwVar",
    "EwZscore",
    # OHLC volatility estimators built on RollingMean.
    "CCI",
    "RollingGarmanKlassVar",
    "RollingGarmanKlassVol",
    "RollingParkinsonVar",
    "RollingParkinsonVol",
    "RollingRogersSatchellVar",
    "RollingRogersSatchellVol",
    "RollingVWAP",
    "RollingYangZhangVar",
    "RollingYangZhangVol",
    "Stoch",
    # Cumulative -- currently propagate-forever; the policy says ignore.
    "CumMax",
    "CumMin",
    "CumProd",
    "CumSum",
    # IIR/state filters: state corrupted permanently by NaN.
    "Butter",
    "ButterBandpass",
    "ButterBandstop",
    "ButterHighpass",
    "KalmanFilter",
    # Drawdown family: peak tracking poisoned by NaN.
    "Drawdown",
    "MaxDrawdown",
}


# Functions that silently drop NaN samples without emitting NaN at the same
# output index. Per ``docs/nan_policy.md``, every ignore-policy function must
# produce NaN at the index where input is NaN; functions here violate that.
# Categories:
#   * Stateless math that returns 0 (or some other finite value) for NaN
#     input rather than letting IEEE arithmetic propagate.
#   * Window-extremum / OST-based functions whose state correctly skips NaN
#     but whose output at the NaN index returns the rolling extremum of the
#     remaining finite samples instead of NaN.
#   * Multi-input financial composites where the NaN of one input is masked
#     by the formula.
KNOWN_NO_NAN_PROPAGATION: set[str] = {
    # Stateless math: returns 0 instead of NaN for NaN input.
    "Relu",
    "Sign",
    # OST / deque-based window extremes: skip NaN from state but compute
    # output from remaining finite samples instead of returning NaN.
    "RollingArgmax",
    "RollingArgmin",
    "RollingMax",
    "RollingMedian",
    "RollingMin",
    "RollingMinMax",
    "RollingPercentile",
    "RollingQuantile",
    "RollingRange",
    "RollingRank",
    "DonchianChannels",
    # OHLC / volume composites whose NaN-handling masks input NaN.
    "AD",
    "ADOSC",
    "ADX",
    "MFI",
    "OBV",
    "RollingRSI",
    "StochRSI",
    "UltimateOscillator",
}


# Functions whose declared ``propagate`` policy formula does not reference
# x[t] at output index t. ``Lag(n)`` returns ``x[t-n]``; a NaN at index t
# legitimately appears as NaN at output index t+n, not at index t. The
# nan-at-nan-index test is skipped for these because the universal
# invariant doesn't apply.
PROPAGATE_NOT_AT_T: set[str] = {"Lag"}


def _load_help() -> dict[str, dict]:
    return json.loads(HELP_JSON.read_text())


HELP = _load_help()


def _instantiate(entry: dict):
    cls = getattr(screamer, entry["name"])
    kwargs = {p["name"]: p["default"] for p in (entry.get("parameters") or [])}
    return cls(**kwargs)


def _call(instance, arrays: list[np.ndarray]):
    if len(arrays) == 1:
        return instance(arrays[0])
    return instance(*arrays)


def _last_finite(out) -> bool:
    """True iff every output stream's last element is finite.

    Single-output functions return a 1-D ndarray. Two-output functions
    (RollingMinMax, DonchianChannels, Cart2Polar, Polar2Cart, ...) return
    a 2-D ``(N, 2)`` array. Multi-output functions are also possible
    via a tuple in principle.
    """
    if isinstance(out, tuple):
        return all(_last_finite(o) for o in out)
    if out.ndim == 2:
        return bool(np.all(np.isfinite(out[-1])))
    return math.isfinite(out[-1])


def _all_nan_at_index(out, idx: int) -> bool:
    """True iff every output stream is NaN at ``idx``."""
    if isinstance(out, tuple):
        return all(_all_nan_at_index(o, idx) for o in out)
    if out.ndim == 2:
        return bool(np.all(np.isnan(out[idx])))
    return math.isnan(out[idx])


def _build_input(n_inputs: int, n_samples: int) -> list[np.ndarray]:
    """Three leading NaN, then finite samples in ``(0.1, 0.9)``.

    The bounded-positive range keeps every function's domain happy:
    ``Log`` and ``Sqrt`` get positive input, ``Asin`` / ``Acos`` get
    ``|x| <= 1``, ``Power`` gets positive base. Using identical arrays
    across multi-input mirrors what build_help_registry does for its
    smoke test, so this stays aligned with the "documented defaults"
    contract.
    """
    rng = np.random.default_rng(42)
    base = rng.uniform(0.1, 0.9, size=n_samples).astype(np.float64)
    base[:3] = np.nan
    return [base.copy() for _ in range(n_inputs)]


def _xfail_if_in(known: set[str], reason_label: str):
    """Return a per-name marker factory tied to a specific failure set."""
    def factory(name: str):
        if name in known:
            return pytest.mark.xfail(
                strict=True,
                reason=(
                    f"{name}: {reason_label}; see docs/nan_policy.md. "
                    "Remove from the corresponding KNOWN_* set when fixed."
                ),
            )
        return ()
    return factory


_xfail_sticky = _xfail_if_in(KNOWN_STICKY_NAN, "sticky-NaN regression")
_xfail_propagation = _xfail_if_in(
    KNOWN_NO_NAN_PROPAGATION, "NaN-at-NaN-index propagation broken"
)


PARAMS_STICKY = [
    pytest.param(name, entry, id=name, marks=_xfail_sticky(name))
    for name, entry in sorted(HELP.items())
]
PARAMS_PROPAGATION = [
    pytest.param(name, entry, id=name, marks=_xfail_propagation(name))
    for name, entry in sorted(HELP.items())
]


@pytest.mark.parametrize("name,entry", PARAMS_STICKY)
def test_no_sticky_nan(name: str, entry: dict):
    """Three leading NaN inputs must not poison state forever.

    For ``ignore`` and ``propagate`` policies, the last output of a 500-sample
    stream (3 leading NaN + 497 finite) must be finite. ``nan-aware`` is
    skipped -- those functions have function-specific contracts documented
    on their own pages.
    """
    policy = entry["nan_policy"]
    if policy == "nan-aware":
        pytest.skip("nan-aware policy is function-specific; not covered here")

    n_inputs = int(entry.get("inputs", 1))
    arrays = _build_input(n_inputs, n_samples=500)
    instance = _instantiate(entry)
    out = _call(instance, arrays)

    assert _last_finite(out), (
        f"{name} ({policy}): last output is NaN. Three leading-NaN inputs "
        f"poisoned internal state for all 497 subsequent finite samples. "
        "See docs/nan_policy.md."
    )


@pytest.mark.parametrize("name,entry", PARAMS_PROPAGATION)
def test_nan_at_nan_index(name: str, entry: dict):
    """Output at any index where input is NaN must itself be NaN.

    Holds for every ``ignore``-policy function and every ``propagate``-policy
    function whose formula references x[t] at output index t (i.e., all of
    Diff / Diff2 / Momentum / ROC / ROCP / ROCR / LogReturn / Return -- not
    Lag, whose formula uses x[t-n]).

    For multi-input functions, the input NaN is set in all input streams at
    the same index.

    Two skip categories:
    * ``KNOWN_STICKY_NAN`` functions are skipped: their outputs are already
      all-NaN because of the sticky-NaN bug, so this property "trivially
      holds" in a way that tells us nothing. When the sticky-NaN fix lands
      and the function is removed from that set, this test starts checking
      it for real.
    * :data:`PROPAGATE_NOT_AT_T` functions are skipped because the universal
      invariant doesn't apply to them by design.
    """
    if name in KNOWN_STICKY_NAN:
        pytest.skip(
            f"{name} is on the KNOWN_STICKY_NAN list; output is all-NaN "
            "until that bug is fixed, so this test would trivially pass."
        )
    if name in PROPAGATE_NOT_AT_T:
        pytest.skip(
            f"{name}'s propagate formula uses x[t-n] at index t; NaN at "
            "input index t propagates to output index t+n, not t."
        )
    policy = entry["nan_policy"]
    if policy == "nan-aware":
        pytest.skip("nan-aware policy is function-specific; not covered here")

    n_inputs = int(entry.get("inputs", 1))
    arrays = _build_input(n_inputs, n_samples=500)
    # Add a NaN deep in the middle, far from warmup and far from end.
    nan_idx = 250
    for a in arrays:
        a[nan_idx] = np.nan
    instance = _instantiate(entry)
    out = _call(instance, arrays)

    assert _all_nan_at_index(out, nan_idx), (
        f"{name} ({policy}): output at index {nan_idx} is not NaN, "
        "but the input at that index is NaN. Every policy except "
        "nan-aware must emit NaN at the same index as the input NaN."
    )
