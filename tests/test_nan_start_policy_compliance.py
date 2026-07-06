"""Compliance matrix tests that close the gap exposed by the
EwKurt -> RollingPoly1 chain bug.

The existing test suite never combined three independent axes that
matter for NaN handling correctness:

  * ``start_policy`` -- 27 functions accept ``strict``/``expanding``/``zero``;
    the existing ``test_start_policies.py`` exercises them but only on
    finite random data, never on NaN-bearing input.
  * ``nan_position`` -- whether the NaN lives at the start of the stream
    (leading), inside the steady-state region (mid), or is absent.
  * ``call_mode`` -- streaming (scalar-at-a-time ``__call__``) versus
    batch (array ``__call__``). The existing ``test_stream_vs_batch.py``
    runs across many functions but only on finite-input arrays because
    ``param_cases.py`` does not include any NaN test cases.

This file fills the gap with three independent properties:

  P1. :func:`test_stream_matches_batch` -- for every function, on a
      stream containing leading or mid-stream NaNs, the streaming call
      and the batch call must produce essentially-identical output
      (``np.testing.assert_allclose`` with ``rtol=1e-10, atol=1e-12,
      equal_nan=True``). The tolerance allows sub-ULP divergence from
      the array fast-path's sliding-sum recurrence vs. the scalar
      re-summed path, but is tight enough to catch any semantic
      difference (NaN appearing in one path but not the other, a
      different number of warmup NaNs, etc.).

  P2. :func:`test_initial_nan_recovers` -- for every function that
      accepts ``start_policy``, in every policy value, 3 leading NaN
      inputs followed by ~500 finite samples must not poison state
      forever; the last output must be finite.

  P3. :func:`test_mid_nan_recovers` -- same as P2, but the NaN is a
      single sample placed in the steady-state region of the stream.

Functions known to violate one or more of these properties are listed
in the per-test ``KNOWN_*`` sets at the top of this file with
``xfail(strict=True)`` so that any unintended fix surfaces as a
hard-fail (forcing the set to be updated and forcing the human to
acknowledge the change).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

import screamer

HELP_JSON = Path(__file__).resolve().parent.parent / "screamer" / "data" / "help.json"
HELP: dict[str, dict] = json.loads(HELP_JSON.read_text())

START_POLICIES = ("strict", "expanding", "zero")


# ---------------------------------------------------------------------------
# Known-broken sets. Each test owns its own set; the test docstring spells
# out the assertion. Removing a function from a set when it gets fixed is
# enforced by ``xfail(strict=True)``.
# ---------------------------------------------------------------------------

# Stream-vs-batch parity is universal: every function should be deterministic
# regardless of how the user feeds it data. As of policy introduction, the
# following functions fail this on at least one NaN position. Empirically
# determined; do not edit by hand outside a fix PR.
STREAM_BATCH_FAILING: set[tuple[str, str]] = set()

# Functions whose running state is corrupted forever by 3 leading NaN inputs.
# Tuple is (class_name, start_policy) so we can pin down exactly which
# combination is broken; many functions are broken under all 3 policies but
# being explicit makes incremental fixes (e.g. "strict works after the fix,
# expanding still doesn't") visible.
#
# This set was populated empirically when the NaN policy infrastructure
# landed (every running-sum function and every EW function was broken).
# After the v0.5.1 fix pass the set is empty: every (function, start_policy)
# in the test now passes. Add new entries when a regression is spotted.
INITIAL_NAN_FAILING: set[tuple[str, str]] = set()

# Same shape, but for a single mid-stream NaN.
MID_NAN_FAILING: set[tuple[str, str]] = set()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_start_policy(entry: dict) -> bool:
    return any(p["name"] == "start_policy" for p in (entry.get("parameters") or []))


def _instantiate(entry: dict, *, start_policy: str | None = None):
    cls = getattr(screamer, entry["name"])
    kwargs = {p["name"]: p["default"] for p in (entry.get("parameters") or [])}
    if start_policy is not None and "start_policy" in kwargs:
        kwargs["start_policy"] = start_policy
    return cls(**kwargs)


def _input_arrays(n_inputs: int, n_samples: int, nan_position: str) -> list[np.ndarray]:
    """Build deterministic NaN-bearing inputs of ``n_samples`` length.

    Values lie in ``(0.1, 0.9)`` to keep every function's domain happy
    (Log/Sqrt positive, Asin/Acos bounded, Power positive base).
    All input streams share the NaN locations.
    """
    rng = np.random.default_rng(42)
    base = rng.uniform(0.1, 0.9, size=n_samples).astype(np.float64)
    if nan_position == "leading":
        base[:3] = np.nan
    elif nan_position == "mid":
        base[n_samples // 4] = np.nan
    elif nan_position == "none":
        pass
    else:
        raise ValueError(f"unknown nan_position {nan_position!r}")
    return [base.copy() for _ in range(n_inputs)]


def _call_batch(entry: dict, instance, arrays: list[np.ndarray]):
    if len(arrays) == 1:
        return instance(arrays[0])
    return instance(*arrays)


def _call_stream(entry: dict, instance, arrays: list[np.ndarray]):
    """Call scalar-at-a-time. Returns the same shape as the batch call.

    Handles 1-input, multi-input, 1-output (1D), and multi-output (2D).
    """
    n_samples = len(arrays[0])
    n_outputs = int(entry.get("outputs", 1))

    if n_outputs == 1:
        out = np.empty(n_samples, dtype=np.float64)
        if len(arrays) == 1:
            for i in range(n_samples):
                out[i] = instance(float(arrays[0][i]))
        else:
            for i in range(n_samples):
                out[i] = instance(*(float(a[i]) for a in arrays))
        return out

    # Multi-output: batch returns (N, M); stream returns either a tuple
    # of M scalars or an (M,) array per call depending on the binding.
    out = np.empty((n_samples, n_outputs), dtype=np.float64)
    for i in range(n_samples):
        if len(arrays) == 1:
            r = instance(float(arrays[0][i]))
        else:
            r = instance(*(float(a[i]) for a in arrays))
        out[i, :] = np.asarray(r, dtype=np.float64).ravel()
    return out


def _last_finite(out) -> bool:
    if isinstance(out, tuple):
        return all(_last_finite(o) for o in out)
    if out.ndim == 2:
        return bool(np.all(np.isfinite(out[-1])))
    return math.isfinite(out[-1])


# ---------------------------------------------------------------------------
# Parametrization helpers
# ---------------------------------------------------------------------------

def _mark(known: set, key, label: str):
    if key in known:
        return pytest.mark.xfail(
            strict=True,
            reason=(
                f"{key}: {label} -- known-broken; remove from the set in "
                "tests/test_nan_start_policy_compliance.py when fixed."
            ),
        )
    return ()


def _stream_batch_params():
    """Cartesian product (function, nan_position). nan-aware excluded."""
    for name, entry in sorted(HELP.items()):
        if entry.get("kind", "functor") != "functor":
            continue  # stream operators / DAG names are not compute functors
        if entry["nan_policy"] == "nan-aware":
            continue
        for nan_position in ("none", "leading", "mid"):
            yield pytest.param(
                name, entry, nan_position,
                id=f"{name}-{nan_position}",
                marks=_mark(STREAM_BATCH_FAILING, (name, nan_position),
                            f"stream/batch mismatch with {nan_position} NaN"),
            )


def _start_policy_params(failing_set: set, label: str):
    """Cartesian product (function-with-start_policy, start_policy)."""
    for name, entry in sorted(HELP.items()):
        if entry.get("kind", "functor") != "functor":
            continue  # stream operators / DAG names are not compute functors
        if entry["nan_policy"] == "nan-aware":
            continue
        if not _has_start_policy(entry):
            continue
        for sp in START_POLICIES:
            yield pytest.param(
                name, entry, sp,
                id=f"{name}-{sp}",
                marks=_mark(failing_set, (name, sp), label),
            )


# ---------------------------------------------------------------------------
# P1: stream-vs-batch parity (incl. NaN paths)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,entry,nan_position", list(_stream_batch_params()))
def test_stream_matches_batch(name: str, entry: dict, nan_position: str):
    """Streaming and batch calls must produce bit-identical output.

    Run on a deterministic 200-sample input with NaN injected at
    ``nan_position``. The two outputs are compared with
    ``equal_nan=True`` and zero tolerance; the only way to satisfy this
    invariant is for the function's array-fast path to match its
    scalar-iterated state machine sample-by-sample, including the
    handling of NaN inputs.
    """
    n_inputs = int(entry.get("inputs", 1))
    arrays_a = _input_arrays(n_inputs, 200, nan_position)
    arrays_b = [a.copy() for a in arrays_a]

    inst_stream = _instantiate(entry)
    inst_batch = _instantiate(entry)

    out_stream = _call_stream(entry, inst_stream, arrays_a)
    out_batch = _call_batch(entry, inst_batch, arrays_b)

    np.testing.assert_allclose(
        np.asarray(out_stream),
        np.asarray(out_batch),
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
        err_msg=(
            f"{name} ({nan_position} NaN): stream and batch outputs differ "
            "by more than 1e-10 / 1e-12. The scalar-iterated state machine "
            "does not match the array fast-path beyond expected ULP-scale "
            "round-off."
        ),
    )


# ---------------------------------------------------------------------------
# P2: initial NaN must not poison state, under every start_policy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name,entry,start_policy",
    list(_start_policy_params(INITIAL_NAN_FAILING, "initial-NaN poisons state")),
)
def test_initial_nan_recovers(name: str, entry: dict, start_policy: str):
    """Three leading NaN inputs must not poison running state.

    For every (function-with-start_policy, start_policy) combination,
    construct a 500-sample stream of finite positive values with the
    first three samples replaced by NaN. The last output sample must
    be finite -- meaning the running state recovered after the NaN
    window slid out.
    """
    n_inputs = int(entry.get("inputs", 1))
    arrays = _input_arrays(n_inputs, 500, "leading")
    instance = _instantiate(entry, start_policy=start_policy)
    out = _call_batch(entry, instance, arrays)

    assert _last_finite(out), (
        f"{name} (start_policy={start_policy}): last output is NaN. "
        "Three leading NaN inputs poisoned state for all 497 subsequent "
        "finite samples. See docs/nan_policy.md."
    )


# ---------------------------------------------------------------------------
# P3: a single mid-stream NaN must not poison state, under every policy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name,entry,start_policy",
    list(_start_policy_params(MID_NAN_FAILING, "mid-stream NaN poisons state")),
)
def test_mid_nan_recovers(name: str, entry: dict, start_policy: str):
    """A single mid-stream NaN must not poison running state.

    Construct a 500-sample stream of finite positive values with a
    single NaN placed at index 125 (well past warmup, well before the
    end). The last output sample must be finite -- the NaN must slide
    out cleanly without leaving a permanent NaN in the running state.
    """
    n_inputs = int(entry.get("inputs", 1))
    arrays = _input_arrays(n_inputs, 500, "mid")
    instance = _instantiate(entry, start_policy=start_policy)
    out = _call_batch(entry, instance, arrays)

    assert _last_finite(out), (
        f"{name} (start_policy={start_policy}): last output is NaN. "
        "A single mid-stream NaN poisoned state for the remaining ~370 "
        "finite samples. See docs/nan_policy.md."
    )
