# SchmittTrigger initial state

## Context

`SchmittTrigger(lower, upper)` is a hysteresis comparator: it latches high when
the input rises above `upper`, low when it falls below `lower`, and holds its
previous value inside the dead band `[lower, upper]`. Until the first sample
crosses a threshold there is no prior state to hold, so the current implementation
emits `NaN`.

That `NaN` warmup is a footgun. A signal that starts inside the dead band reads
`NaN` for every sample until its first crossing, which forces a `nan_to_num` at
the call site. This surfaced in screamer-bots' `positions()`, where the warmup
`NaN` had to be scrubbed to treat the start as flat.

The fix gives the trigger a definite starting latch so warmup resolves to a real
state instead of `NaN`.

## Goal

Add an `initial` latch seed to `SchmittTrigger` and default it to the low state,
so the common case needs no `nan_to_num`, while preserving the current `NaN`
warmup as an explicit opt-in.

## Design

### API

`SchmittTrigger(lower, upper, initial = 0.0)`

- `initial` is the value the output latches to before the first threshold
  crossing. It must be exactly `0.0` (low), `1.0` (high), or `NaN` (undefined
  until the first crossing). Any other value throws `std::invalid_argument`, in
  the same style as the existing `lower < upper` guard.
- The default `0.0` means a fresh trigger reads low until its input first rises
  above `upper`.

### Behavior

- `reset()` seeds the latch with `initial` instead of `NaN`.
- Default `initial=0.0`: a signal that starts inside the dead band reads `0.0`
  until it first crosses `upper`.
- `initial=1.0`: the mirror image, reads `1.0` until the input first falls below
  `lower`.
- `initial=NaN`: reproduces today's behavior (output undefined until the first
  crossing) for callers who want to mask warmup.
- Unchanged: the crossing logic (`> upper` latches `1.0`, `< lower` latches `0.0`,
  the dead band holds the previous value) and the NaN-input path (a `NaN` input
  returns `NaN` and leaves the latch untouched, per the "ignore" policy). The
  `nan_policy` is unchanged.

This is a breaking change to the default warmup output (`0.0` rather than `NaN`),
which is acceptable pre-1.0. A caller that relied on `NaN` to mask warmup passes
`initial=nan` to keep that behavior.

### Surface

- The pybind binding gains `py::arg("initial") = 0.0`.
- The docstring and `docs/functions_fin/SchmittTrigger.md` document the `initial`
  seed and the new low default, and note that `initial=nan` restores the
  undefined-until-crossing warmup.
- A `[Unreleased]` changelog entry records the new parameter and the default
  behavior change.

## Testing

- A signal that starts inside the dead band reads `0.0` during warmup under the
  default (was `NaN`).
- `initial=1.0` reads `1.0` during warmup until the first cross below `lower`.
- `initial=nan` reproduces the old `NaN` warmup.
- An invalid `initial` (for example `0.5`) raises `std::invalid_argument`.
- The crossing behavior (above `upper`, below `lower`, dead-band hold) is
  unchanged from the current tests.
- A `NaN` input returns `NaN` and leaves the latch untouched.
- The standard batch-equals-stream and reset checks still pass (reset restores the
  `initial` seed).

## Out of scope

Updating screamer-bots' `positions()` to drop its `nan_to_num` now that the
default warmup is a real state is a downstream follow-up in that repository, noted
here so it is not lost.
