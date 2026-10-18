# Robust (Huber) Kalman update design

Date: 2026-10-18. Status: ruling made while the user is away, prompted by the user sharing a note on robust
Kalman filtering for pairs (fat-tailed innovations let one outlier jerk the hedge ratio); for review before merge.

## Goal

`KalmanFilter` and `KalmanRegression` assume Gaussian observation noise, so an innovation's pull on the state
grows with its size: a bad print, a gap or an event moves the estimate fully and the distortion carries forward.
Add one optional parameter that caps an outlier's influence on the state, while the outputs that report the
innovation keep showing it in full, so a break test downstream still sees it.

## Semantics

New parameter on both operators: `huber` (float, default `inf` = off, must be > 0). At each step, with
innovation e and its predicted variance Q (Q = P_pred + R for KalmanFilter; Q = h R h' + obs_var for
KalmanRegression), let z = e / sqrt(Q) and w = min(1, huber / |z|). The update uses the observation variance
R / w instead of R (Huber M-estimate step of the Masreliez-Martin robust filter): gain, state and covariance
updates all use it (Joseph form kept in KalmanRegression). With |z| <= huber nothing changes; `huber = inf`
reproduces today's numbers bit for bit.

Outputs are unchanged in count and meaning. KalmanRegression's `spread` stays the raw innovation and
`spread_std` stays sqrt(Q) of the un-inflated prediction, so `spread / spread_std` is the full standardized
innovation for diagnostics. KalmanFilter's output is the state.

Typical value: 2 to 3 (sd). Docs explain the trade-off: a robust filter protects the estimate, and a genuine
break is then absorbed more slowly; pair it with a break test on the raw standardized innovation.

## Bindings, docs, tests

C++ only logic; Python and JS bindings with identical numbers; docs pages updated (parameter, a short paragraph,
reference: Masreliez and Martin 1977; Huber 1964). Tests: huber=inf identical to the current op (bit for bit) on
random data; a single injected outlier moves the state less, by the expected weight (hand-computed one step);
small innovations unaffected; batch == streaming; Python/JS parity; parameter validation.
