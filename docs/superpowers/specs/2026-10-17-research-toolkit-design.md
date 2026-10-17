# Research toolkit design

Date: 2026-10-17. Status: design approved in conversation; awaiting spec review.

## Goal

Close the gaps the first screamer-bots research loop ran into (117
hypotheses, see `screamer-bots/research/SYNTHESIS.md` section 6) and add
probabilistic regime models, so that the second loop needs no hand-built
bookkeeping. Every addition extends an existing screamer concept and copies
its naming, parameters and output conventions; nothing introduces a new way
of doing a thing screamer already does.

## Decisions

- Regime models are self-contained rolling-refit operators (like
  `RollingOU`, `RollingHurst`), not a fit/predict pair.
- Funding enters validation through `walk_forward(carry=)`, which applies
  the existing `BacktestCarry`; no engine gains a funding input.
- Engine output columns do not change. The position column stays in units.
- Bad prices keep raising in `permute_bars`; only the message improves.
- All logic is C++. Python wrappers only label, check and dispatch. Every
  operator ships in Python and JS/WASM (parity rule), in eager, graph and
  lazy use.
- One minor release at the end, by `make minor`, with the user's approval.

## A. Backtest and validation

### A1. `backtest_trades(values, index=None)`

A sibling of `backtest_report`. `values` is the `(T, 4)` engine output
`[equity, pnl, position, cost]`. Returns a dict of equal-length numpy
arrays, one row per round trip:

| key | meaning |
|---|---|
| `entry` | row of the trade's first nonzero position (index value when `index` is given) |
| `exit` | row where the position returns to zero or flips sign (index value when given); the last row for an open trade |
| `side` | +1 long, -1 short |
| `bars` | `exit - entry` in rows |
| `pnl` | net pnl over the trade's rows (pnl column, which is after cost) |
| `cost` | cost over the trade's rows |
| `closed` | False for a trade still open on the last row |

A trade is defined exactly as `BacktestStats` defines a round trip: flat to
flat, and a sign flip closes one trade and opens the next. The C++ reuses
that state machine, so on the same input the closed trades' count, win
rate, average win and loss, and average holding equal `BacktestStats`'s
values. Portfolio output `(T, K, 4)` raises; call it per asset.

### A2. `backtest_report(values, index=None, periods_per_year=1.0, by=None)`

`by` is one label per row (any hashable numpy dtype: datetime64[M] months,
years as ints, strings), named after pandas `groupby(by=)`. When given, the
summary dict gains `by_group`: a dict from label (in order of first
appearance) to a dict with `total_pnl`, `total_cost`, `turnover`, `sharpe`
(annualized with `periods_per_year`), `max_drawdown` (of cumulative pnl
over that group's rows, in order) and `steps`. Trade statistics are not
grouped (a trade can span groups); group `backtest_trades` rows by their
entry instead. The per-group pass is one C++ call on integer group codes.
Without `by`, the output is unchanged.

### A3. `walk_forward(..., carry=None, carry_multiplier=1.0)`

`carry` is a per-step rate series, `(T,)` for a single asset or `(T, K)`
for a portfolio, with the sign and timing `BacktestCarry` documents. When
given, every candidate's engine output, in training and in testing, passes
through `BacktestCarry(multiplier=carry_multiplier)` with the last price
input given to `walk_forward` (the close for OHLC engines) before scoring.
`output` and `stats` in the result are after carry. Default `None` leaves
results unchanged.

### A4. `walk_forward` for portfolio engines

Portfolio engines (`BacktestPortfolioTarget`, `BacktestPortfolioOHLCTarget`)
are accepted. Targets are `(T, K, C)`: K assets, C candidates, the last axis
holding candidates as it does for single assets. Prices are `(T, K)`. The
score is computed on the summed pnl across assets. `output` is `(T, K, 4)`.
Single-asset behaviour is unchanged.

### A5. `permute_bars` / `permute_returns` error message

The `mode="log"` error names the first offending row, its value, and the
number of non-positive values, and still suggests `mode="diff"`.

## B. Operators

All follow the `Rolling*` contract: `window_size`, `start_policy`
(`strict|expanding|zero` where meaningful), declared `nan_policy`, a NaN
input gives NaN output and leaves state unchanged unless the page says
otherwise.

| Operator | In -> out | Definition |
|---|---|---|
| `Hold(n, release=0.0, cooldown=0)` | 1 -> 1 | Existing behaviour, plus: after a hold expires, nonzero inputs in the next `cooldown` steps are ignored (output `release`). A trigger during a hold still restarts it. `cooldown=0` is today's behaviour exactly. |
| `RollingUpsideDeviation(window_size=20, mar=0.0, start_policy="strict")` | 1 -> 1 | Mirror of `RollingDownsideDeviation`: RMS of `max(r - mar, 0)` over the window. |
| `RollingSeasonalMean(period=24, window_size=20, start_policy="strict")` | 1 -> 1 | Mean of `x_t, x_{t-period}, ..., x_{t-(window_size-1) period}`: the same slot of the last `window_size` cycles. Slots are counted in steps, so the input must be on a regular clock. |
| `RollingSeasonalStd(period=24, window_size=20, start_policy="strict")` | 1 -> 1 | Sample standard deviation of the same values. |
| `RollingSeasonalZscore(period=24, window_size=20, start_policy="strict")` | 1 -> 1 | `(x_t - mean) / std` over the same slot values, with the same inclusion of `x_t` as `RollingZscore`. |
| `RollingBipowerVar(window_size=20, start_policy="strict")` | 1 -> 1 | Input returns. `(pi / 2) * sum |r_i| |r_{i-1}|` over the window's adjacent pairs, scaled by `window_size / (window_size - 1)` so it estimates the same quantity as the realized variance `sum r_i^2`. |
| `RollingJumpShare(window_size=20, start_policy="strict")` | 1 -> 1 | `max(0, 1 - BV / RV)`: the share of realized variance from jumps (Barndorff-Nielsen and Shephard). 0 when RV is 0. |
| `RollingVarianceRatio(window_size=256, lag=2, output="ratio", start_policy="strict")` | 1 -> 1 | Input returns. Lo-MacKinlay variance ratio of overlapping `lag`-step sums over the window, with their bias correction. `output="zscore"` gives the heteroskedasticity-robust statistic z*(q). `lag >= 2`, `window_size > 2 lag`. |
| `RollingPermutationEntropy(window_size=100, order=3, delay=1, start_policy="strict")` | 1 -> 1 | Bandt-Pompe permutation entropy of the ordinal patterns of `order` values spaced `delay` apart, over the patterns ending in the window, normalized by `log(order!)` to [0, 1]. Ties rank by order of appearance. `2 <= order <= 7`. |

## C. Probabilistic regime models

### C1. `RollingHMM(window_size=720, refit_every=24, emission="gaussian", max_iter=50, tol=1e-6)`, 1 -> 3

A two-state hidden Markov model of the input (normally returns), refit on a
trailing window and filtered forward one step at a time.

- `emission="gaussian"`: state k emits `N(mu_k, sigma_k^2)`.
  `emission="ar1"`: state k emits `x_t = c_k + phi_k x_{t-1} + sigma_k e_t`
  (Hamilton's Markov-switching AR(1)).
- Fit: Baum-Welch EM on the last `window_size` values, every `refit_every`
  steps, warm-started from the previous fit; the first fit starts from a
  deterministic split (window values below and above the median absolute
  value for `gaussian`; phi from the window's lag-1 autocorrelation plus
  and minus a fixed offset for `ar1`). Stops at `max_iter` or when the
  log-likelihood gain is below `tol`.
- Identification: after each fit the states are ordered so that state 1 is
  the high-variance state (`gaussian`) or the high-phi, more persistent
  state (`ar1`).
- Filter: between refits, the forward recursion runs with fixed parameters
  on each new value; at a refit the filtered probability is recomputed
  over the window with the new parameters. Every output at t uses values
  up to t only.
- Outputs: `prob` (filtered probability of state 1), `pred_mean` and
  `pred_std` (mean and standard deviation of the one-step-ahead predictive
  mixture), as `BayesianRegression` names them.
- NaN for t before the first fit (`window_size` valid values). NaN input:
  NaN output, state unchanged.

### C2. `BayesianChangepoint(expected_run=250, max_run=1000, prior_var=1.0)`, 1 -> 3

Adams-MacKay online Bayesian changepoint detection with a constant hazard
`1 / expected_run` and a normal-gamma prior (mean 0, kappa 1, alpha 1,
beta `prior_var`). The run-length distribution is truncated at `max_run`.
Outputs: `run_length` (posterior expected run length), `pred_mean` and
`pred_std` (posterior predictive Student-t mean and standard deviation of
the next value). The docs say to feed a standardized series (for example a
`RollingZscore` or returns divided by `EwStd`). NaN input: NaN output,
state unchanged.

### C3. `CusumFilter(threshold, drift=0.0)`, 1 -> 1

The symmetric CUSUM event filter (Lopez de Prado, AFML 2.5.2.1). With
`S+ = max(0, S+ + x - drift)` and `S- = min(0, S- + x + drift)`, the output
is +1 when `S+ > threshold`, -1 when `S- < -threshold` (both sums reset to
0), else 0. The input is normally returns or price differences.
`threshold > 0`, `drift >= 0`. NaN input: NaN output, state unchanged.

## Documentation

Each operator gets a `docs/functions_<family>/<Name>.md` page with the
standard frontmatter (`output_names` for multi-output ops), a usage plot,
`see_also` links, and topic tags (`regime` for C1-C3 and
`RollingVarianceRatio`, `RollingPermutationEntropy`, `RollingJumpShare`).
`backtest_trades`, `by=`, and `carry=` / portfolio `walk_forward` are
documented in the backtesting and validation guides. `choosing_an_operator.md`
gains a regime section. CHANGELOG under Unreleased names the Python and JS
spellings. Prose follows the writing guide (no em dashes).

## Tests

- Per operator: hand-computed small cases; a reference implementation in
  `devtools/baselines/` (numpy or scipy or a direct transcription of the
  paper's formulas) compared on random input; batch == streaming == graph;
  NaN compliance; `param_cases.py` entries; JS tests via the generated
  bindings.
- `Hold`: `cooldown=0` reproduces every existing test; cooldown cases.
- `RollingHMM`: on a simulated two-state series with known parameters the
  fitted parameters and the filtered state agree with the truth within set
  tolerances; with `ar1`, a series switching between phi 0.6 and -0.4 is
  identified; outputs never use future values (truncation test: output at t
  is unchanged when values after t change).
- `BayesianChangepoint`: run length drops within a few steps of a mean
  shift of 3 standard deviations in simulated data; equals a direct numpy
  transcription of the recursion on random input.
- `backtest_trades`: agrees with `BacktestStats` on every engine's output in
  the existing backtest fixtures.
- `by=`: group totals add up to the ungrouped totals; a single label equals
  the ungrouped summary.
- `walk_forward`: `carry` of zeros equals no carry; portfolio with K=1
  equals the single-asset result; existing tests unchanged.
- Full suite, `make tidy`, the compat probe diff (no changed numbers),
  and the WASM codegen freshness test.

## Out of scope

Calendar-aware resampling, fit/predict model objects, more than two HMM
states, position-in-target-units engine columns, data cleaning operators.
