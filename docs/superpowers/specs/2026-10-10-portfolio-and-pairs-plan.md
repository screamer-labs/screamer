# Portfolio trading and pairs trading: plan

Status: proposed, awaiting approval.

Scope (user, 2026-10-10): pairs trading and generic portfolio trading and
rebalancing are in scope. Machine-learning tooling (cross-validation, labeling)
is out of scope for now.

## Current state

- Every target engine backtests each column of a `(T, K)` input on its own,
  with its own capital. `PortfolioReport` only sums the per-asset results.
- The pairs recipe builds the legs by hand: it converts a spread position into
  contract counts per leg with Python arithmetic, and runs the two legs as
  independent backtests.
- Hedge ratios: `RollingBeta`, `EwBeta` (OLS slope), `BayesianRegression`
  (forgetting-factor regression with intercept). `KalmanFilter` is a scalar
  random-walk level filter, not a regression.
- Spread analysis: `RollingSpread`, `RollingAlpha`, `RollingZscore`,
  `RollingOU` (mean-reversion rate, half-life), `RollingHurst`.
- Signal state: `SchmittTrigger` (0/1 hysteresis), `Hold`.

## Gaps

1. A portfolio engine: one capital account shared by `K` assets, weights of
   total equity, rebalancing rules, gross/net exposure limits, per-asset costs
   and contract specifications.
2. Cross-sectional operators: transforms across the `K` assets at each step
   (rank, z-score, demean, scale to a gross exposure), to turn scores into
   portfolio weights.
3. Pairs-specific tools: a Kalman-filter regression hedge ratio, a symmetric
   (total least squares) hedge ratio, a rolling cointegration test, a
   three-state entry/exit band signal, and a spread-to-legs conversion.
4. Validation for portfolios: `position_test` and `strategy_test` on a
   portfolio (one result for the whole book, not one per column).

## 1. `BacktestPortfolioTarget` (C++ engine)

Inputs: `target (T, K)` and market data per asset. Two market forms, like the
single-asset engines: prices `(T, K)`, or OHLC as four `(T, K)` arrays with
`fill_at` and `delay`.

- `size`: `"units"`, `"notional"`, or `"weight"`. With `"weight"`, each
  asset's units are `w_k * portfolio_equity / (price_k * multiplier_k)`, where
  portfolio equity is `initial_capital` plus the summed pnl of all assets,
  marked at the execution prices.
- Rebalancing:
  - A row of NaN keeps the current targets (working-target semantics), so a
    caller rebalances on a schedule by passing targets only on rebalance
    steps.
  - `rebalance_band`: trade an asset only when its weight drifts by more than
    this from the target; it shares the definition the single-asset `Sizer`
    uses.
  - `rebalance="all"|"drifted"`: when the band trips, rebalance every asset or
    only those outside the band.
- Exposure limits: `max_gross` and `max_net` (as multiples of equity). Targets
  that exceed them are scaled down proportionally before trading.
- Costs and contracts: `fee`, `slippage`, `multiplier`, `tick_size` accept a
  scalar or one value per asset.
- Output: `(T, K, 4)` per asset, `[equity, pnl, position, cost]`, the same as
  today's vectorized engines. `PortfolioReport` and a new
  `portfolio_report(values, periods_per_year)` helper give the book-level
  statistics, including the `BacktestStats` set.

Implementation: `K` instances of the existing fill cores (price and
OHLC), plus a shared equity accumulator and a portfolio-level `Sizer`. No new
fill logic.

## 2. Cross-sectional operators (C++, `functions_cross`)

Each takes `(T, K)` and transforms every row across its `K` columns, ignoring
NaN entries (assets not yet listed or halted):

- `CrossRank` (rank scaled to [-1, 1] or [0, 1]), `CrossZscore`,
  `CrossDemean`, `CrossWinsorize`.
- `CrossNormalize(gross=1.0, net=None)`: scale a row to a gross exposure,
  optionally demeaned to a net target (dollar-neutral at net = 0).
- `CrossTopBottom(n)`: +1/-1 for the top and bottom `n` scores, 0 otherwise.

These are causal (one row at a time) and stream the same way as every other
operator.

## 3. Pairs trading

- `KalmanRegression`: time-varying intercept and hedge ratio with a random-walk
  state (the standard dynamic hedge-ratio filter); outputs alpha, beta, the
  innovation (spread), and its predicted standard deviation (a ready z-score
  denominator).
- `RollingTLSBeta`: total least squares hedge ratio, symmetric in the two legs.
- `RollingADF` (Engle-Granger on a spread): the ADF statistic over a rolling
  window, with MacKinnon critical values, to gate trading on cointegration.
- `BandSignal(entry, exit, stop=None)`: three-state position from a z-score:
  short above `entry`, long below `-entry`, flat when it crosses back inside
  `exit`, and flat (with a lockout until it returns inside `exit`) beyond
  `stop`.
- Spread to legs: `BacktestPortfolioTarget` with `size="notional"` takes the
  legs directly; a small helper `spread_legs(units, beta, prices, notional)`
  or a `SpreadLegs` operator returns the `(T, 2)` leg targets.
- Rewrite the pairs recipe and notebook on these, with costs, hysteresis, and
  a significance test.

## 4. Validation on portfolios

- `strategy_test` accepts a strategy that returns `(T, K, 4)` and scores the
  summed book.
- `position_test` gains `portfolio=True` for a `(T, K)` target: all columns
  are shifted together and scored as one book (the family mode stays the
  default for K variants).
- `null_stats` and `observed_stats` work unchanged on the book.

## Order of work

1. Cross-sectional operators (small, unblock weights).
2. `BacktestPortfolioTarget` (price form, then OHLC form) and
   `portfolio_report`.
3. Pairs operators: `BandSignal`, `KalmanRegression`, `RollingTLSBeta`,
   `RollingADF`, spread-to-legs.
4. Portfolio validation.
5. Docs: a "portfolio backtest" guide, rewritten pairs recipe and notebook, a
   cross-sectional momentum recipe, and engine-chooser updates.

All with Python and JS/WASM parity, C++ logic, and the usual meta-test
registrations.
