# BayesianRegression: online Bayesian simple regression with uncertainty

## Context

screamer is a library of causal streaming operators, and today almost all of them
are *point* estimators: `RollingMean`, `EwMean`, `RollingLinearRegression`, and the
whole `Ew*` family emit a number, not a distribution. The one recursive-Bayesian
operator, `KalmanFilter`, tracks a posterior variance internally but throws it away
and emits only the point estimate.

The relational operators are all pairwise. `RollingCorr`, `EwCorr`, `RollingCov`,
`EwCov`, `RollingBeta`, and `EwBeta` are each `FunctorBase<_, 2, 1>`: two input
streams, one output. There is no N-series joint estimator anywhere in the library,
and `RollingLinearRegression` (`FunctorBase<_, 2, 4>`) is univariate: it regresses
`y` on a single predictor `x` plus an intercept and emits slope, intercept, r2, and
the slope standard error over a hard window.

This spec adds the first member of a Bayesian streaming layer: an online Bayesian
simple regression that carries a full posterior and emits a calibrated one-step-ahead
forecast with its uncertainty. It is the uncertainty-carrying sibling of
`RollingLinearRegression`, and it slots into the existing 2-input regression and beta
family.

The broader Bayesian streaming layer (conjugate online estimators for probabilities
and rates, Bayesian online changepoint detection, exposing the Kalman posterior, and
a future multivariate regression that would be screamer's first N-series joint
operator) is out of scope here; each is its own spec.

## Goal

Add `BayesianRegression`, an online Bayesian univariate linear regression that
updates a conjugate posterior recursively with an exponential forgetting factor and
emits, causally at every step, a one-step-ahead predictive mean and standard
deviation together with the current learned slope and intercept.

## What it adds over what exists

- **Uncertainty, online.** A calibrated predictive interval for the next `y`, the
  thing the point-estimator library lacks. `EwBeta` gives the slope and
  `RollingLinearRegression` gives a slope standard error, but neither gives a
  predictive distribution for a new observation.
- **Defined from the first sample.** The prior regularizes, so the estimate and its
  interval are sensible immediately, with no NaN warmup (`RollingLinearRegression`)
  and no early blow-up (`EwBeta` on thin data).
- **Self-tuning noise.** The observation noise `sigma^2` is learned from residuals
  rather than fixed.
- **Drift-tracking.** The exponential forgetting factor discounts old data so the fit
  follows a changing relationship, the same reason the `Ew*` family exists.

## The model

Univariate linear model with unknown noise variance:

    y = beta0 + beta1 * x + eps,    eps ~ Normal(0, sigma^2)

Write the design vector `phi = [1, x]` (intercept, slope). The conjugate prior is
Normal-Inverse-Gamma: `beta | sigma^2 ~ Normal(m0, sigma^2 * V0)` and
`sigma^2 ~ InverseGamma(a0, b0)`. Conjugacy makes the posterior after each
observation another Normal-Inverse-Gamma, so the update is exact and recursive.

### Recursive update with forgetting

The operator maintains the natural-parameter (precision) form of the posterior:
a 2x2 precision matrix `Lambda`, a 2-vector `theta = Lambda * m` (where `m` is the
posterior coefficient mean), and the two inverse-gamma accumulators for `sigma^2`
(a shape term that grows with the effective sample count, and a scale term that
tracks the weighted residual sum of squares). Per observation `(phi, y)`, with an
exponential forgetting factor `lam` in `(0, 1]`, the data-driven statistics are
discounted by `lam` and the new observation is folded in:

    Lambda <- lam * Lambda_data + phi phi^T   (+ the un-forgotten prior precision)
    theta  <- lam * theta_data  + phi * y      (+ the un-forgotten prior term)
    and the inverse-gamma accumulators are discounted by `lam` and updated with `y`.

The prior contribution is not forgotten (it is the floor the posterior relaxes to as
evidence is discounted). The exact, numerically stable form of the update (keeping
the prior separate from the forgotten data statistics, and the 2x2 inverse) is
settled in the implementation plan; the model above fixes what it must compute.

Everything is 2x2, so the update is O(1) per step.

### Causal one-step-ahead predictive

At step `t`, before folding in `(x_t, y_t)`, the operator forms the predictive
distribution for `y_t` from `x_t` and the posterior built from steps before `t`. The
posterior-predictive is Student-t:

    dof       nu    = 2 * a
    location        = phi_t^T m
    scale^2         = (b / a) * (1 + phi_t^T Lambda^{-1} phi_t)
    predictive var  = scale^2 * nu / (nu - 2)      for nu > 2

With a proper prior (`a0 > 1`), `nu > 2` from the first step, so the predictive
standard deviation is finite from the start. The predictive is fat-tailed and wide
when evidence is thin and tightens toward Gaussian as data accumulates.

## Interface

`BayesianRegression`, called `(y, x)` to match `RollingLinearRegression`'s argument
order. `FunctorBase<BayesianRegression, 2, 4>`.

Outputs, four columns, in order:

    [pred_mean, pred_std, slope, intercept]

- `pred_mean`, `pred_std`: the one-step-ahead predictive mean and standard deviation
  for `y_t` given `x_t`, using only data before `t` (the causal forecast with its
  interval).
- `slope`, `intercept`: the posterior coefficient means after folding in step `t`
  (the current learned model). Reporting the post-update model at `t` is causal: it
  uses `y_t`, which is known at `t`; only the forecast columns must exclude `y_t`.

### Parameters

- **Forgetting**, parameterized exactly like the `Ew*` family: exactly one of `com`,
  `span`, `halflife`, or `alpha`, mapped to the per-step discount the same way
  `EwMean` maps them (so this reads as an exponentially-forgetting regression). Pure
  no-forgetting (`alpha = 0`) is out of scope for this operator, matching the `Ew*`
  convention that rejects `alpha` outside `(0, 1)`.
- **Prior**: a weak, proper Normal-Inverse-Gamma prior with a small number of tunable
  knobs, a prior coefficient mean (default zero for both intercept and slope), a
  prior strength (precision) controlling shrinkage, and a prior noise scale for
  `sigma^2`. Defaults are chosen in the plan so that the predictive is defined from
  step 1 (`a0 > 1`) and the prior is weak enough to be uninformative after modest
  data.

## Contract

- `FunctorBase<BayesianRegression, 2, 4>`, O(1) per step (2x2 linear algebra).
- Causal: the forecast columns use only data before the current step; the update
  then incorporates the current observation. No future data touches any output.
- Batch equals stream on the emitted rows.
- `nan_policy: ignore`: a NaN `x` or `y` emits an all-NaN row and leaves the posterior
  untouched.
- `reset()` restores the prior (the initial posterior), so a fresh session starts
  from the prior.

## Validation

- **Batch equals stream** and **reset** restores the prior, the standard operator
  invariants.
- **Recovery**: on synthetic `y = a + b x + noise` data, `slope` and `intercept`
  converge to the true `(a, b)` and the learned noise scale converges to the true
  `sigma`.
- **Calibration**: on Gaussian data, roughly 68 percent of realized `y_t` fall within
  `pred_mean +/- pred_std`, and roughly 95 percent within two standard deviations, so
  the intervals are honest.
- **Drift tracking**: with a finite forgetting factor, a deliberate mid-series change
  in the true slope is followed; with a longer memory it lags more, confirming the
  forgetting knob works.
- **Prior regularization**: a thin or degenerate sample (for example a near-constant
  `x`) yields a shrunk, finite estimate rather than a blow-up, and outputs are defined
  from the first step (no NaN warmup).
- **Parity limit**: with no effective forgetting (a very long memory), a weak prior,
  and many samples, `slope` and `intercept` match `RollingLinearRegression` on the
  same data to a set tolerance.
- **nan_policy**: a NaN input emits a NaN row and leaves the next non-NaN step's
  posterior identical to having skipped the NaN.

## Out of scope (future, separate specs)

- Multivariate Bayesian regression (k predictors), which requires a vector-valued
  input or dynamic-arity mechanism screamer does not have today and would be its first
  N-series joint operator.
- The other Bayesian streaming families: conjugate online estimators (Beta-Bernoulli
  probability, Gamma-Poisson rate), Bayesian online changepoint detection, and
  exposing the `KalmanFilter` posterior variance.
