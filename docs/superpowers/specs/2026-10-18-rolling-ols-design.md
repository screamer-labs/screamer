# Multi-regressor rolling regression design

Date: 2026-10-18. Status: ruling made while the user is away; a gap the screamer-bots research loop hit twice
(HAR volatility forecasts, H216 / H223, had to fall back to numpy normal equations). For review before merge.

## Goal

A causal linear regression of one series on k regressors over a trailing window or the whole history so far,
with the one-step-ahead prediction made before the step's own update, so it can serve as an out-of-sample
forecast (HAR-RV, factor betas, multi-leg hedge ratios).

## Shape

`RollingMultiRegression(window, n_regressors, intercept=True)`; inputs: y then x_1..x_k (k = n_regressors, 1 <= k <= 16);
`window` >= k + 1 + (1 if intercept) is a trailing count of rows, or 0 for an expanding fit over every row so
far (follow the library's existing convention for expanding windows if one exists, e.g. a separate parameter
or `window=None`; pick the established one and say so). Outputs (output_names order): `pred` (the prediction
of y_t from x_t with the coefficients fitted on rows before t: NaN until the fit is identified), `resid`
(y_t - pred), `r2` (in-sample R^2 of the fit after the update), then `b0` (intercept, if any) and `b1..bk`
(coefficients after the update). Rows with any NaN input are skipped (do not enter the window) and give NaN
outputs.

Numerics: maintain sums of cross-products for the window (add the new row, remove the expiring one) and solve
the (k+1)x(k+1) normal equations by Cholesky with a small relative ridge only to guard singularity (document it;
default 0, a `ridge` parameter if needed) or, preferably, an updating/downdating QR or a periodic exact refit
every `window` steps to bound drift; the result must match numpy lstsq on the same window to 1e-9 relative on
well-conditioned data. A singular window gives NaN coefficients, not an exception.

## Bindings, docs, tests

C++ logic only; Python and JS bindings with identical numbers; docs page (HAR-RV example: y = log RV_30
forward, x = log RV_1, RV_7, RV_30), see_also RollingLinearRegression, RollingBeta, BayesianRegression,
KalmanRegression. Tests: equals numpy lstsq per window (random data, k = 1..4, with and without intercept,
rolling and expanding), pred uses only past rows (perturbing y_t does not change pred_t), NaN rows skipped,
singular window gives NaN, batch == streaming, Python/JS parity, k = 1 agrees with RollingLinearRegression.

## Follow-up: names and the exponentially weighted variant

The operators are named `RollingMultiRegression` and `ExpandingMultiRegression` (renamed from
their working names) and live in `include/screamer/multi_regression.h` with a third,
`EwMultiRegression(n_regressors, intercept=True, com=None, span=None, halflife=None, alpha=None)`.
It is the weighted least-squares fit over every row so far with weight `(1 - alpha)^i` on the row
`i` rows back (pandas `adjust=True`, as `EwBeta`), with the same inputs, outputs, warmup, NaN-row
skipping and singular handling. It shares the fit of the other two: before each row the total weight
and the centered cross-products are scaled by `1 - alpha`, then the row enters by West's weighted
update; no row ever leaves, so there is no exact recomputation. `n_regressors` comes first because
it is required and the smoothing settings are keyword options (exactly one of them).
