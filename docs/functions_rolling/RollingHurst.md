# `RollingHurst`

## Description

Rolling-window estimator of the **Hurst exponent** $H$, a measure of long-range
dependence in a time series:

- $H \approx 0.5$ — uncorrelated (e.g. Brownian / white noise / GBM log-returns).
- $H > 0.5$ — persistent (trends tend to continue).
- $H < 0.5$ — anti-persistent (reversion).

Estimation is via **rescaled-range (R/S) analysis** with the Anis-Lloyd
small-sample correction. Within the trailing window of `window_size` samples,
the series is partitioned into non-overlapping blocks of length
$n_k = \texttt{min\_scale}, 2\cdot\texttt{min\_scale}, \dots, W/2$ (dyadic
scales). For each scale we compute the average rescaled range across blocks,
adjust for the small-sample bias, and regress on $\log n$:

$$
\begin{aligned}
\text{R/S}(n) &= \frac{1}{\lfloor W/n \rfloor}
  \sum_{p=0}^{\lfloor W/n \rfloor-1}
  \frac{\max_j Y_{p,j} - \min_j Y_{p,j}}{\sigma_p},
  \quad Y_{p,j} = \sum_{i=1}^{j}(x_{p,i} - \bar x_p) \\[2pt]
\text{rsal}(n) &= \text{R/S}(n) - \text{ers}(n) + \sqrt{\tfrac{\pi n}{2}} \\[2pt]
\hat H &= \text{slope of } \log \text{rsal}(n_k) \text{ on } \log n_k
\end{aligned}
$$

where $\text{ers}(n)$ is the Anis-Lloyd theoretical R/S for $H = 1/2$:

$$
\text{ers}(n) = \frac{\Gamma\!\left(\tfrac{n-1}{2}\right)}{\sqrt{\pi}\,\Gamma\!\left(\tfrac{n}{2}\right)}
\sum_{i=1}^{n-1} \sqrt{\tfrac{n-i}{i}} \qquad (n \le 340; \text{asymptotic form otherwise}).
$$

## Parameters

- `window_size` (int): trailing-window length $W$. Must satisfy
  $W \ge 4\,\texttt{min\_scale}$ so the regression has at least 3 scales.
- `min_scale` (int, default `4`): smallest block size $n_0$. Larger values
  reduce small-sample bias but cost scales for the regression.
- `method` (str, default `'rs'`): only `'rs'` is supported in v1 (Anis-Lloyd
  corrected R/S, matching method 2 of the reference paper).

*Warmup*: first valid output at sample index `window_size - 1`. Earlier samples
return `NaN`. Also returns `NaN` if any block in the window has zero variance.

## Implementation Details

* Circular buffer of size `W`.
* Dyadic scale list and the `ers(n_k)` / $\sqrt{\pi n_k / 2}$ / $\log n_k$
  tables are precomputed in the constructor (independent of data).
* Per step: recompute average R/S at each of $\sim \log_2 W$ scales, then a
  closed-form OLS slope of $\log \text{rsal}$ on $\log n$.
* Time complexity: **O(W log W)** per step.
* Space complexity: O(W).

## Usage Example and Plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from screamer import RollingHurst

    rng = np.random.default_rng(0)
    n = 1500

    # Stitch three regimes: white noise, trend (persistent), oscillating (anti-persistent).
    wn   = rng.normal(0, 1, n // 3)
    drift = 0.05 * np.arange(n // 3) + rng.normal(0, 1, n // 3)
    osc  = rng.normal(0, 1, n - 2 * (n // 3))
    osc[1::2] *= -1  # zig-zag → strong mean reversion
    x = np.concatenate([wn, drift, osc])

    H = RollingHurst(window_size=512)(x)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=x, mode='lines', name='signal',
                             line=dict(color='lightsteelblue'), yaxis='y1'))
    fig.add_trace(go.Scatter(y=H, mode='lines', name='RollingHurst(W=512)',
                             line=dict(color='crimson'), yaxis='y2'))
    fig.add_hline(y=0.5, line_dash="dot", line_color="gray",
                  annotation_text="H = 0.5 (uncorrelated)", yref='y2')
    fig.update_layout(
        title="Hurst exponent across regime changes",
        xaxis_title="Index",
        yaxis=dict(title="signal", side='left'),
        yaxis2=dict(title="Hurst H", side='right', overlaying='y', range=[0, 1]),
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.show()
```

`H` hovers near 0.5 in the noise regime, rises above 0.5 during the trending
segment, and drops below 0.5 in the zig-zag regime.

## Notes on usage

- For asset prices, apply `RollingHurst` to **returns** (or log-returns), not
  the price itself. Prices are typically integrated (random-walk-like) and
  would estimate $H \approx 1$ regardless of return-level memory.
- With `window_size=512` and `min_scale=4` the regression has 7 scales
  (4, 8, 16, 32, 64, 128, 256). Smaller windows are noisier; larger windows
  trade responsiveness for stability.
- `'rs'` is the classical Hurst estimator; alternative families
  (DFA, generalised-Hurst from $|\Delta x|^q$ moments) may follow in later
  releases as additional `method` options.

## Reference

Anis, A.A. and Lloyd, E.H. (1976); Peters (1994); Weron (2002);
"Estimating the Hurst exponent" (arXiv:1805.08931). Validated against the
reference Python implementation of the algorithm in `tests/test_rolling_hurst.py`.
