# `RollingCov`

## Description

`RollingCov` computes the rolling sample covariance between two streams over a sliding window of fixed size. The result matches `pandas.Series.rolling(w).cov(other)` to floating-point tolerance.

*Equation*:

$$
\mathrm{cov}_w[t] = \frac{n \sum_{i} x_i y_i - \sum_{i} x_i \sum_{i} y_i}{n\,(n-1)}
$$

with the sums taken over the most recent `window_size` samples. The `n - 1` denominator gives the unbiased sample estimate.

*Parameters*:

- **`window_size`** (`int`, ≥ 2): size of the rolling window.
- **`start_policy`** (`str`, default `"strict"`): controls warmup behavior. See `RollingMean` for the full definition.

*Input shape*: takes **two** parallel streams. See [`RollingCorr`](RollingCorr.md) for the full input/output table; the call signature is identical.

*Return value*: a real number. Returns `NaN` during warmup. Returns `0` when one of the streams is constant within the window (covariance with a constant series is zero, well-defined).

## Usage Example and Plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import RollingCov

    np.random.seed(0)
    N = 300
    eps_a = np.random.normal(size=N)
    eps_b = np.random.normal(size=N)
    rho = np.where((np.arange(N) > N / 3) & (np.arange(N) < 2 * N / 3), 0.85, 0.15)
    a = np.cumsum(eps_a)
    b = np.cumsum(rho * eps_a + np.sqrt(1 - rho ** 2) * eps_b)

    cov_30 = RollingCov(window_size=30)(np.diff(a), np.diff(b))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.5], vertical_spacing=0.05)
    fig.add_trace(go.Scatter(y=a, mode="lines", name="a"), row=1, col=1)
    fig.add_trace(go.Scatter(y=b, mode="lines", name="b"), row=1, col=1)
    fig.add_trace(go.Scatter(y=cov_30, mode="lines", name="RollingCov(30)"), row=2, col=1)
    fig.update_layout(
        title="Rolling covariance of two random walks",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="rolling covariance", row=2, col=1)
    fig.show()
```

## Implementation Details

Three `detail::RollingSum` buffers maintain `Σx`, `Σy`, `Σxy`. Each new sample updates all three sums in `O(1)`, then the covariance is computed from the closed-form expression above.

* **Time**: `O(1)` per new element.
* **Space**: `O(window_size)` (three circular buffers).
* **Reference**: matches `pd.Series(x).rolling(w).cov(pd.Series(y))` to floating-point tolerance, verified in `tests/test_rolling_two_input.py`.
