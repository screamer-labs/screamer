# Range-based volatility estimators

`screamer` provides three classic estimators of return volatility that exploit intraday H/L/O/C information. They are far more *statistically efficient* (lower variance for the same `n`) than simple close-to-close `RollingStd` because they use the full bar shape rather than just the close.

Each estimator ships in **four flavours**:

|                         | Variance form          | Volatility form        |
|-------------------------|------------------------|------------------------|
| **Rolling**             | `Rolling*Var(window)`  | `Rolling*Vol(window)`  |
| **Exponentially weighted** | `Ew*Var(span/com/halflife/alpha)` | `Ew*Vol(...)` |

Where `*` is one of `Parkinson`, `GarmanKlass`, or `RogersSatchell`. The `Vol` class always equals `sqrt` of the corresponding `Var` class (bit-exact in tests). Pick whichever output convention matches the rest of your pipeline.

## The three estimators

| Estimator | Inputs | Per-bar formula | Eff. vs C-to-C | Assumptions |
|---|---|---|---|---|
| **Parkinson** (1980) | H, L | $\dfrac{1}{4\ln 2}(\ln H/L)^2$ | ~5× | zero drift, no overnight gaps |
| **Garman-Klass** (1980) | O, H, L, C | $\tfrac{1}{2}(\ln H/L)^2 - (2\ln 2 - 1)(\ln C/O)^2$ | ~7.4× | zero drift, no overnight gaps |
| **Rogers-Satchell** (1991) | O, H, L, C | $\ln\tfrac{H}{C}\ln\tfrac{H}{O} + \ln\tfrac{L}{C}\ln\tfrac{L}{O}$ | ~6× | **drift-robust** |

"Efficiency vs close-to-close" is the multiplicative reduction in variance of the estimator for the same `n` bars under the model's assumptions. Rogers-Satchell is slightly less efficient than Garman-Klass but works correctly when the underlying drift is non-zero -- a much more realistic assumption for trending markets.

## Implementation Details

Each class is a pure composition: a closed-form per-bar arithmetic expression fed into a `detail::RollingMean` (for the `Rolling*` family) or an `EwMean` (for the `Ew*` family). O(1) amortised per step. The `*Vol` classes hold a `*Var` member and take the square root, so `*Vol[t] == sqrt(*Var[t])` exactly.

## Usage Example and Plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from screamer import RollingParkinsonVol, RollingGarmanKlassVol, RollingRogersSatchellVol

    rng = np.random.default_rng(0)
    n = 300
    open_ = 100 + np.cumsum(rng.normal(0.0, 1.0, n))
    close = open_ + rng.normal(0.0, 0.5, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0.0, 0.3, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0.0, 0.3, n))

    park = RollingParkinsonVol(20)(high, low)
    gk = RollingGarmanKlassVol(20)(open_, high, low, close)
    rs = RollingRogersSatchellVol(20)(open_, high, low, close)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=park, mode='lines', name='Parkinson(20)',
                             line=dict(color='steelblue')))
    fig.add_trace(go.Scatter(y=gk,   mode='lines', name='Garman-Klass(20)',
                             line=dict(color='green')))
    fig.add_trace(go.Scatter(y=rs,   mode='lines', name='Rogers-Satchell(20)',
                             line=dict(color='red')))
    fig.update_layout(
        title="Range-based volatility estimators on the same OHLC series",
        xaxis_title="Index", yaxis_title="sigma estimate",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.show()
```

## Yang-Zhang (completes the quartet)

`RollingYangZhangVar(window_size)` and `RollingYangZhangVol(window_size)` combine three variance components:

$$
\begin{aligned}
\sigma^2_o    &= \text{sample variance of overnight log returns } \ln(O_t / C_{t-1}) \\
\sigma^2_c    &= \text{sample variance of open-to-close log returns } \ln(C_t / O_t) \\
\sigma^2_{RS} &= \text{mean of per-bar Rogers-Satchell estimates} \\
k             &= \dfrac{0.34}{1.34 + (n+1)/(n-1)} \\
\sigma^2_{YZ} &= \sigma^2_o + k \cdot \sigma^2_c + (1-k) \cdot \sigma^2_{RS}
\end{aligned}
$$

**4-input, 1-output** on `(open, high, low, close)`. The only classical estimator that handles **both** drift *and* overnight gaps (~14× efficient vs close-to-close).

**Implementation**: composes two `RollingVar` (overnight + open-to-close log returns) and one `detail::RollingMean` (Rogers-Satchell per-bar). First valid output at sample index `window_size` -- we need n+1 price bars to form n overnight returns. Verified against a plain-numpy reference to ~1e-12.

No EW form is exposed because the `k` factor depends on a discrete window size; an "EW analogue" would require an arbitrary mapping from `span` to `n` that varies by convention.

## Reference

There is no canonical third-party reference that's directly comparable. Each per-bar formula is verified against a hand-rolled numpy implementation in `tests/test_range_volatility.py`, and the rolling / EW smoothers are validated against `pandas.Series.rolling(...).mean()` and `pandas.Series.ewm(...).mean()` to ~1e-15.
