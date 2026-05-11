# `DonchianChannels`, `KeltnerChannels`

Two classic envelope-style indicators that draw an upper and lower line around the price plus a midline. Useful for breakout detection, mean-reversion bounds, and stop placement.

## `DonchianChannels(window_size=20)`

Trend-following envelope. The upper line is the rolling max of `high`, the lower line is the rolling min of `low`, and the midline is their average:

$$
\begin{aligned}
\text{upper}[t] &= \max(\text{high},\ w\ \text{bars}) \\
\text{lower}[t] &= \min(\text{low},\ w\ \text{bars}) \\
\text{mid}[t]   &= (\text{upper} + \text{lower}) / 2
\end{aligned}
$$

**2-input, 3-output** (`FunctorBase<_, 2, 3>`). Inputs `(high, low)`; outputs `(lower, mid, upper)`. First valid at sample index `window_size - 1`.

Composition: two `detail::MonotonicDeque` instances (the same primitive shared by `RollingMin`/`Max`/`MinMax`/`Argmin`/`Argmax`/`Range`/`WilliamsR`/`Stoch`). Amortised O(1) per step. Bit-exact to `pandas-ta-classic.donchian`.

## `KeltnerChannels(window_size=20, num_atr=2.0)`

Volatility-adapted envelope. The midline is an EMA of close; the upper/lower lines are offset by a multiple of ATR:

$$
\begin{aligned}
\text{mid}[t]   &= \text{EMA}(\text{close},\ \text{window\_size}) \\
\text{atr}[t]   &= \text{ATR}(\text{high},\ \text{low},\ \text{close},\ \text{window\_size}) \\
\text{upper}[t] &= \text{mid} + \text{num\_atr} \cdot \text{atr} \\
\text{lower}[t] &= \text{mid} - \text{num\_atr} \cdot \text{atr}
\end{aligned}
$$

**3-input, 3-output** (`FunctorBase<_, 3, 3>`). Inputs `(high, low, close)`; outputs `(lower, mid, upper)`. First valid at sample index `window_size` (ATR's warmup). The EMA midline is gated together with the ATR-derived bands, so all three lines become valid simultaneously.

Composition: one `EwMean(span=window_size)` for the midline + one `ATR(window_size)`. O(1) per step.

## Output shape (both classes)

| You pass... | You get back... |
|---|---|
| scalars | tuple `(lower, mid, upper)` |
| 1D arrays of shape `(T,)` | array of shape `(T, 3)` |
| 2D arrays of shape `(T, K)` | array of shape `(T, K, 3)` |

`out[..., 0]` is `lower`, `out[..., 1]` is `mid`, `out[..., 2]` is `upper`.

## Usage Example and Plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from screamer import DonchianChannels, KeltnerChannels

    rng = np.random.default_rng(0)
    n = 300
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + np.abs(rng.normal(0, 0.5, n))
    low = close - np.abs(rng.normal(0, 0.5, n))

    dc = DonchianChannels(20)(high, low)
    kc = KeltnerChannels(20, num_atr=2.0)(high, low, close)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=close, mode='lines', name='Close',
                             line=dict(color='steelblue')))
    fig.add_trace(go.Scatter(y=dc[:, 2], mode='lines',
                             name='Donchian upper',
                             line=dict(color='green', dash='dot')))
    fig.add_trace(go.Scatter(y=dc[:, 0], mode='lines',
                             name='Donchian lower',
                             line=dict(color='green', dash='dot')))
    fig.add_trace(go.Scatter(y=kc[:, 2], mode='lines',
                             name='Keltner upper',
                             line=dict(color='red')))
    fig.add_trace(go.Scatter(y=kc[:, 1], mode='lines',
                             name='Keltner mid (EMA)',
                             line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(y=kc[:, 0], mode='lines',
                             name='Keltner lower',
                             line=dict(color='red')))
    fig.update_layout(
        title="Donchian (rolling H/L) vs Keltner (EMA +/- ATR)",
        xaxis_title="Index", yaxis_title="Price",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.show()
```

Donchian widens around volatile periods (recent H/L spread) but is otherwise insensitive to within-window movement. Keltner adapts to ATR, so it tightens during low-volatility regimes and widens during high-volatility ones, often more smoothly than Donchian.

## Reference

`DonchianChannels` matches `pandas-ta-classic.donchian` bit-exactly. `KeltnerChannels` has no canonical TA-Lib reference (TA-Lib lacks Keltner); the implementation matches the explicit `EwMean ± num_atr * ATR` composition bit-exactly.
