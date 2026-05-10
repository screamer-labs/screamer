# `CumMax`

## Description

The `CumMax` function returns the running maximum of all samples seen since the start of the stream (or since the last `reset`). The output is monotonically non-decreasing while inputs are finite. It is the streaming equivalent of `numpy.maximum.accumulate`. Memory is `O(1)` regardless of how many samples have been processed.

This is an *expanding* (cumulative-from-zero) reduction, not a sliding window. For a fixed-window peak see `RollingMax`.

*Equation*:

$$
y[t] = \max_{0 \le i \le t} x[i]
$$

*Parameters*: none.

*NaN handling*: Once an input is NaN, every subsequent output is NaN. This matches `numpy.maximum.accumulate`.

## Usage Example and Plot

A canonical application is the *high-water mark* of a price or equity curve, which is the building block of drawdown:

$$
\text{drawdown}[t] = \frac{x[t]}{\text{CumMax}(x)[t]} - 1
$$

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import CumMax

    rng = np.random.default_rng(2)
    returns = rng.normal(0.0003, 0.012, size=400)
    price = 100.0 * np.cumprod(1.0 + returns)
    peak = CumMax()(price)
    drawdown = price / peak - 1.0

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.5], vertical_spacing=0.1)
    fig.add_trace(go.Scatter(y=price, mode='lines', name='Price'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(y=peak, mode='lines', name='CumMax (high-water mark)',
                             line=dict(color='green', dash='dash')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(y=drawdown, mode='lines', name='Drawdown',
                             fill='tozeroy', line=dict(color='red')),
                  row=2, col=1)
    fig.update_layout(
        title="CumMax: High-Water Mark and Drawdown",
        xaxis_title="Index",
        yaxis_title="Price",
        yaxis2_title="Drawdown",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.show()
```

## Implementation Details

`CumMax` keeps a single `double` initialised to `-infinity`. Each input is compared and the larger value retained. There is no warmup. The numpy reference is `numpy.maximum.accumulate`.
