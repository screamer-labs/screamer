# `CumMin`

## Description

The `CumMin` function returns the running minimum of all samples seen since the start of the stream (or since the last `reset`). The output is monotonically non-increasing while inputs are finite. It is the streaming equivalent of `numpy.minimum.accumulate`. Memory is `O(1)` regardless of how many samples have been processed.

This is an *expanding* (cumulative-from-zero) reduction, not a sliding window. For a fixed-window trough see `RollingMin`.

*Equation*:

$$
y[t] = \min_{0 \le i \le t} x[i]
$$

*Parameters*: none.

*NaN handling*: Once an input is NaN, every subsequent output is NaN. This matches `numpy.minimum.accumulate`.

## Usage Example and Plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from screamer import CumMin

    rng = np.random.default_rng(3)
    data = rng.normal(0.0, 1.0, size=200)
    running_min = CumMin()(data)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data, mode='lines',
                             name='x[t]', line=dict(color='steelblue')))
    fig.add_trace(go.Scatter(y=running_min, mode='lines',
                             name='CumMin(x)[t]',
                             line=dict(color='red', dash='dash')))
    fig.update_layout(
        title="CumMin: Running Minimum from t=0",
        xaxis_title="Index",
        yaxis_title="Value",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.show()
```

## Implementation Details

`CumMin` keeps a single `double` initialised to `+infinity`. Each input is compared and the smaller value retained. There is no warmup. The numpy reference is `numpy.minimum.accumulate`.
