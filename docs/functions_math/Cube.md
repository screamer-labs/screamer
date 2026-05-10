# `Cube`

## Description

Cube each element. Equivalent to `Power(3)` but skips the `std::pow` logarithm.

*Equation*:

$$
y[i] = x[i]^3
$$

*Parameters*: `Cube` takes no parameters.

*NaN handling*: `NaN` values pass through unchanged.

## Usage Example and Plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import Cube

    np.random.seed(0)
    data = np.random.normal(size=80) * 1.5
    transformed = Cube()(data)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.5], vertical_spacing=0.1)
    fig.add_trace(go.Scatter(y=data, mode="lines+markers",
                             name="input"), row=1, col=1)
    fig.add_trace(go.Scatter(y=transformed, mode="lines+markers",
                             name="Cube",
                             line=dict(color="red")), row=2, col=1)
    fig.update_layout(
        title="Cube transformation",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="input", row=1, col=1)
    fig.update_yaxes(title_text="Cube(input)", row=2, col=1)
    fig.show()
```

## Reference

Equivalent to `x ** 3` in numpy.
