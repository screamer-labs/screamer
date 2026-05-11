# `Momentum`

## Description

`Momentum(k)` returns the *raw price displacement* over `k` steps:

$$
\text{Momentum}[t] = x[t] - x[t-k]
$$

**This is exactly `Diff(k)`.** It exists as a separately-named class because TA-Lib calls this indicator `MOM` and traders look for "momentum" in the API. Internally `Momentum` is a thin subclass of `Diff` -- the implementation (delay buffer + subtraction) is shared, not duplicated.

*Parameters*:

- `window_size` (int, positive): the lookback `k`.
- `start_policy` (str, optional): `"strict"` (default), `"expanding"`, or `"zero"`. Same semantics as `Diff`.

## When to use which

| You want... | Use |
|---|---|
| TA-Lib parity (writing `MOM` in published strategies) | `Momentum(k)` |
| Any other context | `Diff(k)` |

Both produce identical output bit-for-bit (matches `talib.MOM` to 0.0 -- exact integer arithmetic).

## Reference

See [`Diff`](Diff.md) for the full documentation. Equivalent to `numpy.diff(x, n=1, prepend=NaN)[::k]`-like indexing or TA-Lib's `MOM`.
