"""The start policies of screamer's IIR filters, on top of scipy's lfilter.

"zero" is plain lfilter. "steady" starts from the steady state for a constant
input equal to the first non-NaN value, lfilter_zi(b, a) * x0, so a low-pass
output starts at x0 and a high-pass output at 0. NaN inputs are skipped and
give NaN, as in screamer.
"""
import numpy as np
from scipy.signal import lfilter, lfilter_zi


def _column(b, a, x, start_policy):
    out = np.full_like(x, np.nan)
    ok = ~np.isnan(x)
    xf = x[ok]
    if xf.size == 0:
        return out
    if start_policy == "zero":
        out[ok] = lfilter(b, a, xf)
    elif start_policy == "steady":
        out[ok] = lfilter(b, a, xf, zi=lfilter_zi(b, a) * xf[0])[0]
    else:
        raise ValueError(f"unknown start_policy {start_policy!r}")
    return out


def iir_filter(b, a, x, start_policy="steady"):
    """Filter x (1-D, or 2-D along axis 0) with transfer function b / a."""
    x = np.asarray(x, dtype=float)
    if x.ndim <= 1:
        return _column(b, a, x.reshape(-1), start_policy).reshape(x.shape)
    return np.column_stack([_column(b, a, x[:, k], start_policy) for k in range(x.shape[1])])
