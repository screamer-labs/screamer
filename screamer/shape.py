"""Name the axes of data with a layout string and reshape it to match.

Every array in screamer reads as ``(T, *A, F?)``: time on axis 0, zero or more
asset axes of independent series, and an optional last axis of fields. A layout
names an array's axes with the letters ``T``, ``A`` and ``F``; ``->`` gives the
wanted result. The rules live in C++ (``include/screamer/common/shape.h``);
this module applies them to each container.
"""
import numpy as np

from . import screamer_bindings as _b
from .dag import _is_vi_pair

__all__ = ["reshape"]


def reshape_array(x, layout):
    """Reshape one NumPy array by a layout; a view unless axes are reordered."""
    x = np.asarray(x)
    _, source, dropped, _ = _b._layout_plan(layout, list(x.shape))
    if dropped:
        x = np.squeeze(x, axis=tuple(dropped))
    kept = [a for a in range(x.ndim + len(dropped)) if a not in dropped]
    position = {axis: n for n, axis in enumerate(kept)}
    x = np.transpose(x, [position[a] for a in source if a >= 0])
    for axis, a in enumerate(source):
        if a < 0:
            x = np.expand_dims(x, axis)
    return x


def _event(e, layout):
    if not layout.replace(" ", "").startswith("T"):
        raise ValueError(
            f"layout {layout!r}: for a sequence of events the layout must start with T, "
            "the axis of the sequence")
    return reshape_array(np.asarray(e, dtype=np.float64)[None], layout)[0]


def frame_axes(obj):
    """Values and per-axis labels of a pandas Series or DataFrame.

    A DataFrame with two-level columns reads as three axes, (index, level 0,
    level 1), and must hold every pair of the two levels.
    """
    import pandas as pd
    if isinstance(obj, pd.Series):
        return obj.to_numpy(dtype=np.float64), [obj.index]
    cols = obj.columns
    if isinstance(cols, pd.MultiIndex):
        if not is_product_frame(obj):
            if cols.nlevels != 2:
                raise ValueError("a DataFrame for screamer has flat or two-level columns")
            raise ValueError(
                "a DataFrame with two-level columns must hold every pair of the two "
                "levels, e.g. every (asset, field)")
        levels = [cols.unique(level=0), cols.unique(level=1)]
        full = pd.MultiIndex.from_product(levels)
        values = obj[full].to_numpy(dtype=np.float64).reshape(
            len(obj), len(levels[0]), len(levels[1]))
        return values, [obj.index, levels[0], levels[1]]
    return obj.to_numpy(dtype=np.float64), [obj.index, cols]


def is_product_frame(obj):
    """True for a DataFrame whose two-level columns hold every pair of the levels."""
    import pandas as pd
    cols = getattr(obj, "columns", None)
    if not isinstance(cols, pd.MultiIndex) or cols.nlevels != 2:
        return False
    full = pd.MultiIndex.from_product([cols.unique(level=0), cols.unique(level=1)])
    return len(cols) == len(full) and set(cols) == set(full)


def _frame(obj, layout):
    import pandas as pd
    values, labels = frame_axes(obj)
    out, source, _, _ = _b._layout_plan(layout, list(values.shape))
    arr = reshape_array(values, layout)
    names = [labels[a] if a >= 0 else (["value"] if letter == "F" else [0])
             for letter, a in zip(out, source)]
    if arr.ndim == 1:
        return pd.Series(arr, index=names[0])
    if arr.ndim == 2:
        return pd.DataFrame(arr, index=names[0], columns=names[1])
    if arr.ndim == 3:
        return pd.DataFrame(arr.reshape(len(arr), -1), index=names[0],
                            columns=pd.MultiIndex.from_product([names[1], names[2]]))
    raise ValueError(f"layout {layout!r} gives {arr.ndim} axes; a DataFrame holds at most 3")


def reshape(x, layout):
    """Reshape data by a layout string and return the same kind of container.

    ``layout`` names the axes of ``x`` in order with ``T`` (time), ``A`` (asset)
    and ``F`` (field). Without ``->`` the result is the canonical
    ``(T, *A, F)``; with ``->`` it is the layout on the right. A letter missing
    on the left is inserted with size 1; a letter missing on the right is
    dropped and must have size 1.

    ``x`` may be a NumPy array, a ``(values, index)`` tuple, a pandas Series or
    DataFrame, a list of events, or an iterator of events (``T`` then refers to
    the sequence).

    >>> reshape(np.zeros((100, 3)), "TA").shape
    (100, 3, 1)
    >>> reshape(np.zeros((100, 3, 1)), "TAF->TA").shape
    (100, 3)
    """
    import pandas as pd
    if isinstance(x, np.ndarray):
        return reshape_array(x, layout)
    if _is_vi_pair(x):
        return reshape_array(x[0], layout), x[1]
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return _frame(x, layout)
    if isinstance(x, list):
        return [_event(e, layout) for e in x]
    if hasattr(x, "__next__"):
        return (_event(e, layout) for e in x)
    raise TypeError(
        "reshape takes a NumPy array, a (values, index) tuple, a pandas Series or "
        f"DataFrame, a list of events or an iterator of events; got {type(x).__name__}")
