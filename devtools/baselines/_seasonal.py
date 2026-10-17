"""The slot windows of the seasonal operators: step t belongs to slot
t mod period, and each slot keeps its last window_size finite values. A NaN
step advances the slot and enters no window."""
import numpy as np


def slot_windows(x, period, window_size, start_policy, minimum):
    """For each step, the values of its slot's window (newest last), or None
    where the step is NaN or the policy gives no output."""
    x = np.asarray(x, dtype=float)
    slots = [[0.0] * window_size if start_policy == "zero" else [] for _ in range(period)]
    out = []
    for t, v in enumerate(x):
        s = slots[t % period]
        if np.isnan(v):
            out.append(None)
            continue
        s.append(v)
        if len(s) > window_size:
            s.pop(0)
        need = window_size if start_policy == "strict" else minimum
        out.append(np.array(s) if len(s) >= need else None)
    return out
