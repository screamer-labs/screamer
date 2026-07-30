"""The sliding-window extremum is O(1) per sample whatever the data does.

`RollingMax` and friends use a monotonic deque: amortised O(1) with a bounded
worst case. The obvious alternative, which TA-Lib uses, is to track the current
extremum and rescan the window when it expires. That is faster on typical data
and degrades to O(window) per sample on a monotone run, because the extremum
then expires at every step.

Measured on 1M samples, `RollingMax(50)` against `talib.MAX`:

    input                  screamer   TA-Lib
    random uniform             8.6      1.2
    monotonic increasing       2.4      0.5
    monotonic decreasing       3.0     16.2
    random walk               10.6      3.0

A monotonically falling series is not a contrived input; it is a sell-off, and
it is exactly when a latency spike is least welcome. This test pins the
property that motivates the choice: per-sample cost must not blow up on a
monotone run.

It asserts a ratio between two timings in the same process rather than an
absolute time, so it does not depend on the machine. The threshold is loose
because timing in CI is noisy; the failure it guards against is a change of
algorithmic class, which would show as a factor of window_size.
"""
import time

import numpy as np
import pytest

from screamer import RollingArgmax, RollingMax, RollingMin, RollingRange

N = 200_000
WINDOW = 200


def _seconds(op_factory, data, repeat=3):
    return min(
        (lambda t0=time.perf_counter(): (op_factory()(data), time.perf_counter() - t0)[1])()
        for _ in range(repeat)
    )


@pytest.mark.parametrize("factory,name", [
    (lambda: RollingMax(WINDOW), "RollingMax"),
    (lambda: RollingMin(WINDOW), "RollingMin"),
    (lambda: RollingRange(WINDOW), "RollingRange"),
    (lambda: RollingArgmax(WINDOW), "RollingArgmax"),
])
def test_monotone_input_does_not_degrade(factory, name):
    """A monotone run must not cost a window scan per sample.

    Both directions are checked: for a max, a rising series keeps replacing the
    extremum and a falling one keeps expiring it. A rescan-on-expiry
    implementation would blow up on one of the two.
    """
    rng = np.random.default_rng(0)
    random_input = rng.uniform(0.1, 10.0, N)
    rising = np.arange(N, dtype=float)
    falling = np.arange(N, 0, -1, dtype=float)

    baseline = _seconds(factory, random_input)
    for label, data in (("rising", rising), ("falling", falling)):
        elapsed = _seconds(factory, data)
        assert elapsed < 10 * baseline, (
            f"{name} took {elapsed / baseline:.1f}x longer on a {label} series "
            f"than on random input. A monotone run should stay O(1) per sample; "
            f"this looks like a scan of the {WINDOW}-sample window."
        )
