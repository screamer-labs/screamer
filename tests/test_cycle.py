import numpy as np
import pytest

from screamer import DominantCycle
from tests.regime_helpers import assert_batch_equals_scalar


def _tone(period, n=600, amp=1.0, seed=0):
    t = np.arange(n)
    return amp * np.sin(2 * np.pi * t / period)


class TestDominantCycle:
    @pytest.mark.parametrize("period", [14.0, 20.0, 30.0])
    def test_recovers_known_period(self, period):
        x = _tone(period, n=800)
        out = np.asarray(DominantCycle()(x))
        tail = out[-200:]
        tail = tail[np.isfinite(tail)]
        assert tail.size > 50
        # The homodyne discriminator recovers the dominant cycle within 15%.
        assert abs(np.median(tail) - period) / period < 0.15, \
            f"median {np.median(tail):.2f} vs period {period}"

    def test_batch_equals_stream(self):
        x = _tone(20.0, n=400)
        assert_batch_equals_scalar(lambda: DominantCycle(), x)
