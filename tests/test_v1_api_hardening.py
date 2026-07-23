import numpy as np
import pytest
import screamer
from screamer import EwKyleLambda, Propagator


def test_ewkylelambda_accepts_the_ew_mutex():
    rng = np.random.default_rng(0)
    flow = rng.standard_normal(100); ret = rng.standard_normal(100)
    # all four spellings construct and run
    for kw in [dict(span=20), dict(com=9.5), dict(halflife=10), dict(alpha=0.1)]:
        out = EwKyleLambda(**kw)(flow, ret)
        assert len(out) == 100
    with pytest.raises((ValueError, Exception)):
        EwKyleLambda()                       # none provided
    with pytest.raises((ValueError, Exception)):
        EwKyleLambda(span=20, alpha=0.1)     # two provided


def test_propagator_uses_window_size():
    x = np.arange(30.0)
    out = Propagator(window_size=10)(x)
    assert len(out) == 30
    with pytest.raises(TypeError):
        Propagator(window=10)                # old name is gone
