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


def test_nonfinite_alpha_is_rejected():
    from screamer import EwMean, EwStd, BayesianRegression
    for op in [EwMean, EwStd]:
        with pytest.raises((ValueError, Exception)):
            op(alpha=float("nan"))
        with pytest.raises((ValueError, Exception)):
            op(alpha=float("inf"))
    with pytest.raises((ValueError, Exception)):
        BayesianRegression(alpha=float("nan"))


def test_hampel_impulseclip_string_output_modes():
    from screamer import Hampel, ImpulseClip
    rng = np.random.default_rng(42)
    x = rng.standard_normal(100)
    x[50] += 20.0   # one large spike
    for Op in (Hampel, ImpulseClip):
        cleaned = Op(window_size=5, output="cleaned")(x)
        flag = Op(window_size=5, output="flag")(x)
        nan = Op(window_size=5, output="nan")(x)
        # flag marks the spike as 1.0 somewhere; nan-mode puts a NaN where flag is 1
        assert np.nanmax(flag) == 1.0
        assert np.isnan(nan).any()
        with pytest.raises((ValueError, Exception)):
            Op(window_size=5, output="bogus")


def test_sigmaclip_ou_string_output_modes():
    from screamer import RollingSigmaClip, RollingOU
    rng = np.random.default_rng(1)
    x = rng.standard_normal(200)
    for mode in ("clipped", "mean", "std", "nan"):
        assert len(RollingSigmaClip(20, output=mode)(x)) == 200
    with pytest.raises((ValueError, Exception)):
        RollingSigmaClip(20, output="bogus")
    for mode in ("mrr", "mean", "relmean", "std"):
        assert len(RollingOU(20, output=mode)(np.cumsum(x))) == 200
    with pytest.raises((ValueError, Exception)):
        RollingOU(20, output="bogus")


def test_internal_names_not_public():
    import screamer
    assert "install_param_capture" not in dir(screamer)
    assert "screamer_bindings" not in dir(screamer)
