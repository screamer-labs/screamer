"""Every functor must be registered under EvalOp (exposes num_inputs/num_outputs)."""
import pytest
import screamer

# (constructor-args, expected num_inputs, expected num_outputs) per functor.
# Covers the arity spread and both base categories.
CASES = {
    # FunctorBase (now under EvalOp)
    "ATR": ((14,), 3, 1),
    "AD": ((), 4, 1),
    "ADOSC": ((3, 10), 4, 1),
    "BOP": ((), 4, 1),
    "MFI": ((14,), 4, 1),
    "Stoch": ((14,), 3, 2),
    "MACD": ((), 1, 3),
    "KeltnerChannels": ((20,), 3, 3),
    "DonchianChannels": ((20,), 2, 3),
    "RollingMinMax": ((10,), 1, 2),
    "RollingLinearRegression": ((10,), 2, 4),
    "Hypot": ((), 2, 1),
    "RollingSpread": ((20,), 2, 1),
    "WilliamsR": ((14,), 3, 1),
    "StochRSI": ((14,), 1, 2),
    # ScreamerBase-bound-bare (now under ScreamerBase -> inherits EvalOp)
    "RollingPoly1": ((10,), 1, 1),
    "RollingPoly2": ((10,), 1, 1),
    "RollingSigmaClip": ((10,), 1, 1),
    "RollingOU": ((10,), 1, 1),
}


@pytest.mark.parametrize("name,args,n_in,n_out", [(k, *v) for k, v in CASES.items()])
def test_functor_arity_registered(name, args, n_in, n_out):
    cls = getattr(screamer, name)
    obj = cls(*args)
    assert obj.num_inputs == n_in, f"{name}.num_inputs"
    assert obj.num_outputs == n_out, f"{name}.num_outputs"


def test_screamerbase_bare_keeps_process_scalar():
    # RollingPoly2 was bound bare; after `, ScreamerBase` it must still expose
    # process_scalar (would be lost if wrongly bound under EvalOp).
    f = screamer.RollingPoly2(10)
    assert hasattr(f, "process_scalar")
    assert isinstance(f.process_scalar(1.0), float)
