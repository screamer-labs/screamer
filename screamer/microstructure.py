"""Microstructure and order-flow operators (Python synonym layer).

The order-flow MODELS - order-flow imbalance, tick-rule / bulk-volume signing,
Kyle's lambda, Amihud illiquidity, Roll's spread, Hawkes intensity, and the
Bouchaud propagator - are C++ core operators (see the docs/functions_micro
pages). This module holds only the thin Python bindings the design allows: a
documented synonym of a single operator, or a specialization that binds one
operator's parameters. None of them add logic of their own.
"""
import numpy as np

# The inner operators come from the package namespace at construction time, not
# at import time: install_param_capture wraps the namespace classes after this
# module is imported, and only the wrapped classes can copy themselves once per
# asset for an iterator of asset events.
import screamer as _s

__all__ = ["SignedVolume", "RollingKyleLambda", "EwKyleLambda",
           "RollingOrderImbalance", "QueueImbalance"]


def _packed(x):
    """True if one input can hold every field: a packed array (last axis the
    fields), one tuple event, a list of events, or an iterator of events."""
    if hasattr(x, "ndim"):
        return x.ndim >= 2
    if isinstance(x, tuple):
        return True
    if isinstance(x, list):
        return bool(x) and not isinstance(x[0], (int, float, np.number))
    return hasattr(x, "__iter__")


def _bind(who, names, inputs, kwargs):
    """Merge 2.8.0 keyword inputs into the positional inputs.

    Positional inputs fill ``names`` from the left and keywords fill the rest,
    as for a Python signature. Without keywords the inputs pass through, so
    the packed and iterator forms are unchanged. One input that cannot hold
    every field, such as a 1-D array or a scalar, is a missing input.
    """
    for key in kwargs:
        if key not in names:
            raise TypeError(f"{who}() got an unexpected keyword argument {key!r}")
    if len(inputs) > len(names):
        raise TypeError(f"{who}() takes {len(names)} inputs, got {len(inputs)} positional")
    for name in names[:len(inputs)]:
        if name in kwargs:
            raise TypeError(f"{who}() got multiple values for argument {name!r}")
    if not kwargs and len(inputs) == 1 and _packed(inputs[0]):
        return inputs
    missing = [n for n in names[len(inputs):] if n not in kwargs]
    if missing:
        raise TypeError(f"{who}() missing {len(missing)} required input"
                        f"{'s' if len(missing) > 1 else ''}: "
                        + ", ".join(repr(n) for n in missing))
    return tuple(inputs) + tuple(kwargs[n] for n in names[len(inputs):])


def _swap_event(event):
    if isinstance(event, np.ndarray):
        return event[..., ::-1]
    return tuple(reversed(event))


def _swap_fields(inputs):
    """Reorder (signed_flow, return_) into the beta operator's (return_, flow).

    Takes the same input forms as any two-input operator: two separate inputs,
    one packed array whose last axis holds the two fields (reversed as a view,
    no copy), or one iterator of two-field events.
    """
    if len(inputs) == 2:
        return inputs[1], inputs[0]
    if len(inputs) != 1:
        raise TypeError(f"expected (signed_flow, return_) or one packed input, got {len(inputs)} inputs")
    packed = inputs[0]
    if isinstance(packed, np.ndarray):
        return (packed[..., ::-1],)
    if isinstance(packed, list):          # a list in gives a list out
        return ([_swap_event(e) for e in packed],)
    return (map(_swap_event, packed),)


class SignedVolume:
    """Signed order flow: sign * volume (aggressor-signed volume). A documented
    synonym of the elementwise Mul operator."""

    def __init__(self):
        """__init__(self: SignedVolume) -> None"""
        self._mul = _s.Mul()

    def __call__(self, *inputs, **kwargs):
        """Inputs: sign, then volume. Pass them as two separate arrays or
        iterators, one packed (T, ..., 2) array, or one iterator of
        (sign, volume) events. The keywords ``sign=`` and ``volume=`` also
        work."""
        return self._mul(*_bind("SignedVolume", ("sign", "volume"), inputs, kwargs))

    def reset(self):
        self._mul.reset()


class RollingKyleLambda:
    """Kyle's lambda over a trailing window: the price-impact / illiquidity slope
    of return on signed order flow (Kyle 1985). Specializes RollingBeta.
    """

    def __init__(self, window_size=20, start_policy="strict"):
        """__init__(self: RollingKyleLambda, window_size: int = 20, start_policy: str = 'strict') -> None"""
        self._beta = _s.RollingBeta(window_size, start_policy)

    def __call__(self, *inputs, **kwargs):
        """Inputs: signed flow, then return. Pass them as two separate arrays
        or iterators, one packed (T, ..., 2) array, or one iterator of
        (signed_flow, return_) events. The keywords ``signed_flow=`` and
        ``return_=`` also work."""
        inputs = _bind("RollingKyleLambda", ("signed_flow", "return_"), inputs, kwargs)
        return self._beta(*_swap_fields(inputs))   # slope of return on flow

    def reset(self):
        self._beta.reset()


class EwKyleLambda:
    """Kyle's lambda, exponentially weighted (recursive). Specializes EwBeta."""

    def __init__(self, com=None, span=None, halflife=None, alpha=None):
        """__init__(self: EwKyleLambda, com: float = None, span: float = None, halflife: float = None, alpha: float = None) -> None"""
        self._beta = _s.EwBeta(com=com, span=span, halflife=halflife, alpha=alpha)

    def __call__(self, *inputs, **kwargs):
        """Inputs: signed flow, then return. Pass them as two separate arrays
        or iterators, one packed (T, ..., 2) array, or one iterator of
        (signed_flow, return_) events. The keywords ``signed_flow=`` and
        ``return_=`` also work."""
        inputs = _bind("EwKyleLambda", ("signed_flow", "return_"), inputs, kwargs)
        return self._beta(*_swap_fields(inputs))

    def reset(self):
        self._beta.reset()


class RollingOrderImbalance:
    """Trailing-window sum of signed order flow (Chordia-Roll-Subrahmanyam order
    imbalance). Specializes RollingSum.
    """

    def __init__(self, window_size=20, start_policy="strict"):
        """__init__(self: RollingOrderImbalance, window_size: int = 20, start_policy: str = 'strict') -> None"""
        self._sum = _s.RollingSum(window_size, start_policy)

    def __call__(self, signed_flow):
        return self._sum(signed_flow)

    def reset(self):
        self._sum.reset()


class QueueImbalance:
    """L1 order-book (queue) imbalance: (bid_size - ask_size) / (bid_size +
    ask_size), in [-1, 1]. A documented synonym of OFI (the same normalized
    imbalance operator) applied to resting queue sizes rather than trade flow.
    """

    def __init__(self):
        """__init__(self: QueueImbalance) -> None"""
        self._ofi = _s.OFI()

    def __call__(self, *inputs, **kwargs):
        """Inputs: bid size, then ask size. Pass them as two separate arrays
        or iterators, one packed (T, ..., 2) array, or one iterator of
        (bid_size, ask_size) events. The keywords ``bid_size=`` and
        ``ask_size=`` also work."""
        return self._ofi(*_bind("QueueImbalance", ("bid_size", "ask_size"), inputs, kwargs))

    def reset(self):
        self._ofi.reset()
