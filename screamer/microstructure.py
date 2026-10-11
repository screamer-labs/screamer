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

    def __call__(self, *inputs):
        """Inputs: sign, then volume. Pass them as two separate arrays or
        iterators, one packed (T, ..., 2) array, or one iterator of
        (sign, volume) events."""
        return self._mul(*inputs)

    def reset(self):
        self._mul.reset()


class RollingKyleLambda:
    """Kyle's lambda over a trailing window: the price-impact / illiquidity slope
    of return on signed order flow (Kyle 1985). Specializes RollingBeta.
    """

    def __init__(self, window_size=20, start_policy="strict"):
        """__init__(self: RollingKyleLambda, window_size: int = 20, start_policy: str = 'strict') -> None"""
        self._beta = _s.RollingBeta(window_size, start_policy)

    def __call__(self, *inputs):
        """Inputs: signed flow, then return. Pass them as two separate arrays
        or iterators, one packed (T, ..., 2) array, or one iterator of
        (signed_flow, return_) events."""
        return self._beta(*_swap_fields(inputs))   # slope of return on flow

    def reset(self):
        self._beta.reset()


class EwKyleLambda:
    """Kyle's lambda, exponentially weighted (recursive). Specializes EwBeta."""

    def __init__(self, com=None, span=None, halflife=None, alpha=None):
        """__init__(self: EwKyleLambda, com: float = None, span: float = None, halflife: float = None, alpha: float = None) -> None"""
        self._beta = _s.EwBeta(com=com, span=span, halflife=halflife, alpha=alpha)

    def __call__(self, *inputs):
        """Inputs: signed flow, then return. Pass them as two separate arrays
        or iterators, one packed (T, ..., 2) array, or one iterator of
        (signed_flow, return_) events."""
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

    def __call__(self, *inputs):
        """Inputs: bid size, then ask size. Pass them as two separate arrays
        or iterators, one packed (T, ..., 2) array, or one iterator of
        (bid_size, ask_size) events."""
        return self._ofi(*inputs)

    def reset(self):
        self._ofi.reset()
