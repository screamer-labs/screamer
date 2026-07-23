"""Microstructure and order-flow operators (Python synonym layer).

The order-flow MODELS - order-flow imbalance, tick-rule / bulk-volume signing,
Kyle's lambda, Amihud illiquidity, Roll's spread, Hawkes intensity, and the
Bouchaud propagator - are C++ core operators (see the docs/functions_micro
pages). This module holds only the thin Python bindings the design allows: a
documented synonym of a single operator, or a specialization that binds one
operator's parameters. None of them add logic of their own.
"""
from . import Mul, RollingSum
from .screamer_bindings import RollingBeta, EwBeta, OFI

__all__ = ["SignedVolume", "RollingKyleLambda", "EwKyleLambda",
           "RollingOrderImbalance", "QueueImbalance"]


class SignedVolume:
    """Signed order flow: sign * volume (aggressor-signed volume). A documented
    synonym of the elementwise Mul operator."""

    def __init__(self):
        """__init__(self: SignedVolume) -> None"""
        self._mul = Mul()

    def __call__(self, sign, volume):
        return self._mul(sign, volume)

    def reset(self):
        self._mul.reset()


class RollingKyleLambda:
    """Kyle's lambda over a trailing window: the price-impact / illiquidity slope
    of return on signed order flow (Kyle 1985). Specializes RollingBeta.
    """

    def __init__(self, window_size=20, start_policy="strict"):
        """__init__(self: RollingKyleLambda, window_size: int = 20, start_policy: str = 'strict') -> None"""
        self._beta = RollingBeta(window_size, start_policy)

    def __call__(self, signed_flow, return_):
        return self._beta(return_, signed_flow)   # slope of return on flow

    def reset(self):
        self._beta.reset()


class EwKyleLambda:
    """Kyle's lambda, exponentially weighted (recursive). Specializes EwBeta."""

    def __init__(self, com=None, span=None, halflife=None, alpha=None):
        """__init__(self: EwKyleLambda, com: float = None, span: float = None, halflife: float = None, alpha: float = None) -> None"""
        self._beta = EwBeta(com=com, span=span, halflife=halflife, alpha=alpha)

    def __call__(self, signed_flow, return_):
        return self._beta(return_, signed_flow)

    def reset(self):
        self._beta.reset()


class RollingOrderImbalance:
    """Trailing-window sum of signed order flow (Chordia-Roll-Subrahmanyam order
    imbalance). Specializes RollingSum.
    """

    def __init__(self, window_size=20, start_policy="strict"):
        """__init__(self: RollingOrderImbalance, window_size: int = 20, start_policy: str = 'strict') -> None"""
        self._sum = RollingSum(window_size, start_policy)

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
        self._ofi = OFI()

    def __call__(self, bid_size, ask_size):
        return self._ofi(bid_size, ask_size)

    def reset(self):
        self._ofi.reset()
