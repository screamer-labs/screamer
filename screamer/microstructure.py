"""Microstructure and order-flow operators.

Causal streaming operators for order flow, price impact, and liquidity.
Stateless elementwise operators (such as OFI) are trivially causal on their
own. Operators that compose or alias screamer nodes inherit the batch == stream
guarantee from the engine. Popular models are exposed under their canonical name
with teaching-quality docs (see docs/functions_micro/).
"""
import numpy as np
from . import Diff, Sign, Abs, Div, Ffill
from .screamer_bindings import RollingBeta, EwBeta, RollingMean

__all__ = ["OFI", "SignedVolume", "TickRuleSign", "RollingKyleLambda", "EwKyleLambda",
           "AmihudIlliquidity"]


class OFI:
    """Order-flow imbalance: (buy_volume - sell_volume) / (buy_volume + sell_volume).

    Cont-Kukanov-Stoikov style normalized signed flow, in [-1, 1]; 0 on an empty
    bucket. Stateless and elementwise, so it is trivially causal.
    """

    def __call__(self, buy_volume, sell_volume):
        buy = np.asarray(buy_volume, dtype=float)
        sell = np.asarray(sell_volume, dtype=float)
        total = buy + sell
        zero_total = (total == 0.0)                      # empty bucket -> 0.0
        safe = np.where(zero_total, 1.0, total)          # avoid 0/0; NaN stays NaN
        return np.where(zero_total, 0.0, (buy - sell) / safe)

    def reset(self):
        pass


class SignedVolume:
    """Signed order flow: sign * volume (aggressor-signed volume)."""

    def __call__(self, sign, volume):
        return np.asarray(sign, dtype=float) * np.asarray(volume, dtype=float)

    def reset(self):
        pass


class TickRuleSign:
    """Trade sign by the tick rule: +1 on an up-tick, -1 on a down-tick, and the
    previous sign carried forward when the price is unchanged (first bar NaN).

    Composed as Ffill(sign(diff) / |sign(diff)|): the division maps an unchanged
    tick (0/0) to NaN, and Ffill carries the last known +/-1 across it. The C++
    sub-operators are held on the instance so the state (Diff's previous price,
    Ffill's last value) advances whether the instance is driven with a whole
    array (batch) or one sample at a time (streaming) - so batch == stream holds.
    A missing price (NaN input) yields NaN (nan_policy: ignore).
    """

    def __init__(self):
        self._diff = Diff(1)
        self._sign = Sign()
        self._abs = Abs()
        self._div = Div()
        self._ffill = Ffill()

    def __call__(self, price):
        d = self._sign(self._diff(price))        # -1 / 0 / +1; first bar NaN (Diff warmup)
        signed = self._div(d, self._abs(d))      # unchanged tick (0/0) -> NaN, carried by Ffill
        out = self._ffill(signed)
        # missing price -> NaN; elementwise so it works for a scalar or an array step
        return np.where(np.isnan(price), np.nan, out)

    def reset(self):
        self._diff.reset()
        self._sign.reset()
        self._abs.reset()
        self._div.reset()
        self._ffill.reset()


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

    def __init__(self, span=20.0):
        """__init__(self: EwKyleLambda, span: float = 20.0) -> None"""
        self._beta = EwBeta(span=span)

    def __call__(self, signed_flow, return_):
        return self._beta(return_, signed_flow)

    def reset(self):
        self._beta.reset()


class AmihudIlliquidity:
    """Amihud (2002) illiquidity: rolling mean of |return| / notional. Large
    values mean price moves a lot per dollar traded (an illiquid, high-impact
    regime). A robust, cheap cousin of Kyle's lambda.
    """

    def __init__(self, window_size=20, start_policy="strict"):
        """__init__(self: AmihudIlliquidity, window_size: int = 20, start_policy: str = 'strict') -> None"""
        self._mean = RollingMean(window_size, start_policy)

    def __call__(self, return_, notional):
        ret = np.asarray(return_, dtype=float)
        notl = np.asarray(notional, dtype=float)
        ratio = np.where(notl == 0.0, np.nan, np.abs(ret) / np.where(notl == 0.0, 1.0, notl))
        return self._mean(ratio)

    def reset(self):
        self._mean.reset()
