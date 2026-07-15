"""Microstructure and order-flow operators.

Causal streaming operators for order flow, price impact, and liquidity.
Stateless elementwise operators (such as OFI) are trivially causal on their
own. Operators that compose or alias screamer nodes inherit the batch == stream
guarantee from the engine. Popular models are exposed under their canonical name
with teaching-quality docs (see docs/functions_micro/).
"""
import numpy as np
from . import Diff, Sign, Abs, Div, Ffill

__all__ = ["OFI", "SignedVolume", "TickRuleSign"]


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


class SignedVolume:
    """Signed order flow: sign * volume (aggressor-signed volume)."""

    def __call__(self, sign, volume):
        return np.asarray(sign, dtype=float) * np.asarray(volume, dtype=float)


class TickRuleSign:
    """Trade sign by the tick rule: +1 on an up-tick, -1 on a down-tick, and the
    previous sign carried forward when the price is unchanged (first bar 0).

    Composed as Ffill(sign(diff) / |sign(diff)|): the division maps an unchanged
    tick (0/0) to NaN, and Ffill carries the last known +/-1 across it. Built on
    causal C++ operators, so it is causal and batch == stream by construction.
    """

    def __call__(self, price):
        price = np.asarray(price, dtype=float)
        d = Sign()(Diff(1)(price))          # -1 / 0 / +1; first bar NaN (Diff warmup)
        signed = Div()(d, Abs()(d))         # unchanged tick (0/0) -> NaN, carried by Ffill
        out = np.asarray(Ffill()(signed), dtype=float)
        out[np.isnan(price)] = np.nan       # missing price -> NaN (nan_policy: ignore)
        return out
