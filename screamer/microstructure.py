"""Microstructure and order-flow operators.

Causal streaming operators for order flow, price impact, and liquidity.
Stateless elementwise operators (such as OFI) are trivially causal on their
own. Operators that compose or alias screamer nodes inherit the batch == stream
guarantee from the engine. Popular models are exposed under their canonical name
with teaching-quality docs (see docs/functions_micro/).
"""
import numpy as np

__all__ = ["OFI"]


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
