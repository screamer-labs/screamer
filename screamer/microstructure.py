"""Microstructure and order-flow operators.

Causal streaming operators for order flow, price impact, and liquidity. Each one
either composes existing screamer operators or aliases one, so causality and the
batch == stream guarantee are inherited from the engine. Popular models are
exposed under their canonical name with teaching-quality docs (see
docs/functions_micro/).
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
        safe = np.where(total > 0, total, 1.0)
        return np.where(total > 0, (buy - sell) / safe, 0.0)
