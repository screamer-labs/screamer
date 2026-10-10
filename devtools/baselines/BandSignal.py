"""Entry, exit, and stop bands on a z-score, written as a plain loop."""
import numpy as np


class BandSignal_numpy:

    def __init__(self, entry=2.0, exit=0.0, stop=np.inf):
        self.entry, self.exit, self.stop = entry, exit, stop

    def __call__(self, z):
        pos, lock_s, lock_l = 0.0, False, False
        out = np.empty(len(z))
        for i, v in enumerate(np.asarray(z, dtype=float)):
            if np.isnan(v):
                out[i] = np.nan
                continue
            if lock_s and v <= self.exit:
                lock_s = False
            if lock_l and v >= -self.exit:
                lock_l = False
            if pos == -1.0:
                if v >= self.stop:
                    pos, lock_s = 0.0, True
                elif v <= self.exit:
                    pos = 0.0
            elif pos == 1.0:
                if v <= -self.stop:
                    pos, lock_l = 0.0, True
                elif v >= -self.exit:
                    pos = 0.0
            if pos == 0.0:
                if self.entry <= v < self.stop and not lock_s:
                    pos = -1.0
                elif -self.stop < v <= -self.entry and not lock_l:
                    pos = 1.0
            out[i] = pos
        return out
