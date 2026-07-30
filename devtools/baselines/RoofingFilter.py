import numpy as np
from scipy.signal import lfilter


class RoofingFilter_ehlers:
    def __init__(self, hp_period=None, lp_period=None, hp_cutoff=None, lp_cutoff=None):
        if (hp_period is None) == (hp_cutoff is None):
            raise ValueError("Provide exactly one of hp_period or hp_cutoff.")
        if (lp_period is None) == (lp_cutoff is None):
            raise ValueError("Provide exactly one of lp_period or lp_cutoff.")
        hp = hp_period if hp_period is not None else 2.0 / hp_cutoff
        lp = lp_period if lp_period is not None else 2.0 / lp_cutoff
        w = 0.707 * 2 * np.pi / hp
        alpha = (np.cos(w) + np.sin(w) - 1) / np.cos(w)
        g, om = 1 - alpha / 2, 1 - alpha
        self.b_hp = [g * g, -2 * g * g, g * g]
        self.a_hp = [1.0, -2 * om, om * om]
        a1 = np.exp(-np.sqrt(2) * np.pi / lp)
        b1 = 2 * a1 * np.cos(np.sqrt(2) * np.pi / lp)
        c2, c3 = b1, -a1 * a1
        c1 = 1 - c2 - c3
        self.b_lp = [c1 / 2, c1 / 2]
        self.a_lp = [1.0, -c2, -c3]

    def __call__(self, array):
        x = np.asarray(array, float)
        return lfilter(self.b_lp, self.a_lp, lfilter(self.b_hp, self.a_hp, x))
