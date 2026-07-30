"""General-order IIR Butterworth band-pass filter.

`scipy.signal.butter` designs the coefficients and `lfilter` applies them
causally, which is the same contract screamer implements. Cutoffs are a
fraction of Nyquist in (0, 1), which is scipy's normalised `Wn`.
"""
import numpy as np
from scipy.signal import butter, lfilter


class ButterBandpass_scipy:

    def __init__(self, order=2, low_cutoff=0.05, high_cutoff=0.2):
        self.b, self.a = butter(order, [low_cutoff, high_cutoff], btype="band", analog=False)

    def __call__(self, x):
        return lfilter(self.b, self.a, np.asarray(x, dtype=float))
