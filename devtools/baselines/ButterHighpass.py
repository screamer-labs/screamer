"""General-order IIR Butterworth high-pass filter.

`scipy.signal.butter` designs the coefficients and `lfilter` applies them
causally, which is the same contract screamer implements. Cutoffs are a
fraction of Nyquist in (0, 1), which is scipy's normalised `Wn`.
"""
import numpy as np
from scipy.signal import butter, lfilter


class ButterHighpass_scipy:

    def __init__(self, order=2, cutoff_freq=0.1):
        self.b, self.a = butter(order, cutoff_freq, btype="high", analog=False)

    def __call__(self, x):
        return lfilter(self.b, self.a, np.asarray(x, dtype=float))
