import numpy as np
from scipy.signal import butter

from ._iir_start import iir_filter

class Butter_scipy:
    def __init__(self, order, cutoff_freq, start_policy="steady"):
        self.start_policy = start_policy
        # Generate Butterworth filter coefficients
        self.b, self.a = butter(order, cutoff_freq, btype='low', analog=False)

    def __call__(self, array):
        return iir_filter(self.b, self.a, array, self.start_policy)
