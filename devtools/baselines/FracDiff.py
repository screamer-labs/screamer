"""Reference fractional differencing, transcribed from Lopez de Prado's
``getWeights_FFD`` / ``fracDiff_FFD`` (Advances in Financial Machine Learning,
chapter 5) rather than from screamer's implementation.

The book builds its weight vector oldest-first and dots it against a slice of
the series; screamer stores taps newest-first and convolves. The two agree by
construction, which is the point of deriving this one from the book.

NaN follows screamer's ``ignore`` policy: a NaN sample is not stored and does
not advance warmup, so the window is measured in finite samples.
"""
import numpy as np


class FracDiff_lopezdeprado:

    def __init__(self, d=0.4, window_size=100, threshold=1e-5, start_policy="strict"):
        self.d = d
        self.window_size = window_size
        self.threshold = threshold
        self.start_policy = start_policy
        self.weights = self._weights()

    def _weights(self):
        # weights[0] multiplies the current sample; the book's vector is this
        # one reversed.
        w = [1.0]
        for k in range(1, self.window_size):
            w_ = -w[-1] * (self.d - k + 1) / k
            if abs(w_) < self.threshold:
                break
            w.append(w_)
        return np.array(w)

    def __call__(self, array):
        x = np.asarray(array, dtype=float)
        width = len(self.weights)
        out = np.full(len(x), np.nan)
        finite = []
        for t, value in enumerate(x):
            if np.isnan(value):
                continue
            finite.append(value)
            if len(finite) < width and self.start_policy == "strict":
                continue
            window = np.array(finite[-width:][::-1])   # newest first
            out[t] = float(np.dot(self.weights[:len(window)], window))
        return out
