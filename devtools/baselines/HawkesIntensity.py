import numpy as np


class HawkesIntensity_numpy:
    """Independent numpy reference for HawkesIntensity: the conditional intensity
    of an exponential-kernel Hawkes process, lambda_t = mu + kappa_t with
    kappa_{t+1} = decay * (kappa_t + alpha * x_t), kappa_0 = 0. Emit before update
    (causal); a NaN mark is ignored (emit NaN, state untouched)."""

    def __init__(self, decay=0.9, alpha=1.0, mu=0.0):
        self.decay = decay
        self.alpha = alpha
        self.mu = mu

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        out = np.empty(x.shape[0], dtype=float)
        kappa = 0.0
        for t in range(x.shape[0]):
            if np.isnan(x[t]):
                out[t] = np.nan                       # ignore: emit NaN, state untouched
                continue
            out[t] = self.mu + kappa                  # emit before update
            kappa = self.decay * (kappa + self.alpha * x[t])
        return out
