"""Scalar Kalman filter for a noisy random walk.

    predict: x_pred = x[t-1],            P_pred = P[t-1] + process_var
    update:  K = P_pred / (P_pred + observation_var)
             x[t] = x_pred + K * (z - x_pred)
             P[t] = (1 - K) * P_pred
"""
import numpy as np


class KalmanFilter_numpy:

    def __init__(self, process_var=0.01, observation_var=1.0,
                 initial_state=0.0, initial_variance=1.0):
        self.q, self.r = process_var, observation_var
        self.x0, self.p0 = initial_state, initial_variance

    def __call__(self, z):
        x, p = self.x0, self.p0
        out = np.empty(len(z), dtype=float)
        for i, obs in enumerate(np.asarray(z, dtype=float)):
            p_pred = p + self.q
            k = p_pred / (p_pred + self.r)
            x = x + k * (obs - x)
            p = (1.0 - k) * p_pred
            out[i] = x
        return out
