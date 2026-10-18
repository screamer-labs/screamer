"""Scalar Kalman filter for a noisy random walk.

    predict: x_pred = x[t-1],            P_pred = P[t-1] + process_var
    update:  K = P_pred / (P_pred + observation_var)
             x[t] = x_pred + K * (z - x_pred)
             P[t] = (1 - K) * P_pred

With a finite huber, after the first observation, an innovation e beyond huber
standard deviations (|z| = |e| / sqrt(P_pred + observation_var) > huber) uses
the observation variance observation_var * |z| / huber in the gain.
"""
import math

import numpy as np


class KalmanFilter_numpy:

    def __init__(self, process_var=0.01, observation_var=1.0,
                 initial_state=0.0, initial_variance=1.0, huber=math.inf):
        self.q, self.r, self.h = process_var, observation_var, huber
        self.x0, self.p0 = initial_state, initial_variance

    def __call__(self, z):
        x, p = self.x0, self.p0
        out = np.empty(len(z), dtype=float)
        for i, obs in enumerate(np.asarray(z, dtype=float)):
            p_pred = p + self.q
            r = self.r
            az = abs(obs - x) / math.sqrt(p_pred + r)
            if i > 0 and az > self.h:
                r = r * az / self.h
            k = p_pred / (p_pred + r)
            x = x + k * (obs - x)
            p = (1.0 - k) * p_pred
            out[i] = x
        return out
