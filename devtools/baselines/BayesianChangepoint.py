"""Adams-MacKay online Bayesian changepoint detection, written as a direct
numpy transcription of the recursion: constant hazard 1 / expected_run, a
normal-gamma prior (mean 0, kappa 1, alpha 1, beta prior_var), Student-t
predictives, and the run-length distribution truncated at max_run. The value
of a step belongs to the run it starts or continues, so a run of length r has
seen r + 1 values."""
import math

import numpy as np


def _t_logpdf(x, mu, s2, nu):
    return (math.lgamma((nu + 1.0) / 2.0) - math.lgamma(nu / 2.0) - 0.5 * np.log(nu * math.pi * s2)
            - (nu + 1.0) / 2.0 * np.log1p((x - mu) ** 2 / (nu * s2)))


class BayesianChangepoint_numpy:
    def __init__(self, expected_run=250.0, max_run=1000, prior_var=1.0):
        self.h, self.R, self.b0 = 1.0 / expected_run, max_run + 1, prior_var

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        out = np.full((len(x), 3), np.nan)
        prob = mu = beta = None
        for t, v in enumerate(x):
            if math.isnan(v):
                continue
            if prob is not None:
                with np.errstate(all="ignore"):
                    r = np.arange(len(prob))
                    kappa, alpha = r + 2.0, 1.5 + r / 2.0
                    lp = np.array([_t_logpdf(v, mu[i], beta[i] * (kappa[i] + 1) / (alpha[i] * kappa[i]),
                                             2.0 * alpha[i]) for i in range(len(prob))])
                    lp0 = _t_logpdf(v, 0.0, 2.0 * self.b0, 2.0)
                    lp = np.where(np.isnan(lp), -np.inf, lp)
                    if not max(lp0, lp.max()) > -np.inf:
                        prob = None          # every term is -inf: restart at the prior
            if prob is None:
                prob = np.array([1.0])
                mu, beta = np.array([v / 2.0]), np.array([self.b0 + v * v / 4.0])
            else:
                m = max(lp0, lp.max())
                grow = prob * np.exp(lp - m) * (1.0 - self.h)
                new = np.concatenate([[self.h * math.exp(lp0 - m)], grow])[: self.R]
                prob = new / new.sum()
                mu_g = (kappa * mu + v) / (kappa + 1.0)
                beta_g = beta + kappa * (v - mu) ** 2 / (2.0 * (kappa + 1.0))
                mu = np.concatenate([[v / 2.0], mu_g])[: self.R]
                beta = np.concatenate([[self.b0 + v * v / 4.0], beta_g])[: self.R]
            r = np.arange(len(prob))
            kappa, alpha = r + 2.0, 1.5 + r / 2.0
            var = beta * (kappa + 1.0) / (kappa * (alpha - 1.0))
            live = prob > 0.0
            pl, rl, ml, vl = prob[live], r[live], mu[live], var[live]
            with np.errstate(all="ignore"):
                mean = pl @ ml
                out[t] = (pl @ rl, mean, math.sqrt(max(pl @ (vl + ml * ml) - mean * mean, 0.0)))
        return out
