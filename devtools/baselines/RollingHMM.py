"""Two-state hidden Markov model refit on a trailing window, written as a
direct numpy transcription of the algorithm in include/screamer/rolling_hmm.h:
the deterministic first split, Baum-Welch with a scaled forward-backward pass,
the variance floor, the state ordering, the warm start, and forward filtering
between refits."""
import math

import numpy as np

_LOG2PI = math.log(2.0 * math.pi)
_A0 = ((0.95, 0.05), (0.05, 0.95))


class _Params:
    def __init__(self, pi, A, mu, phi, var):
        self.pi = np.array(pi, dtype=float)
        self.A = np.array(A, dtype=float)
        self.mu = np.array(mu, dtype=float)
        self.phi = np.array(phi, dtype=float)
        self.var = np.array(var, dtype=float)

    def copy(self):
        return _Params(self.pi, self.A, self.mu, self.phi, self.var)


class RollingHMM_numpy:
    def __init__(self, window_size=720, refit_every=24, emission="gaussian", max_iter=50, tol=1e-6):
        self.W, self.R = window_size, refit_every
        self.ar1 = emission == "ar1"
        self.max_iter, self.tol = max_iter, tol

    def _means(self, th, z):
        if self.ar1:
            return th.mu[None, :] + th.phi[None, :] * z[:, None]
        return np.broadcast_to(th.mu, (len(z), 2))

    def _logb(self, th, y, z):
        d = y[:, None] - self._means(th, z)
        return -0.5 * (_LOG2PI + np.log(th.var))[None, :] - d * d / (2.0 * th.var[None, :])

    def _forward(self, th, lb):
        N = len(lb)
        alpha, eb, scale = np.empty((N, 2)), np.empty((N, 2)), np.empty(N)
        ll, prev = 0.0, None
        for i in range(N):
            mx = max(lb[i, 0], lb[i, 1])
            eb[i] = np.exp(lb[i] - mx)
            q = th.pi if i == 0 else prev @ th.A
            a = q * eb[i]
            s = a[0] + a[1]
            if not (s > 0.0 and math.isfinite(s)):
                return math.nan, alpha, eb, scale
            alpha[i] = a / s
            scale[i] = s
            ll += math.log(s) + mx
            prev = alpha[i]
        return ll, alpha, eb, scale

    @staticmethod
    def _backward(th, eb, scale):
        N = len(eb)
        beta = np.ones((N, 2))
        for i in range(N - 2, -1, -1):
            beta[i] = (th.A @ (eb[i + 1] * beta[i + 1])) / scale[i + 1]
        return beta

    def _mstep(self, th, y, z, alpha, beta, eb, scale, center, floor):
        g = alpha * beta
        g = g / g.sum(axis=1, keepdims=True)
        new = th.copy()
        new.pi = g[0] / g[0].sum()
        X = np.zeros((2, 2))
        for i in range(len(y) - 1):
            X += alpha[i][:, None] * th.A * (eb[i + 1] * beta[i + 1])[None, :] / scale[i + 1]
        for j in range(2):
            den = X[j].sum()
            if den > 0.0:
                new.A[j] = X[j] / den
        for k in range(2):
            w = g[:, k]
            S0 = w.sum()
            if not S0 > 0.0:
                continue
            if self.ar1:
                yc, zc = y - center, z - center
                Sz, Sy = (w * zc).sum(), (w * yc).sum()
                Szz, Szy = (w * zc * zc).sum(), (w * zc * yc).sum()
                den = S0 * Szz - Sz * Sz
                phi = (S0 * Szy - Sz * Sy) / den if den > 1e-12 * S0 * Szz else 0.0
                c = (Sy - phi * Sz) / S0 + center * (1.0 - phi)
                r = y - c - phi * z
            else:
                phi = 0.0
                c = center + (w * (y - center)).sum() / S0
                r = y - c
            new.mu[k], new.phi[k] = c, phi
            new.var[k] = max((w * r * r).sum() / S0, floor)
        return new

    def _initial(self, win, floor):
        if self.ar1:
            xm = win.mean()
            d = win - xm
            c0, c1 = (d * d).sum(), (d[1:] * d[:-1]).sum()
            rho = c1 / c0 if c0 > 0.0 else 0.0
            phi = np.clip([rho - 0.5, rho + 0.5], -0.99, 0.99)
            v = max(c0 / len(win) * (1.0 - rho * rho), floor)
            return _Params((0.5, 0.5), _A0, xm * (1.0 - phi), phi, (v, v))
        a = np.abs(win)
        med = np.median(a)
        lo, hi = win[a <= med], win[a > med]
        if len(hi) == 0:
            lo = hi = win
        return _Params((0.5, 0.5), _A0, (lo.mean(), hi.mean()), (0.0, 0.0),
                       (max(lo.var(), floor), max(hi.var(), floor)))

    def _order(self, th):
        key = th.phi if self.ar1 else th.var
        if key[0] > key[1]:
            P = [1, 0]
            th = _Params(th.pi[P], th.A[np.ix_(P, P)], th.mu[P], th.phi[P], th.var[P])
        return th

    def _fit(self, win, warm):
        y, z = (win[1:], win[:-1]) if self.ar1 else (win, win)
        center = win.mean()
        floor = max(1e-6 * win.var(), 1e-300)
        if warm is None:
            th = self._initial(win, floor)
        else:
            th = warm.copy()
            th.pi = np.array([0.5, 0.5])
        ll_prev = -math.inf
        for it in range(self.max_iter):
            ll, alpha, eb, scale = self._forward(th, self._logb(th, y, z))
            if math.isnan(ll) or (it > 0 and not (ll - ll_prev >= self.tol)):
                break
            beta = self._backward(th, eb, scale)
            th = self._mstep(th, y, z, alpha, beta, eb, scale, center, floor)
            ll_prev = ll
        th = self._order(th)
        ll, alpha, _, _ = self._forward(th, self._logb(th, y, z))
        return th, (np.array([0.5, 0.5]) if math.isnan(ll) else alpha[-1].copy())

    def _filter(self, th, p, x, prev):
        m = th.mu + th.phi * prev if self.ar1 else th.mu
        lb = -0.5 * (_LOG2PI + np.log(th.var)) - (x - m) ** 2 / (2.0 * th.var)
        q = p @ th.A
        a = q * np.exp(lb - lb.max())
        s = a.sum()
        return a / s if (s > 0.0 and math.isfinite(s)) else q / q.sum()

    def _outputs(self, th, p, x):
        q = p @ th.A
        m = th.mu + th.phi * x if self.ar1 else th.mu
        mean = q @ m
        var = q @ (th.var + m * m) - mean * mean
        return p[1], mean, math.sqrt(max(var, 0.0))

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        out = np.full((len(x), 3), np.nan)
        buf, th, p, since, prev = [], None, None, 0, 0.0
        for t, v in enumerate(x):
            if math.isnan(v):
                continue
            buf.append(v)
            if len(buf) > self.W:
                buf.pop(0)
            before, prev = prev, v
            if len(buf) < self.W:
                continue
            win = np.array(buf)
            if th is None:
                th, p = self._fit(win, None)
                since = 0
            else:
                since += 1
                if since == self.R:
                    th, p = self._fit(win, th)
                    since = 0
                else:
                    p = self._filter(th, p, v, before)
            out[t] = self._outputs(th, p, v)
        return out
