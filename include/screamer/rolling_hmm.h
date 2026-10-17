#ifndef SCREAMER_ROLLING_HMM_H
#define SCREAMER_ROLLING_HMM_H

// RollingHMM: a two-state hidden Markov model of a series (normally
// returns), refit on a trailing window by Baum-Welch EM and filtered forward
// one step at a time. 1 -> 3: [prob, pred_mean, pred_std].
//
// Emissions: "gaussian", state k emits N(mu_k, var_k) (W observations, the
// window); "ar1", state k emits x_t = c_k + phi_k x_{t-1} + e_t, e_t ~
// N(0, var_k) (Hamilton's Markov-switching AR(1); the W - 1 pairs of the
// window), with c_k kept in mu_k.
//
// Schedule: the first fit runs on the step the window first holds
// window_size values, then every refit_every steps. The first fit starts
// from a deterministic split ("gaussian": values with |x| at or below the
// median |x| form state 0, the rest state 1; "ar1": phi = rho -/+ 0.5,
// clipped to [-0.99, 0.99], rho the window's lag-1 autocorrelation), later
// ones from the previous fit with the initial distribution reset to
// (0.5, 0.5) and its variances floored for the current window. A warm start
// whose two states coincide, whose first E-step fails, or whose first E-step
// leaves a state without weight falls back to the deterministic split. EM stops after max_iter M-steps or when the
// log-likelihood gains less than tol; an M-step whose parameters give a
// non-finite log-likelihood is undone. Every variance is floored at 1e-6
// times the window variance. After a fit the states are ordered so that state 1 has the
// larger variance ("gaussian") or the larger phi ("ar1"), and a forward pass
// over the window gives the filtered probability. Between fits the forward
// recursion runs with fixed parameters on each new value.
//
// Outputs at t use values up to t: prob = P(state 1 | x_1..x_t); with
// next-state probabilities q = A^T p and component means m_k = mu_k or
// c_k + phi_k x_t, pred_mean = sum q_k m_k and pred_std the standard
// deviation of the mixture, sqrt(sum q_k (var_k + m_k^2) - pred_mean^2),
// computed as sqrt(q_0 var_0 + q_1 var_1 + q_0 q_1 (m_0 - m_1)^2).
//
// NaN before the first fit. nan_policy: ignore (a NaN gives a NaN row and
// leaves the window, the schedule, and the filter unchanged). A fit costs
// O(max_iter * window_size); a step between fits O(1). Every buffer is
// allocated at construction.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "screamer/common/finite_settings.h"
#include "screamer/common/float_info.h"
#include "screamer/common/functor_base.h"

namespace screamer {

class RollingHMM : public FunctorBase<RollingHMM, 1, 3> {
public:
    // The batch column loop calls call() out of line, the same code as a
    // per-event call: inlined, the compiler may contract the multiply-adds
    // differently and the last bits of batch and per-event output differ.
    static constexpr bool kInlineColumnCall = false;

    RollingHMM(int window_size = 720, int refit_every = 24,
               const std::string& emission = "gaussian", int max_iter = 50, double tol = 1e-6)
        : W_(check_window(window_size)), refit_every_(refit_every),
          ar1_(parse_emission(emission)), max_iter_(max_iter), tol_(tol)
    {
        require_finite("tol", tol);
        if (refit_every < 1) throw std::invalid_argument("refit_every must be 1 or more.");
        if (max_iter < 1) throw std::invalid_argument("max_iter must be 1 or more.");
        if (tol < 0.0) throw std::invalid_argument("tol must be 0 or more.");
        buf_.assign(W_, 0.0);
        win_.assign(W_, 0.0);
        eb_.assign(2 * W_, 0.0);
        alpha_.assign(2 * W_, 0.0);
        beta_.assign(2 * W_, 0.0);
        scale_.assign(W_, 0.0);
        gamma_.assign(2 * W_, 0.0);
        reset();
    }

    static bool parse_emission(const std::string& s) {
        if (s == "gaussian") return false;
        if (s == "ar1") return true;
        throw std::invalid_argument("emission must be \"gaussian\" or \"ar1\".");
    }

    void reset() override {
        head_ = 0;
        count_ = 0;
        since_ = 0;
        fitted_ = false;
        prev_ = 0.0;
        p_[0] = 0.5;
        p_[1] = 0.5;
        th_ = Params{};
    }

    ResultTuple call(const InputArray& in) override {
        const double x = in[0];
        const double nan = std::numeric_limits<double>::quiet_NaN();
        if (isnan2(x)) return std::make_tuple(nan, nan, nan);
        buf_[head_] = x;
        if (++head_ == W_) head_ = 0;
        if (count_ < W_) ++count_;
        const double prev = prev_;
        prev_ = x;
        if (count_ < W_) return std::make_tuple(nan, nan, nan);
        if (!fitted_) {
            fit();
            since_ = 0;
        } else if (++since_ == refit_every_) {
            fit();
            since_ = 0;
        } else {
            filter_step(x, prev);
        }
        return outputs(x);
    }

private:
    struct Params {
        double pi[2] = {0.5, 0.5};
        double A[2][2] = {{0.95, 0.05}, {0.05, 0.95}};
        double mu[2] = {0.0, 0.0};      // mean ("gaussian") or intercept c ("ar1")
        double phi[2] = {0.0, 0.0};     // AR coefficient ("ar1"); 0 for "gaussian"
        double var[2] = {1.0, 1.0};
        double logv[2] = {0.0, 0.0};    // log(var), from set_logv()
        void set_logv() {
            logv[0] = std::log(var[0]);
            logv[1] = std::log(var[1]);
        }
    };

    static constexpr double kLog2Pi = 1.8378770664093453;   // log(2 pi)

    static std::size_t check_window(int w) {
        if (w < 10) throw std::invalid_argument("window_size must be 10 or more.");
        return static_cast<std::size_t>(w);
    }

    std::size_t observations() const { return ar1_ ? W_ - 1 : W_; }
    const double* ys() const { return win_.data() + (ar1_ ? 1 : 0); }
    const double* zs() const { return win_.data(); }    // ys()[i]'s predecessor ("ar1")

    double log_b(const Params& th, int k, double y, double z) const {
        const double m = ar1_ ? th.mu[k] + th.phi[k] * z : th.mu[k];
        const double d = y - m;
        return -0.5 * (kLog2Pi + th.logv[k]) - d * d / (2.0 * th.var[k]);
    }

    // The window, oldest first.
    void load_window() {
        for (std::size_t i = 0; i < W_; ++i) {
            std::size_t j = head_ + i;
            if (j >= W_) j -= W_;
            win_[i] = buf_[j];
        }
    }

    // Scaled forward pass: alpha_ (filtered probabilities), scale_, and eb_
    // = exp(log b - max_k log b). The log-likelihood, or NaN when a step's
    // mass vanishes.
    double forward(const Params& th, std::size_t N) {
        const double* y = ys();
        const double* z = zs();
        double ll = 0.0;
        for (std::size_t i = 0; i < N; ++i) {
            const double l0 = log_b(th, 0, y[i], z[i]);
            const double l1 = log_b(th, 1, y[i], z[i]);
            const double mx = l0 > l1 ? l0 : l1;
            const double e0 = std::exp(l0 - mx), e1 = std::exp(l1 - mx);
            double q0 = th.pi[0], q1 = th.pi[1];
            if (i > 0) {
                const double* a = &alpha_[2 * (i - 1)];
                q0 = a[0] * th.A[0][0] + a[1] * th.A[1][0];
                q1 = a[0] * th.A[0][1] + a[1] * th.A[1][1];
            }
            const double a0 = q0 * e0, a1 = q1 * e1;
            const double s = a0 + a1;
            if (!(s > 0.0) || !std::isfinite(s)) return std::numeric_limits<double>::quiet_NaN();
            alpha_[2 * i] = a0 / s;
            alpha_[2 * i + 1] = a1 / s;
            scale_[i] = s;
            eb_[2 * i] = e0;
            eb_[2 * i + 1] = e1;
            ll += std::log(s) + mx;
        }
        return ll;
    }

    void backward(const Params& th, std::size_t N) {
        beta_[2 * (N - 1)] = 1.0;
        beta_[2 * (N - 1) + 1] = 1.0;
        for (std::size_t i = N - 1; i-- > 0;) {
            const double n0 = eb_[2 * (i + 1)] * beta_[2 * (i + 1)];
            const double n1 = eb_[2 * (i + 1) + 1] * beta_[2 * (i + 1) + 1];
            const double s = scale_[i + 1];
            beta_[2 * i] = (th.A[0][0] * n0 + th.A[0][1] * n1) / s;
            beta_[2 * i + 1] = (th.A[1][0] * n0 + th.A[1][1] * n1) / s;
        }
    }

    // gamma_ = the normalized state probabilities alpha * beta, once per M-step.
    void smooth(std::size_t N) {
        for (std::size_t i = 0; i < N; ++i) {
            const double g0 = alpha_[2 * i] * beta_[2 * i];
            const double g1 = alpha_[2 * i + 1] * beta_[2 * i + 1];
            gamma_[2 * i] = g0 / (g0 + g1);
            gamma_[2 * i + 1] = g1 / (g0 + g1);
        }
    }

    // Some state has no weight in gamma_.
    bool empty_state(std::size_t N) const {
        double s0 = 0.0, s1 = 0.0;
        for (std::size_t i = 0; i < N; ++i) {
            s0 += gamma_[2 * i];
            s1 += gamma_[2 * i + 1];
        }
        return !(s0 > 0.0) || !(s1 > 0.0);
    }

    // One M-step from gamma_ (smooth() after backward()).
    Params m_step(const Params& th, std::size_t N, double center, double floor) const {
        Params nw = th;
        const double* y = ys();
        const double* z = zs();
        const double g00 = gamma_[0], g01 = gamma_[1];
        nw.pi[0] = g00 / (g00 + g01);
        nw.pi[1] = g01 / (g00 + g01);
        double X[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
        for (std::size_t i = 0; i + 1 < N; ++i) {
            const double s = scale_[i + 1];
            const double n0 = eb_[2 * (i + 1)] * beta_[2 * (i + 1)];
            const double n1 = eb_[2 * (i + 1) + 1] * beta_[2 * (i + 1) + 1];
            for (int j = 0; j < 2; ++j) {
                X[j][0] += alpha_[2 * i + static_cast<std::size_t>(j)] * th.A[j][0] * n0 / s;
                X[j][1] += alpha_[2 * i + static_cast<std::size_t>(j)] * th.A[j][1] * n1 / s;
            }
        }
        for (int j = 0; j < 2; ++j) {
            const double den = X[j][0] + X[j][1];
            if (den > 0.0) {
                nw.A[j][0] = X[j][0] / den;
                nw.A[j][1] = X[j][1] / den;
            }
        }
        for (int k = 0; k < 2; ++k) {
            double S0 = 0.0, Sy = 0.0, Sz = 0.0, Szz = 0.0, Szy = 0.0;
            for (std::size_t i = 0; i < N; ++i) {
                const double w = gamma_[2 * i + static_cast<std::size_t>(k)];
                const double yc = y[i] - center;
                S0 += w;
                Sy += w * yc;
                if (ar1_) {
                    const double zc = z[i] - center;
                    Sz += w * zc;
                    Szz += w * zc * zc;
                    Szy += w * zc * yc;
                }
            }
            if (!(S0 > 0.0)) continue;
            double c = 0.0, phi = 0.0;
            if (ar1_) {
                const double den = S0 * Szz - Sz * Sz;
                if (den > 1e-12 * S0 * Szz) phi = (S0 * Szy - Sz * Sy) / den;
                c = (Sy - phi * Sz) / S0 + center * (1.0 - phi);
            } else {
                c = center + Sy / S0;
            }
            double ss = 0.0;
            for (std::size_t i = 0; i < N; ++i) {
                const double r = ar1_ ? y[i] - c - phi * z[i] : y[i] - c;
                ss += gamma_[2 * i + static_cast<std::size_t>(k)] * r * r;
            }
            nw.mu[k] = c;
            nw.phi[k] = phi;
            nw.var[k] = std::max(ss / S0, floor);
        }
        nw.set_logv();
        return nw;
    }

    Params initial(double floor) {
        Params th;
        const double Wd = static_cast<double>(W_);
        if (ar1_) {
            double xm = 0.0;
            for (double v : win_) xm += v;
            xm /= Wd;
            double c0 = 0.0, c1 = 0.0;
            for (std::size_t i = 0; i < W_; ++i) {
                const double d = win_[i] - xm;
                c0 += d * d;
                if (i > 0) c1 += d * (win_[i - 1] - xm);
            }
            const double rho = c0 > 0.0 ? c1 / c0 : 0.0;
            const double v = std::max(c0 / Wd * (1.0 - rho * rho), floor);
            th.phi[0] = std::clamp(rho - 0.5, -0.99, 0.99);
            th.phi[1] = std::clamp(rho + 0.5, -0.99, 0.99);
            for (int k = 0; k < 2; ++k) {
                th.mu[k] = xm * (1.0 - th.phi[k]);
                th.var[k] = v;
            }
        } else {
            // median of |x|, in scale_ (free until the first E-step)
            for (std::size_t i = 0; i < W_; ++i) scale_[i] = std::fabs(win_[i]);
            const std::size_t mid = W_ / 2;
            std::nth_element(scale_.begin(), scale_.begin() + static_cast<std::ptrdiff_t>(mid),
                             scale_.begin() + static_cast<std::ptrdiff_t>(W_));
            double med = scale_[mid];
            if (W_ % 2 == 0)
                med = 0.5 * (*std::max_element(scale_.begin(),
                                               scale_.begin() + static_cast<std::ptrdiff_t>(mid)) + med);
            bool split = false;                      // some |x| above the median
            for (double v : win_) split = split || std::fabs(v) > med;
            for (int k = 0; k < 2; ++k) {
                const auto member = [&](double v) {
                    return !split || ((std::fabs(v) <= med) == (k == 0));
                };
                double cnt = 0.0, mean = 0.0;
                for (double v : win_)
                    if (member(v)) { cnt += 1.0; mean += v; }
                mean /= cnt;
                double ss = 0.0;
                for (double v : win_)
                    if (member(v)) ss += (v - mean) * (v - mean);
                th.mu[k] = mean;
                th.var[k] = std::max(ss / cnt, floor);
            }
        }
        th.set_logv();
        return th;
    }

    void order(Params& th) const {
        const bool swap = ar1_ ? th.phi[0] > th.phi[1] : th.var[0] > th.var[1];
        if (!swap) return;
        std::swap(th.pi[0], th.pi[1]);
        std::swap(th.mu[0], th.mu[1]);
        std::swap(th.phi[0], th.phi[1]);
        std::swap(th.var[0], th.var[1]);
        std::swap(th.logv[0], th.logv[1]);
        std::swap(th.A[0][0], th.A[1][1]);
        std::swap(th.A[0][1], th.A[1][0]);
    }

    void fit() {
        load_window();
        const std::size_t N = observations();
        const double Wd = static_cast<double>(W_);
        double center = 0.0;
        for (double v : win_) center += v;
        center /= Wd;
        double ss = 0.0;
        for (double v : win_) ss += (v - center) * (v - center);
        const double floor = std::max(1e-6 * (ss / Wd), 1e-300);
        Params th;
        bool warm = fitted_;
        if (warm) {
            th = th_;
            th.pi[0] = 0.5;
            th.pi[1] = 0.5;
            for (int k = 0; k < 2; ++k) th.var[k] = std::max(th.var[k], floor);
            th.set_logv();
            // two equal states carry nothing to start from
            warm = !(th.mu[0] == th.mu[1] && th.phi[0] == th.phi[1] && th.var[0] == th.var[1]);
        }
        if (!warm) th = initial(floor);
        Params good = th;                 // the last parameters with a finite log-likelihood
        double ll_prev = -std::numeric_limits<double>::infinity();
        for (int it = 0; it < max_iter_; ++it) {
            double ll = forward(th, N);
            bool smoothed = false;
            if (it == 0 && warm) {
                if (std::isfinite(ll)) {
                    backward(th, N);
                    smooth(N);
                    smoothed = true;
                }
                if (!smoothed || empty_state(N)) {
                    th = initial(floor);
                    good = th;
                    smoothed = false;
                    ll = forward(th, N);
                }
            }
            if (!std::isfinite(ll)) {
                th = good;
                break;
            }
            if (it > 0 && !(ll - ll_prev >= tol_)) break;
            good = th;
            if (!smoothed) {
                backward(th, N);
                smooth(N);
            }
            th = m_step(th, N, center, floor);
            ll_prev = ll;
        }
        order(th);
        th_ = th;
        fitted_ = true;
        if (std::isnan(forward(th_, N))) {
            p_[0] = 0.5;
            p_[1] = 0.5;
        } else {
            p_[0] = alpha_[2 * (N - 1)];
            p_[1] = alpha_[2 * (N - 1) + 1];
        }
    }

    void filter_step(double x, double prev) {
        const double l0 = log_b(th_, 0, x, prev), l1 = log_b(th_, 1, x, prev);
        const double mx = l0 > l1 ? l0 : l1;
        const double q0 = p_[0] * th_.A[0][0] + p_[1] * th_.A[1][0];
        const double q1 = p_[0] * th_.A[0][1] + p_[1] * th_.A[1][1];
        const double a0 = q0 * std::exp(l0 - mx), a1 = q1 * std::exp(l1 - mx);
        const double s = a0 + a1;
        if (s > 0.0 && std::isfinite(s)) {
            p_[0] = a0 / s;
            p_[1] = a1 / s;
        } else {
            const double t = q0 + q1;
            p_[0] = q0 / t;
            p_[1] = q1 / t;
        }
    }

    ResultTuple outputs(double x) const {
        const double q0 = p_[0] * th_.A[0][0] + p_[1] * th_.A[1][0];
        const double q1 = p_[0] * th_.A[0][1] + p_[1] * th_.A[1][1];
        const double m0 = ar1_ ? th_.mu[0] + th_.phi[0] * x : th_.mu[0];
        const double m1 = ar1_ ? th_.mu[1] + th_.phi[1] * x : th_.mu[1];
        const double mean = q0 * m0 + q1 * m1;
        const double dm = m0 - m1;
        const double var = q0 * th_.var[0] + q1 * th_.var[1] + q0 * q1 * dm * dm;
        return std::make_tuple(p_[1], mean, std::sqrt(var));
    }

    const std::size_t W_;
    const int refit_every_;
    const bool ar1_;
    const int max_iter_;
    const double tol_;
    std::vector<double> buf_, win_, eb_, alpha_, beta_, scale_, gamma_;
    std::size_t head_ = 0, count_ = 0;
    int since_ = 0;
    bool fitted_ = false;
    double prev_ = 0.0;
    double p_[2] = {0.5, 0.5};
    Params th_;
};

}  // namespace screamer

#endif  // SCREAMER_ROLLING_HMM_H
