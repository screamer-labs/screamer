#ifndef SCREAMER_BAYESIAN_CHANGEPOINT_H
#define SCREAMER_BAYESIAN_CHANGEPOINT_H

// BayesianChangepoint: online Bayesian changepoint detection (Adams and
// MacKay 2007). The run length r is the number of steps since the last
// changepoint (0 on the step of a change); the value of a step belongs to the
// run it starts or continues, so run r has seen r + 1 values. Each run has a
// normal-gamma posterior from the prior (mean 0, kappa 1, alpha 1, beta
// prior_var): kappa_r = r + 2, alpha_r = 1.5 + r / 2, and running mu_r,
// beta_r. With hazard H = 1 / expected_run, a value x moves the run-length
// distribution P to
//     P'(r + 1) ~ P(r) t_r(x) (1 - H),   P'(0) ~ H t_0(x),
// normalized, where t_r is run r's Student-t predictive and t_0 the prior
// predictive. Runs longer than max_run are dropped. 1 -> 3:
//     [run_length, pred_mean, pred_std]
// run_length = sum r P(r); pred_mean and pred_std are the mean and standard
// deviation of the predictive of the next value mixed over the current run
// lengths, given no change at the next step: weights P(r), component means
// mu_r and variances beta_r (kappa_r + 1) / (kappa_r (alpha_r - 1)).
//
// The distribution is kept as log-probabilities and normalized every step
// against its largest term, so no run's mass underflows to zero and the
// normalizer is at least 1 whatever the value. O(min(t, max_run)) per step
// (two logs and one exp per run); every array is allocated at construction.
// nan_policy: ignore.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>
#include "screamer/common/finite_settings.h"
#include "screamer/common/float_info.h"
#include "screamer/common/functor_base.h"

namespace screamer {

class BayesianChangepoint : public FunctorBase<BayesianChangepoint, 1, 3> {
public:
    // The batch column loop calls call() out of line, the same code as a
    // per-event call: inlined, clang contracts the loops' multiply-adds
    // differently and the last bits of batch and per-event output differ.
    // A step costs O(max_run) logs and exps, so the call itself is free.
    static constexpr bool kInlineColumnCall = false;

    BayesianChangepoint(double expected_run = 250.0, int max_run = 1000, double prior_var = 1.0)
        : prior_var_(check(expected_run, max_run, prior_var)),
          log_h_(-std::log(expected_run)),
          log_1mh_(std::log1p(-1.0 / expected_run)),
          R_(static_cast<std::size_t>(max_run) + 1),
          prob_(R_, 0.0), logp_(R_, 0.0), next_(R_, 0.0), mu_(R_, 0.0), beta_(R_, 0.0),
          c_(R_, 0.0), h_(R_, 0.0), nu_(R_, 0.0), g_(R_, 0.0), v_(R_, 0.0)
    {
        constexpr double kPi = 3.14159265358979323846;
        for (std::size_t r = 0; r < R_; ++r) {
            const double kappa = static_cast<double>(r) + 2.0;
            const double alpha = 1.5 + 0.5 * static_cast<double>(r);
            nu_[r] = 2.0 * alpha;
            h_[r] = alpha + 0.5;                                  // (nu + 1) / 2
            c_[r] = std::lgamma(alpha + 0.5) - std::lgamma(alpha) - 0.5 * std::log(nu_[r] * kPi);
            g_[r] = (kappa + 1.0) / (alpha * kappa);              // scale^2 = beta * g
            v_[r] = (kappa + 1.0) / (kappa * (alpha - 1.0));      // variance = beta * v
        }
        // prior predictive: nu 2, location 0, scale^2 s0 = 2 prior_var
        s0_ = 2.0 * prior_var_;
        c0_ = std::lgamma(1.5) - 0.5 * std::log(2.0 * kPi) - 0.5 * std::log(s0_) + log_h_;
    }

    void reset() override { len_ = 0; }

    ResultTuple call(const InputArray& in) override {
        const double x = in[0];
        if (isnan2(x)) {
            const double nan = std::numeric_limits<double>::quiet_NaN();
            return std::make_tuple(nan, nan, nan);
        }
        if (len_ == 0) {
            prob_[0] = 1.0;
            logp_[0] = 0.0;
            len_ = 1;
        } else {
            step(x);
        }
        mu_[0] = 0.5 * x;                                         // the run that starts at x
        beta_[0] = prior_var_ + 0.25 * x * x;
        return outputs();
    }

private:
    static double check(double expected_run, int max_run, double prior_var) {
        require_finite("expected_run", expected_run);
        require_finite("prior_var", prior_var);
        if (!(expected_run >= 1.0)) throw std::invalid_argument("expected_run must be 1 or more.");
        if (max_run < 1) throw std::invalid_argument("max_run must be 1 or more.");
        if (!(prior_var > 0.0)) throw std::invalid_argument("prior_var must be positive.");
        return prior_var;
    }

    // One value after the first. next_ first holds the unnormalized
    // log-probabilities: the changepoint at 0 and run r grown to r + 1.
    void step(double x) {
        const std::size_t n = std::min(len_ + 1, R_);
        next_[0] = c0_ - 1.5 * std::log1p(x * x / (2.0 * s0_));   // log H + log t_0(x)
        double m = next_[0];
        for (std::size_t r = 0; r + 1 < n; ++r) {
            const double s2 = beta_[r] * g_[r];
            const double d = x - mu_[r];
            const double a = logp_[r] + log_1mh_ + c_[r] - 0.5 * std::log(s2)
                             - h_[r] * std::log1p(d * d / (nu_[r] * s2));
            next_[r + 1] = a;
            if (a > m) m = a;
        }
        double total = 0.0;
        for (std::size_t r = 0; r < n; ++r) {
            next_[r] -= m;
            prob_[r] = std::exp(next_[r]);
            total += prob_[r];
        }
        // total >= 1: the largest term is exp(0).
        const double inv = 1.0 / total;
        const double log_total = std::log(total);
        for (std::size_t r = 0; r < n; ++r) {
            prob_[r] *= inv;
            logp_[r] = next_[r] - log_total;
        }
        // grown runs' parameters, index r + 1 from r, so downwards (n >= 2)
        for (std::size_t r = n - 1; r >= 1; --r) {
            const double kappa = static_cast<double>(r) + 1.0;    // kappa_{r - 1}
            const double d = x - mu_[r - 1];
            mu_[r] = (kappa * mu_[r - 1] + x) / (kappa + 1.0);
            beta_[r] = beta_[r - 1] + kappa * d * d / (2.0 * (kappa + 1.0));
        }
        len_ = n;
    }

    ResultTuple outputs() const {
        double rl = 0.0, mean = 0.0, second = 0.0;
        for (std::size_t r = 0; r < len_; ++r) {
            const double p = prob_[r];
            rl += p * static_cast<double>(r);
            mean += p * mu_[r];
            second += p * (beta_[r] * v_[r] + mu_[r] * mu_[r]);
        }
        const double var = second - mean * mean;
        return std::make_tuple(rl, mean, std::sqrt(var < 0.0 ? 0.0 : var));   // NaN stays NaN
    }

    const double prior_var_;
    const double log_h_;     // log H
    const double log_1mh_;   // log (1 - H); -inf when expected_run is 1
    const std::size_t R_;    // max_run + 1 run lengths, 0 .. max_run
    double s0_ = 0.0, c0_ = 0.0;
    std::vector<double> prob_, logp_, next_, mu_, beta_;
    std::vector<double> c_, h_, nu_, g_, v_;                     // per run-length constants
    std::size_t len_ = 0;
};

}  // namespace screamer

#endif  // SCREAMER_BAYESIAN_CHANGEPOINT_H
