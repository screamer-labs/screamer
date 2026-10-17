#ifndef SCREAMER_ROLLING_VARIANCE_RATIO_H
#define SCREAMER_ROLLING_VARIANCE_RATIO_H

// RollingVarianceRatio: the Lo-MacKinlay (1988) variance ratio of a window
// of returns r_1..r_n, with q = lag:
//     mu        = mean(r)
//     sigma_a^2 = sum (r_i - mu)^2 / (n - 1)
//     sigma_c^2 = sum_{k=q..n} (r_k + ... + r_{k-q+1} - q mu)^2 / m,
//                 m = q (n - q + 1) (1 - q / n)
//     ratio     = sigma_c^2 / sigma_a^2
// the overlapping q-step sums with the bias-corrected variances. Near 1 for
// uncorrelated returns, above 1 when returns trend, below 1 when they revert.
// output "zscore" is the heteroskedasticity-robust statistic
//     z*(q) = sqrt(n) (ratio - 1) / sqrt(theta),
//     theta = sum_{j=1..q-1} (2 (q - j) / q)^2 delta_j,
//     delta_j = n sum_{i>j} (r_i - mu)^2 (r_{i-j} - mu)^2 / (sum_i (r_i - mu)^2)^2.
// NaN when the window has no dispersion.
//
// start_policy: "strict" gives NaN until the window holds window_size
// returns, "expanding" uses the returns seen so far once there are more than
// 2 lag, "zero" starts from a window of zeros. nan_policy: ignore.
// O(window_size * lag) per step, from a ring of the window and one scratch
// array allocated at construction.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "screamer/common/base.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/start_policy.h"

namespace screamer {

class RollingVarianceRatio : public ScreamerBase {
public:
    static bool parse_output(const std::string& s) {
        if (s == "ratio") return false;
        if (s == "zscore") return true;
        throw std::invalid_argument("output must be \"ratio\" or \"zscore\".");
    }

    RollingVarianceRatio(int window_size = 256, int lag = 2,
                         const std::string& output = "ratio",
                         const std::string& start_policy = "strict")
        : w_(window_size), q_(lag), zscore_(parse_output(output)),
          policy_(detail::parse_start_policy(start_policy))
    {
        if (lag < 2) throw std::invalid_argument("lag must be 2 or more.");
        if (window_size <= 2 * lag)
            throw std::invalid_argument("window_size must be more than 2 lag.");
        ring_.assign(static_cast<std::size_t>(w_), 0.0);
        dev_.assign(static_cast<std::size_t>(w_), 0.0);
        reset();
    }

    void reset() override {
        std::fill(ring_.begin(), ring_.end(), 0.0);
        head_ = 0;
        count_ = policy_ == detail::StartPolicy::Zero ? static_cast<std::size_t>(w_) : 0;
    }

    double process_scalar(double x) override {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        if (isnan2(x)) return nan;
        const std::size_t W = ring_.size();
        ring_[head_] = x;
        if (++head_ == W) head_ = 0;
        if (count_ < W) ++count_;
        if (policy_ == detail::StartPolicy::Strict && count_ < W) return nan;
        if (count_ <= 2 * static_cast<std::size_t>(q_)) return nan;
        return value();
    }

private:
    // The statistic of the count_ newest values, oldest first in dev_.
    double value() {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        const std::size_t W = ring_.size(), n = count_;
        const std::size_t start = (head_ + W - n) % W;   // the oldest value in the window
        double sum = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double v = ring_[(start + i) % W];
            dev_[i] = v;
            sum += v;
        }
        const double nd = static_cast<double>(n);
        const double mu = sum / nd;
        const std::size_t q = static_cast<std::size_t>(q_);
        const double qd = static_cast<double>(q_);
        // overlapping q-sums of the returns, about q mu
        double s = 0.0;
        for (std::size_t i = 0; i < q; ++i) s += dev_[i];
        double acc = (s - qd * mu) * (s - qd * mu);
        for (std::size_t k = q; k < n; ++k) {
            s += dev_[k] - dev_[k - q];
            const double d = s - qd * mu;
            acc += d * d;
        }
        double s2 = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            dev_[i] -= mu;
            s2 += dev_[i] * dev_[i];
        }
        if (!(s2 > 0.0)) return nan;
        const double var_a = s2 / (nd - 1.0);
        const double m = qd * (nd - qd + 1.0) * (1.0 - qd / nd);
        const double ratio = acc / m / var_a;
        if (!zscore_) return ratio;
        double theta = 0.0;
        for (std::size_t j = 1; j < q; ++j) {
            double d = 0.0;
            for (std::size_t i = j; i < n; ++i) d += dev_[i] * dev_[i] * dev_[i - j] * dev_[i - j];
            const double wj = 2.0 * static_cast<double>(q - j) / qd;
            theta += wj * wj * nd * d / (s2 * s2);
        }
        return std::sqrt(nd) * (ratio - 1.0) / std::sqrt(theta);
    }

    const int w_;
    const int q_;
    const bool zscore_;
    const detail::StartPolicy policy_;
    std::vector<double> ring_, dev_;
    std::size_t head_ = 0, count_ = 0;
};

}  // namespace screamer

#endif  // SCREAMER_ROLLING_VARIANCE_RATIO_H
