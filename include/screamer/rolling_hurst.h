#ifndef SCREAMER_ROLLING_HURST_H
#define SCREAMER_ROLLING_HURST_H

// RollingHurst: rolling-window Hurst exponent estimator.
//
// Method 'rs' (default): Anis-Lloyd-corrected rescaled-range analysis,
// matching the form
//
//     rsal[k] = R/S[n_k] - ers(n_k) + sqrt(0.5 * pi * n_k)
//     H = slope of log(rsal) on log(n)
//
// at scales n_k = min_scale, 2*min_scale, 4*min_scale, ... up to W/2.
//
// References:
//   Anis & Lloyd (1976); Peters (1994); Weron (2002);
//   "Estimating the Hurst Exponent..." (arXiv:1805.08931).
//
// 1 -> 1. Output is the estimated Hurst exponent (~0.5 for white noise,
// >0.5 persistent, <0.5 anti-persistent). First valid at sample W-1.
// O(W log W) per step; the ers(n_k) table is precomputed.

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include "screamer/common/base.h"

namespace screamer {

class RollingHurst : public ScreamerBase {
public:
    explicit RollingHurst(int window_size,
                          int min_scale = 4,
                          const std::string& method = "rs")
        : window_size_(window_size), min_scale_(min_scale), method_(method)
    {
        if (method_ != "rs") {
            throw std::invalid_argument(
                "RollingHurst: only method='rs' is supported (Anis-Lloyd corrected R/S).");
        }
        if (min_scale_ < 4) {
            throw std::invalid_argument("RollingHurst: min_scale must be >= 4.");
        }
        if (window_size_ < 4 * min_scale_) {
            throw std::invalid_argument(
                "RollingHurst: window_size must be >= 4 * min_scale "
                "(need at least 3 scales for a stable regression).");
        }

        for (int n = min_scale_; n <= window_size_ / 2; n *= 2) {
            scales_.push_back(n);
            ers_.push_back(anis_lloyd_ers(n));
            log_n_.push_back(std::log(double(n)));
            asym_.push_back(std::sqrt(0.5 * M_PI * double(n)));
        }
        if (scales_.size() < 3) {
            throw std::invalid_argument(
                "RollingHurst: need at least 3 scales; increase window_size or decrease min_scale.");
        }

        // Precompute the regression denominator: K * sum(x^2) - (sum x)^2.
        const int K = (int)log_n_.size();
        double sx = 0.0, sxx = 0.0;
        for (int k = 0; k < K; ++k) { sx += log_n_[k]; sxx += log_n_[k] * log_n_[k]; }
        sum_x_ = sx;
        denom_ = K * sxx - sx * sx;

        buffer_.resize(window_size_);
        block_.resize(window_size_ / 2);  // largest scale
        reset();
    }

    void reset() override {
        std::fill(buffer_.begin(), buffer_.end(), 0.0);
        index_ = 0;
        size_ = 0;
    }

    double process_scalar(double x) override {
        buffer_[index_] = x;
        index_++;
        if (index_ == window_size_) index_ = 0;
        if (size_ < window_size_) {
            size_++;
            if (size_ < window_size_) {
                return std::numeric_limits<double>::quiet_NaN();
            }
        }
        return compute_hurst();
    }

private:
    static double anis_lloyd_ers(int n) {
        double sum = 0.0;
        for (int i = 1; i < n; ++i) {
            sum += std::sqrt(double(n - i) / double(i));
        }
        if (n <= 340) {
            sum *= std::tgamma((n - 1) / 2.0) /
                   (std::sqrt(M_PI) * std::tgamma(n / 2.0));
        } else {
            sum *= 1.0 / std::sqrt(M_PI * double(n) / 2.0);
        }
        return sum;
    }

    // Average R/S across non-overlapping blocks of length n inside the
    // window. The buffer is a circular structure; the oldest sample
    // lives at `index_` (next write slot).
    // Returns NaN if any block has zero variance.
    double mean_rescaled_range(int n) {
        const int num_patterns = window_size_ / n;
        double total = 0.0;
        for (int p = 0; p < num_patterns; ++p) {
            // Pull block p into a contiguous buffer (block_) so we can
            // walk it twice.
            double mean = 0.0;
            for (int j = 0; j < n; ++j) {
                const double v = buffer_[(index_ + p * n + j) % window_size_];
                block_[j] = v;
                mean += v;
            }
            mean /= double(n);

            // Std (population: divisor n, matching the reference code).
            double ss = 0.0;
            for (int j = 0; j < n; ++j) {
                const double d = block_[j] - mean;
                block_[j] = d;            // demean in place for cumsum
                ss += d * d;
            }
            const double std_pop = std::sqrt(ss / double(n));
            if (std_pop == 0.0) {
                return std::numeric_limits<double>::quiet_NaN();
            }

            // Cumulative sum of demeaned block; track running min/max.
            double cum = 0.0;
            double mn = 0.0, mx = 0.0;
            for (int j = 0; j < n; ++j) {
                cum += block_[j];
                if (cum < mn) mn = cum;
                if (cum > mx) mx = cum;
            }
            const double range = mx - mn;
            total += range / std_pop;
        }
        return total / double(num_patterns);
    }

    double compute_hurst() {
        const int K = (int)scales_.size();
        // y_k = log(R/S(n_k) - ers(n_k) + sqrt(0.5 * pi * n_k))
        // Slope of y vs log_n via closed-form OLS.
        double sy = 0.0, sxy = 0.0;
        for (int k = 0; k < K; ++k) {
            const double rs = mean_rescaled_range(scales_[k]);
            if (!std::isfinite(rs)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            const double rsal = rs - ers_[k] + asym_[k];
            if (!(rsal > 0.0)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            const double y = std::log(rsal);
            sy += y;
            sxy += log_n_[k] * y;
        }
        const double slope = (double(K) * sxy - sum_x_ * sy) / denom_;
        return slope;
    }

    const int window_size_;
    const int min_scale_;
    const std::string method_;
    std::vector<int> scales_;
    std::vector<double> ers_;
    std::vector<double> log_n_;
    std::vector<double> asym_;
    double sum_x_ = 0.0;
    double denom_ = 0.0;
    std::vector<double> buffer_;
    std::vector<double> block_;
    int index_ = 0;
    int size_ = 0;
};

}  // namespace screamer

#endif
