#ifndef SCREAMER_ROLLING_PERMUTATION_ENTROPY_H
#define SCREAMER_ROLLING_PERMUTATION_ENTROPY_H

// RollingPermutationEntropy: the Bandt-Pompe (2002) permutation entropy of a
// series over a trailing window. At each step with (order - 1) * delay earlier
// values, the values x_{t-(order-1) delay}, ..., x_{t-delay}, x_t form an
// ordinal pattern (their ranks; equal values rank by order of appearance, the
// earlier one lower). The output is the Shannon entropy of the patterns that
// end at the last window_size steps, divided by log(order!), so it lies in
// [0, 1]: 0 when one pattern repeats (a monotone or constant series), near 1
// for white noise.
//
// A pattern is stored as its Lehmer code, sum_i c_i (order - 1 - i)!, with c_i
// the number of later values strictly below value i, which is a bijection
// with the tie-broken ranks. The counts of the order! codes and
// S = sum_c n_c log n_c are updated as patterns enter and leave, so a step
// costs O(order^2); S is recomputed from the counts every window_size steps,
// which bounds the rounding the updates accumulate. Entropy =
// (W log W - S) / (W log(order!)), W the patterns in the window, and exactly 0
// when one distinct pattern fills the window.
//
// start_policy: "strict" gives NaN until window_size patterns, "expanding"
// uses the patterns seen so far, "zero" behaves as if the input were preceded
// by zeros (a full history and window_size all-tie patterns). nan_policy:
// ignore (a NaN is not stored and forms no pattern).

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

class RollingPermutationEntropy : public ScreamerBase {
public:
    RollingPermutationEntropy(int window_size = 100, int order = 3, int delay = 1,
                              const std::string& start_policy = "strict")
        : W_(check(window_size, order, delay)), m_(order), tau_(delay),
          span_(static_cast<std::size_t>((order - 1) * delay)),
          policy_(detail::parse_start_policy(start_policy))
    {
        std::size_t f = 1;
        weights_.assign(static_cast<std::size_t>(m_), 1);
        for (int i = m_ - 1; i >= 0; --i) {          // weights_[i] = (m - 1 - i)!
            weights_[static_cast<std::size_t>(i)] = f;
            f *= static_cast<std::size_t>(m_ - i);
        }
        n_codes_ = f;                                 // order!
        log_norm_ = std::log(static_cast<double>(n_codes_));
        nlogn_.assign(W_ + 1, 0.0);
        for (std::size_t k = 2; k <= W_; ++k)
            nlogn_[k] = static_cast<double>(k) * std::log(static_cast<double>(k));
        values_.assign(span_ + 1, 0.0);
        codes_.assign(W_, 0);
        counts_.assign(n_codes_, 0);
        pattern_.assign(static_cast<std::size_t>(m_), 0.0);
        reset();
    }

    void reset() override {
        std::fill(values_.begin(), values_.end(), 0.0);
        std::fill(codes_.begin(), codes_.end(), 0);
        std::fill(counts_.begin(), counts_.end(), 0);
        vhead_ = 0;
        chead_ = 0;
        since_exact_ = 0;
        if (policy_ == detail::StartPolicy::Zero) {
            nvalues_ = span_ + 1;
            ncodes_ = W_;
            counts_[0] = W_;                          // W all-tie patterns, code 0
            S_ = nlogn_[W_];
            distinct_ = 1;
        } else {
            nvalues_ = 0;
            ncodes_ = 0;
            S_ = 0.0;
            distinct_ = 0;
        }
    }

    double process_scalar(double x) override {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        if (isnan2(x)) return nan;
        const std::size_t V = values_.size();
        values_[vhead_] = x;
        if (++vhead_ == V) vhead_ = 0;
        if (nvalues_ < V) ++nvalues_;
        if (nvalues_ < V) return nan;                 // no pattern yet
        add_pattern(code());
        if (policy_ == detail::StartPolicy::Strict && ncodes_ < W_) return nan;
        return entropy();
    }

private:
    static std::size_t check(int window_size, int order, int delay) {
        if (order < 2) throw std::invalid_argument("order must be 2 or more.");
        if (order > 7) throw std::invalid_argument("order must be 7 or less.");
        if (delay < 1) throw std::invalid_argument("delay must be 1 or more.");
        if (window_size < 2) throw std::invalid_argument("Window size must be 2 or more.");
        return static_cast<std::size_t>(window_size);
    }

    // The Lehmer code of the newest pattern.
    std::size_t code() {
        const std::size_t V = values_.size();
        const std::size_t newest = (vhead_ + V - 1) % V;
        const std::size_t m = static_cast<std::size_t>(m_);
        const std::size_t tau = static_cast<std::size_t>(tau_);
        for (std::size_t k = 0; k < m; ++k)          // oldest first
            pattern_[k] = values_[(newest + V - (m - 1 - k) * tau) % V];
        std::size_t c = 0;
        for (std::size_t i = 0; i + 1 < m; ++i) {
            std::size_t below = 0;
            for (std::size_t j = i + 1; j < m; ++j) below += pattern_[j] < pattern_[i] ? 1 : 0;
            c += below * weights_[i];
        }
        return c;
    }

    void add_pattern(std::size_t c) {
        if (ncodes_ == W_) {
            const std::size_t old = codes_[chead_];
            if (old == c) {                           // counts unchanged
                if (++chead_ == W_) chead_ = 0;
                tick();
                return;
            }
            const std::size_t n = counts_[old];
            S_ += nlogn_[n - 1] - nlogn_[n];
            counts_[old] = n - 1;
            if (n == 1) --distinct_;
        } else {
            ++ncodes_;
        }
        codes_[chead_] = c;
        if (++chead_ == W_) chead_ = 0;
        const std::size_t n = counts_[c];
        S_ += nlogn_[n + 1] - nlogn_[n];
        counts_[c] = n + 1;
        if (n == 0) ++distinct_;
        tick();
    }

    // Every W_ patterns, S from the counts (bounds the rounding of the updates).
    void tick() {
        if (++since_exact_ < W_) return;
        since_exact_ = 0;
        double s = 0.0;
        for (std::size_t n : counts_) s += nlogn_[n];
        S_ = s;
    }

    double entropy() const {
        if (distinct_ <= 1) return 0.0;
        const double w = static_cast<double>(ncodes_);
        const double h = (nlogn_[ncodes_] - S_) / (w * log_norm_);
        return h < 0.0 ? 0.0 : (h > 1.0 ? 1.0 : h);
    }

    const std::size_t W_;
    const int m_;
    const int tau_;
    const std::size_t span_;
    const detail::StartPolicy policy_;
    std::size_t n_codes_ = 1;
    double log_norm_ = 0.0;
    std::vector<std::size_t> weights_;
    std::vector<double> nlogn_;
    std::vector<double> values_;
    std::vector<std::size_t> codes_;
    std::vector<std::size_t> counts_;
    std::vector<double> pattern_;
    std::size_t vhead_ = 0, chead_ = 0, nvalues_ = 0, ncodes_ = 0, since_exact_ = 0, distinct_ = 0;
    double S_ = 0.0;
};

}  // namespace screamer

#endif  // SCREAMER_ROLLING_PERMUTATION_ENTROPY_H
