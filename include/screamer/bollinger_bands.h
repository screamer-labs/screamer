#ifndef SCREAMER_BOLLINGER_BANDS_H
#define SCREAMER_BOLLINGER_BANDS_H

// BollingerBands: rolling Bollinger bands of a single price stream.
// First real consumer of FunctorBase<_, 1, 3>.
//
//     mid_w[t]   = mean of x over the last `window_size` samples
//     std_w[t]   = sample std-dev of x over the same window
//     lower_w[t] = mid_w[t] - num_std * std_w[t]
//     upper_w[t] = mid_w[t] + num_std * std_w[t]
//
// Returns the triple (lower, mid, upper) per step.
//
// Two detail::RollingSum buffers under the hood (Sx, Sxx), O(1) per step.

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include "screamer/common/functor_base.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/rolling_sum.h"
#include "screamer/detail/start_policy.h"

namespace screamer {

class BollingerBands : public FunctorBase<BollingerBands, 1, 3> {
public:
    BollingerBands(int window_size,
                   double num_std = 2.0,
                   const std::string& start_policy = "strict")
        : window_size_(window_size),
          num_std_(num_std),
          start_policy_(detail::parse_start_policy(start_policy)),
          sum_x_buffer(window_size, "expanding"),
          sum_xx_buffer(window_size, "expanding")
    {
        if (window_size < 2) {
            throw std::invalid_argument("Window size must be 2 or more.");
        }
        if (!(num_std >= 0.0)) {  // also rejects NaN
            throw std::invalid_argument("num_std must be a non-negative finite number.");
        }
        reset();
    }

    void reset() override {
        sum_x_buffer.reset();
        sum_xx_buffer.reset();
        n_ = (start_policy_ == detail::StartPolicy::Zero) ? window_size_ : 0;
    }

    ResultTuple call(const InputArray& inputs) override {
        const double x = inputs[0];

        if (isnan2(x)) {
            const double nan = std::numeric_limits<double>::quiet_NaN();
            return std::make_tuple(nan, nan, nan);
        }

        const double sum_x  = sum_x_buffer .append(x);
        const double sum_xx = sum_xx_buffer.append(x * x);

        if ((n_ < window_size_) && (start_policy_ != detail::StartPolicy::Zero)) {
            n_++;
        }

        const double nan = std::numeric_limits<double>::quiet_NaN();
        if (start_policy_ == detail::StartPolicy::Strict && n_ < window_size_) {
            return std::make_tuple(nan, nan, nan);
        }
        if (n_ < 2) {
            return std::make_tuple(nan, nan, nan);
        }

        const double nd = static_cast<double>(n_);
        const double mean = sum_x / nd;
        // Unbiased sample variance (n-1 denominator). Matches pandas
        // Series.rolling(w).std() which defaults to ddof=1.
        const double var = (nd * sum_xx - sum_x * sum_x) / (nd * (nd - 1.0));
        const double std_val = std::sqrt(std::max(var, 0.0));

        const double lower = mean - num_std_ * std_val;
        const double upper = mean + num_std_ * std_val;
        return std::make_tuple(lower, mean, upper);
    }

private:
    const int window_size_;
    const double num_std_;
    const detail::StartPolicy start_policy_;
    detail::RollingSum sum_x_buffer;
    detail::RollingSum sum_xx_buffer;
    size_t n_ = 0;
};

}  // namespace screamer

#endif  // SCREAMER_BOLLINGER_BANDS_H
