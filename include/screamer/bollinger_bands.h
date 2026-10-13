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
// Window sums about an origin near the mean (detail::ShiftedWindowSums), so
// the bands stay accurate when the price is large next to its spread. O(1)
// per step.

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include "screamer/common/functor_base.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/moment_formulas.h"
#include "screamer/detail/shifted_window_sums.h"
#include "screamer/detail/start_policy.h"
#include <cstddef>

namespace screamer {

class BollingerBands : public StepperFunctorBase<BollingerBands, 1, 3> {
    using Sums = detail::ShiftedWindowSums<2>;

public:
    BollingerBands(int window_size,
                   double num_std = 2.0,
                   const std::string& start_policy = "strict")
        : num_std_(num_std),
          start_policy_(detail::parse_start_policy(start_policy)),
          sums_(window_size >= 2 ? window_size : 2, start_policy_)
    {
        if (window_size < 2) {
            throw std::invalid_argument("Window size must be 2 or more.");
        }
        if (!(num_std >= 0.0 && std::isfinite(num_std))) {  // also rejects NaN and inf
            throw std::invalid_argument("num_std must be a non-negative finite number.");
        }
        reset();
    }

    void reset() override { sums_.reset(); }

    // One event on the window's state, for both the batch column loop and
    // call() (StepperFunctorBase). Each product is its own statement (see
    // shifted_window_sums.h).
    struct Stepper {
        Sums::Cursor c;
        bool strict;
        double num_std;

        inline ResultTuple call(const InputArray& inputs) {
            const double nan = std::numeric_limits<double>::quiet_NaN();
            const double x = inputs[0];
            if (isnan2(x)) {
                return std::make_tuple(nan, nan, nan);
            }
            c.push(x);
            const std::size_t n_ = c.st.count;
            if ((strict && n_ < c.cap) || n_ < 2) {
                return std::make_tuple(nan, nan, nan);
            }
            const double nd = static_cast<double>(n_);
            const double sum_d  = c.st.s[0];
            const double mean = c.st.origin + sum_d / nd;
            // Unbiased sample variance (n-1 denominator). Matches pandas
            // Series.rolling(w).std() which defaults to ddof=1.
            const double var = detail::var_from_sums(sum_d, c.st.s[1], nd);
            const double std_val = std::sqrt(std::max(var, 0.0));
            const double half_width = num_std * std_val;
            const double lower = mean - half_width;
            const double upper = mean + half_width;
            return std::make_tuple(lower, mean, upper);
        }
    };
    Stepper stepper() {
        return Stepper{sums_.cursor(), start_policy_ == detail::StartPolicy::Strict, num_std_};
    }
    void store(const Stepper& k) { sums_.set_state(k.c.st); }

private:
    const double num_std_;
    const detail::StartPolicy start_policy_;
    Sums sums_;
};

}  // namespace screamer

#endif  // SCREAMER_BOLLINGER_BANDS_H
