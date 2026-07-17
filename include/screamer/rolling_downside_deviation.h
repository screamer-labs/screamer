#ifndef SCREAMER_ROLLING_DOWNSIDE_DEVIATION_H
#define SCREAMER_ROLLING_DOWNSIDE_DEVIATION_H

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include "screamer/common/base.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/rolling_sum.h"
#include "screamer/detail/start_policy.h"

namespace screamer {

    // RollingDownsideDeviation: the trailing-window downside semideviation of a
    // return series, the root-mean-square of the shortfalls below a minimum
    // acceptable return `mar`,
    //     sqrt( mean( min(x - mar, 0)^2 ) ).
    // Only returns below `mar` contribute; returns at or above it count as zero.
    // This is the denominator of the Sortino ratio, exposed on its own as a
    // one-sided risk measure. 1 -> 1; composes a single detail::RollingSum of the
    // squared shortfall. nan_policy: ignore (a NaN input yields NaN and leaves the
    // window untouched).
    class RollingDownsideDeviation : public ScreamerBase {
    public:
        RollingDownsideDeviation(int window_size, double mar = 0.0,
                                 const std::string& start_policy = "strict")
            : window_size_(window_size),
              mar_(mar),
              start_policy_(detail::parse_start_policy(start_policy)),
              sum_d2_(window_size, start_policy)
        {
            if (window_size_ < 2) {
                throw std::invalid_argument("Window size must be 2 or more.");
            }
            reset();
        }

        void reset() override {
            sum_d2_.reset();
            n_ = (start_policy_ != detail::StartPolicy::Zero) ? 0 : window_size_;
        }

        double process_scalar(double x) override {
            if (isnan2(x)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            if ((n_ < window_size_) && (start_policy_ != detail::StartPolicy::Zero)) {
                n_++;
            }
            double shortfall = x - mar_;
            if (shortfall > 0.0) shortfall = 0.0;          // downside only
            const double sum = sum_d2_.append(shortfall * shortfall);
            return std::sqrt(sum / n_);
        }

    private:
        const int window_size_;
        const double mar_;
        const detail::StartPolicy start_policy_;
        size_t n_ = 0;
        detail::RollingSum sum_d2_;
    };

} // namespace screamer

#endif // SCREAMER_ROLLING_DOWNSIDE_DEVIATION_H
