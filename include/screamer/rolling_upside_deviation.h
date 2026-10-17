#ifndef SCREAMER_ROLLING_UPSIDE_DEVIATION_H
#define SCREAMER_ROLLING_UPSIDE_DEVIATION_H

#include "screamer/common/finite_settings.h"
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include "screamer/common/base.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/rolling_sum.h"
#include "screamer/detail/start_policy.h"

namespace screamer {

    // RollingUpsideDeviation: the trailing-window upside semideviation of a
    // return series, the root-mean-square of the excesses above a minimum
    // acceptable return `mar`,
    //     sqrt( mean( max(x - mar, 0)^2 ) ).
    // The mirror of RollingDownsideDeviation: only returns above `mar`
    // contribute. 1 -> 1; one detail::RollingSum of the squared excess.
    // nan_policy: ignore (a NaN input yields NaN and leaves the window
    // untouched).
    class RollingUpsideDeviation : public ScreamerBase {
    public:
        RollingUpsideDeviation(int window_size, double mar = 0.0,
                               const std::string& start_policy = "strict")
            : window_size_(window_size),
              mar_(mar),
              start_policy_(detail::parse_start_policy(start_policy)),
              sum_e2_(window_size >= 2 ? static_cast<std::size_t>(window_size) : 2, start_policy)
        {
            require_finite("mar", mar);
            if (window_size_ < 2) {
                throw std::invalid_argument("Window size must be 2 or more.");
            }
            reset();
        }

        void reset() override {
            sum_e2_.reset();
            n_ = (start_policy_ != detail::StartPolicy::Zero) ? 0 : static_cast<std::size_t>(window_size_);
        }

        double process_scalar(double x) override {
            if (isnan2(x)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            if ((n_ < static_cast<std::size_t>(window_size_)) &&
                (start_policy_ != detail::StartPolicy::Zero)) {
                n_++;
            }
            double excess = x - mar_;
            if (excess < 0.0) excess = 0.0;               // upside only
            const double sum = sum_e2_.append(excess * excess);
            return std::sqrt(sum / static_cast<double>(n_));
        }

    private:
        const int window_size_;
        const double mar_;
        const detail::StartPolicy start_policy_;
        std::size_t n_ = 0;
        detail::RollingSum sum_e2_;
    };

} // namespace screamer

#endif // SCREAMER_ROLLING_UPSIDE_DEVIATION_H
