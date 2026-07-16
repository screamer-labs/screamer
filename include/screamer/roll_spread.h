#ifndef SCREAMER_ROLL_SPREAD_H
#define SCREAMER_ROLL_SPREAD_H

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include "screamer/common/base.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/rolling_sum.h"
#include "screamer/detail/start_policy.h"

namespace screamer {

    // Roll's (1984) effective spread from trade prices alone:
    //     spread = 2 * sqrt(-cov(dP_t, dP_{t-1}))
    // over a trailing window, where dP is the one-step price change. Bid-ask
    // bounce makes successive price changes negatively correlated; when the
    // serial covariance is non-negative the estimate is undefined and returns
    // NaN. Internally this is a rolling covariance of the price change against
    // its own one-step lag (three detail::RollingSum buffers, O(1) per step). A
    // NaN price is ignored (nan_policy: ignore).
    class RollSpread : public ScreamerBase {
    public:
        RollSpread(int window_size, const std::string& start_policy = "strict")
            : window_size_(window_size),
              start_policy_(detail::parse_start_policy(start_policy)),
              sum_x_buffer(window_size, "expanding"),
              sum_y_buffer(window_size, "expanding"),
              sum_xy_buffer(window_size, "expanding")
        {
            if (window_size_ < 2) {
                throw std::invalid_argument("Window size must be 2 or more.");
            }
            reset();
        }

        void reset() override {
            sum_x_buffer.reset();
            sum_y_buffer.reset();
            sum_xy_buffer.reset();
            have_price_ = false;
            have_dp_ = false;
            prev_price_ = 0.0;
            prev_dp_ = 0.0;
            n_ = (start_policy_ == detail::StartPolicy::Zero) ? window_size_ : 0;
        }

        double process_scalar(double price) override {
            const double nan = std::numeric_limits<double>::quiet_NaN();
            if (isnan2(price)) {                       // ignore: missing tick
                return nan;
            }
            if (!have_price_) {                        // need one prior price for dP
                have_price_ = true;
                prev_price_ = price;
                return nan;
            }
            const double dp = price - prev_price_;     // dP_t
            prev_price_ = price;
            if (!have_dp_) {                           // need one prior dP for the lag
                have_dp_ = true;
                prev_dp_ = dp;
                return nan;
            }
            const double x = dp;                       // dP_t
            const double y = prev_dp_;                 // dP_{t-1}
            prev_dp_ = dp;

            const double sum_x = sum_x_buffer.append(x);
            const double sum_y = sum_y_buffer.append(y);
            const double sum_xy = sum_xy_buffer.append(x * y);

            if ((n_ < window_size_) && (start_policy_ != detail::StartPolicy::Zero)) {
                n_++;
            }
            if (start_policy_ == detail::StartPolicy::Strict && n_ < window_size_) {
                return nan;
            }
            if (n_ < 2) {
                return nan;
            }

            const double nd = static_cast<double>(n_);
            const double cov = (nd * sum_xy - sum_x * sum_y) / (nd * (nd - 1.0));
            // Bid-ask bounce implies cov < 0; a non-negative cov leaves Roll's
            // estimate undefined.
            return (cov < 0.0) ? 2.0 * std::sqrt(-cov) : nan;
        }

    private:
        const int window_size_;
        const detail::StartPolicy start_policy_;
        bool have_price_ = false;
        bool have_dp_ = false;
        double prev_price_ = 0.0;
        double prev_dp_ = 0.0;
        size_t n_ = 0;
        detail::RollingSum sum_x_buffer;
        detail::RollingSum sum_y_buffer;
        detail::RollingSum sum_xy_buffer;
    };

} // namespace screamer

#endif // SCREAMER_ROLL_SPREAD_H
