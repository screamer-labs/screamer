#ifndef SCREAMER_ROLLING_CVAR_H
#define SCREAMER_ROLLING_CVAR_H

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include "screamer/common/buffer.h"
#include "screamer/common/base.h"
#include "common/order_statistic_tree.h"
#include "screamer/common/float_info.h"

namespace screamer {

    // RollingCVaR: historical Conditional Value-at-Risk (Expected Shortfall) of a
    // return series over a trailing window. With the worst k = max(1,
    // floor(alpha * window)) returns in the window,
    //     CVaR = -mean(the k smallest returns),
    // a positive number: the average loss in the worst alpha tail. It is the
    // coherent tail-risk measure that VaR is not (VaR is just the tail quantile,
    // -RollingQuantile(window, alpha)); CVaR averages beyond it. Maintains the
    // window in an order-statistic tree, so each step is O(log W) to update plus
    // O(k) to read the tail. The output is NaN until the window is full; a NaN
    // input leaves the window untouched (nan_policy: ignore).
    class RollingCVaR : public ScreamerBase {
    public:
        RollingCVaR(int window_size, double alpha = 0.05)
            : window_size_(window_size),
              alpha_(alpha),
              buffer_(window_size, std::numeric_limits<double>::quiet_NaN()),
              ost_(window_size)
        {
            if (window_size_ <= 0) {
                throw std::invalid_argument("Window size must be positive.");
            }
            if (alpha_ <= 0.0 || alpha_ >= 1.0) {
                throw std::invalid_argument("alpha must be between 0 and 1 (exclusive).");
            }
        }

        void reset() override {
            buffer_.reset(std::numeric_limits<double>::quiet_NaN());
            ost_.clear();
        }

        double process_scalar(double x) override {
            if (isnan2(x)) {
                return std::numeric_limits<double>::quiet_NaN();   // ignore
            }
            const double evicted = buffer_.append(x);
            if (!isnan2(evicted)) {
                ost_.erase(evicted);
            }
            ost_.insert(x);

            const int n = ost_.size();
            if (n < window_size_) {
                return std::numeric_limits<double>::quiet_NaN();   // warmup
            }
            const int k = std::max(1, static_cast<int>(std::floor(alpha_ * n)));
            double tail_sum = 0.0;
            for (int i = 0; i < k; ++i) {
                tail_sum += ost_.kth_element(i);          // the i-th smallest return
            }
            return -(tail_sum / k);                       // expected shortfall, a positive loss
        }

    private:
        const int window_size_;
        const double alpha_;
        FixedSizeBuffer buffer_;
        OrderStatisticTree ost_;
    };

} // namespace screamer

#endif // SCREAMER_ROLLING_CVAR_H
