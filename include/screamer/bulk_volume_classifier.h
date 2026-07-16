#ifndef SCREAMER_BULK_VOLUME_CLASSIFIER_H
#define SCREAMER_BULK_VOLUME_CLASSIFIER_H

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include "screamer/common/base.h"
#include "screamer/common/float_info.h"
#include "screamer/common/math.h"
#include "screamer/detail/rolling_sum.h"
#include "screamer/detail/start_policy.h"

namespace screamer {

    // Bulk Volume Classification (Easley-Lopez de Prado-O'Hara 2012): the
    // buy-initiated share of a bar's volume, estimated as the standard normal CDF
    // of the bar return divided by its trailing-window volatility,
    //     buy_fraction = Phi(return / sigma),
    // where sigma is the rolling sample standard deviation of the return. Works on
    // aggregate bars, no tick data needed; the output is a fraction in [0, 1].
    // Two detail::RollingSum buffers under the hood (Sy, Syy), O(1) per step.
    class BulkVolumeClassifier : public ScreamerBase {
    public:
        BulkVolumeClassifier(int window_size, const std::string& start_policy = "strict")
            : window_size_(window_size),
              start_policy_(detail::parse_start_policy(start_policy)),
              sum_y_buffer(window_size, "expanding"),
              sum_yy_buffer(window_size, "expanding")
        {
            if (window_size_ < 2) {
                throw std::invalid_argument("Window size must be 2 or more.");
            }
            reset();
        }

        void reset() override {
            sum_y_buffer.reset();
            sum_yy_buffer.reset();
            n_ = (start_policy_ == detail::StartPolicy::Zero) ? window_size_ : 0;
        }

        double process_scalar(double newValue) override {
            // NaN policy "ignore": leave n_ and the running sums untouched.
            if (isnan2(newValue)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            const double sum_y = sum_y_buffer.append(newValue);
            const double sum_yy = sum_yy_buffer.append(newValue * newValue);

            if ((n_ < window_size_) && (start_policy_ != detail::StartPolicy::Zero)) {
                n_++;
            }

            const double nan = std::numeric_limits<double>::quiet_NaN();
            if (start_policy_ == detail::StartPolicy::Strict && n_ < window_size_) {
                return nan;
            }
            if (n_ < 2) {
                return nan;
            }

            double var;
            var_from_stats(sum_y, sum_yy, static_cast<int>(n_), var);
            const double sigma = std::sqrt(var);
            if (!(sigma > 0.0)) {
                return nan;                            // zero-variance window: undefined
            }
            return standard_normal_cdf(newValue / sigma);
        }

    private:
        const int window_size_;
        const detail::StartPolicy start_policy_;
        size_t n_ = 0;
        detail::RollingSum sum_y_buffer;
        detail::RollingSum sum_yy_buffer;
    };

} // namespace screamer

#endif // SCREAMER_BULK_VOLUME_CLASSIFIER_H
