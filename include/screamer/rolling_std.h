
#ifndef SCREAMER_ROLLING_STD_H
#define SCREAMER_ROLLING_STD_H

#include <limits>
#include <screamer/detail/rolling_sum.h>
#include "screamer/common/base.h"
#include "screamer/common/float_info.h"
#include "screamer/common/math.h"
/*
todo: this implementation might  suffer from numerical instability
      https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
*/
namespace screamer {

    class RollingStd : public ScreamerBase {
    public:

        RollingStd(int window_size, const std::string& start_policy = "strict") : 
            window_size_(window_size), 
            start_policy_(detail::parse_start_policy(start_policy)),
            sum_y_buffer(window_size, start_policy),
            sum_y2_buffer(window_size, start_policy)
        {
            if (window_size_ < 2) {
                throw std::invalid_argument("Window size must be 2 or more.");
            }

            reset();
        }

        void reset() override {
            sum_y_buffer.reset();
            sum_y2_buffer.reset();   
            n_ = (start_policy_ != detail::StartPolicy::Zero) ? 0 : window_size_;
        }
        
        double process_scalar(double newValue) override {
            // NaN policy "ignore": leave n_ and the running sums untouched.
            if (isnan2(newValue)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            if ((n_ < window_size_) && (start_policy_ != detail::StartPolicy::Zero) ) {
                n_++;
            }
            double sum_y = sum_y_buffer.append(newValue);
            double sum_y2 = sum_y2_buffer.append(newValue * newValue);

            double var;
            var_from_stats(sum_y, sum_y2, n_, var);
            return std::sqrt(var);
        }

        // Fast-path override removed; scalar fallback honors the "ignore"
        // NaN policy. See rolling_sum.h for rationale.
    private:
        const int window_size_;
        const detail::StartPolicy start_policy_;
        size_t n_ = 0;
        screamer::detail::RollingSum sum_y_buffer;
        screamer::detail::RollingSum sum_y2_buffer;

    }; // end of class

} // end of namespace

#endif // end of include guards

