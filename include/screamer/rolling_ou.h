
#ifndef SCREAMER_ROLLING_OU_H
#define SCREAMER_ROLLING_OU_H

#include <limits>
#include "screamer/common/base.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/rolling_sum.h"

/*
## OLS Fit

Calibrating the Ornstein-Uhlenbeck
S[i+1] = a S[i] + b + err

if a OLS linear regression problem with
y = a x + b + err

where x is the firt n-1 elements in the price windows, and y 
the last n-1 elements
 
Define the following summations:
Sx = x0 + x1 + ... + x(n-1)
Sy = y0 + y1 + ... + y(n-1)
Sxy = x0*y0 + x1*y1 + ... + x(n-1)*y(n-1)
Sxx = x0*x0 + x1*x1 + ... + x(n-1)*x(n-1)

The slope and intercept of OLS fit of y = a * x + b are then:

a = [n * Sxy - Sx * Sy] / [n * Sxx - Sx * Sx]
b = [Sy - a * Sx] / n

*/
namespace screamer {

    class RollingOU : public ScreamerBase {
    public:

        static int parse_output(const std::string& s) {
            if (s == "mrr")     return 0;
            if (s == "mean")    return 1;
            if (s == "relmean") return 2;
            if (s == "std")     return 3;
            throw std::invalid_argument(
                "output must be \"mrr\", \"mean\", \"relmean\", or \"std\".");
        }

        RollingOU(
            int window_size,
            const std::string& output = "mrr",
            const std::string& start_policy = "strict"
        ) :
            window_size_(window_size),
            output_(parse_output(output)),
            start_policy_(detail::parse_start_policy(start_policy)),
            sum_y_buffer(window_size - 1, start_policy),
            sum_yy_buffer(window_size - 1, start_policy),
            sum_xy_buffer(window_size - 1, start_policy)
        {
            if (window_size <= 0) {
                throw std::invalid_argument("Window size must be positive.");
            }

            reset();
        }

        void reset() override {
            sum_y_buffer.reset();
            sum_yy_buffer.reset();
            sum_xy_buffer.reset();
            n_ = (start_policy_ != detail::StartPolicy::Zero) ? 0 : window_size_;
            x = 0;
            y = 0;
            sum_x = 0;
            sum_y = 0;
            sum_xx = 0;
            sum_yy = 0;
            sum_xy = 0;
        }
        
        double process_scalar(double newValue) override {
            if (isnan2(newValue)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            if ((n_ < window_size_) && (start_policy_ != detail::StartPolicy::Zero) ) {
                n_++;
            }
            // the x values is the previous y
            x = y;
            y = newValue;

            sum_x = sum_y;
            sum_y = sum_y_buffer.append(y);

            sum_xx = sum_yy;
            sum_yy = sum_yy_buffer.append(y*y);

            sum_xy = sum_xy_buffer.append(x*y);

            double a_denominator = ((n_ - 1) * sum_xx - sum_x * sum_x);
            if (a_denominator == 0) return std::numeric_limits<double>::quiet_NaN();

            double a_numerator = ((n_ - 1) * sum_xy - sum_x * sum_y);
            double a = a_numerator / a_denominator;

            double mrr = 1.0 - a;
            if (output_ == 0) return mrr;

            double b = (sum_y - a * sum_x) / n_;
            double mu = b / mrr;
            if (output_ == 1) return mu;

            if (output_ == 2) return mu - y;

            double e = std::sqrt((n_ * sum_yy - sum_y * sum_y - a * a_numerator) / (n_ * (n_-2)));

            if (output_ == 3) return e;

            return std::numeric_limits<double>::quiet_NaN();
        }

    private:
        const int window_size_;
        const int output_;
        const detail::StartPolicy start_policy_;
        detail::RollingSum sum_y_buffer;
        detail::RollingSum sum_yy_buffer;
        detail::RollingSum sum_xy_buffer;
        int n_ = 0;
        double x = 0.0;
        double y = 0.0;
        double sum_x = 0.0;
        double sum_y = 0.0;
        double sum_xx = 0.0;
        double sum_yy = 0.0;
        double sum_xy = 0.0;

    }; // end of class

} // end of namespace

#endif // end of include guards

