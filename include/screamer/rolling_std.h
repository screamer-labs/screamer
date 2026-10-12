#ifndef SCREAMER_ROLLING_STD_H
#define SCREAMER_ROLLING_STD_H

#include <limits>
#include "screamer/detail/moment_formulas.h"
#include "screamer/detail/window_moments_op.h"
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace screamer {

    // Sample standard deviation (ddof=1) of the window. The window sums are
    // taken about an origin near the mean (detail::ShiftedWindowSums), so the
    // result stays accurate when the mean is large next to the spread.
    class RollingStd : public detail::WindowMomentsOp<RollingStd, 2> {
    public:

        RollingStd(int window_size, const std::string& start_policy = "strict") :
            WindowMomentsOp(window_size, detail::parse_start_policy(start_policy),
                            detail::parse_start_policy(start_policy))
        {
            if (window_size_ < 2) {
                throw std::invalid_argument("Window size must be 2 or more.");
            }
            reset();
        }

        inline double value(const State& st, double) const {
            if (start_policy_ == detail::StartPolicy::Strict &&
                st.count != static_cast<std::size_t>(window_size_)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            const double n = static_cast<double>(st.count);
            return std::sqrt(detail::var_from_sums(st.s[0], st.s[1], n));
        }
    }; // end of class

} // end of namespace

#endif // end of include guards
