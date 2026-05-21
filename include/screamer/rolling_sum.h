#ifndef SCREAMER_ROLLING_SUM_H
#define SCREAMER_ROLLING_SUM_H

#include "screamer/detail/rolling_sum.h"
#include "screamer/common/base.h"
namespace screamer {

    class RollingSum : public ScreamerBase {
    public:

        RollingSum(int window_size, const std::string& start_policy = "strict") : 
            rolling_sum_(window_size, start_policy)
        {
        }

        void reset() override {
            rolling_sum_.reset();
        }
        
        double process_scalar(double newValue) override {
            return rolling_sum_.append(newValue);
        }

        // NOTE: the previous fast-path overrides used the sliding-sum
        // recurrence y[i] = y[i-1] + x[i] - x[i-w]. That recurrence cannot
        // handle the "ignore"-policy NaN semantics: skipping a NaN input
        // means the buffer does NOT slide forward, so x[i-w] does not refer
        // to the value w positions ago in the FINITE-sample history. The
        // scalar fallback in ScreamerBase::process_array_no_stride calls
        // process_scalar in a loop and produces correct output. If perf
        // becomes a concern we can re-add a NaN-aware fast path that
        // detects all-finite chunks and uses the recurrence on those.

    private:
        screamer::detail::RollingSum rolling_sum_;

    }; // end of class

} // end of namespace

#endif // end of include guards
