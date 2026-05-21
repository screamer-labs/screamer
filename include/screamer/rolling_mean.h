#ifndef SCREAMER_ROLLING_MEAN_H
#define SCREAMER_ROLLING_MEAN_H

#include "screamer/detail/rolling_mean.h"
#include "screamer/common/base.h"
namespace screamer {

    class RollingMean : public ScreamerBase {
    public:

        RollingMean(int window_size, const std::string& start_policy = "strict") : 
            rolling_mean_(window_size, start_policy)
        {
        }

        void reset() override {
            rolling_mean_.reset();
        }
        
        double process_scalar(double newValue) override {
            return rolling_mean_.append(newValue);
        }

        // NOTE: previous fast-path overrides used the sliding-sum recurrence,
        // which cannot honor the "ignore" NaN policy. See the corresponding
        // note in rolling_sum.h.

    private:
        screamer::detail::RollingMean rolling_mean_;

    }; // end of class

} // end of namespace

#endif // end of include guards
