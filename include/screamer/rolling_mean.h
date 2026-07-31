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

        // Batch path, experiment: hoist the recurrence state into locals so
        // it stays in registers across the loop, then write it back. The
        // per-sample path keeps state behind `this`, and the compiler cannot
        // prove the output buffer does not alias the operator, so it must
        // reload the running sum and index every iteration.
        void process_array_no_stride(double* y, const double* x, size_t size) override {
            auto& state = rolling_mean_;
            double* buffer = state.buffer_data();
            const size_t capacity = state.capacity();
            size_t index = state.index();
            size_t count = state.count();
            double sum = state.sum();
            const auto policy = state.start_policy();

            for (size_t i = 0; i < size; ++i) {
                const double v = x[i];
                if (isnan2(v)) {
                    y[i] = std::numeric_limits<double>::quiet_NaN();
                    continue;
                }
                const double old = buffer[index];
                buffer[index] = v;
                ++index;
                if (index == capacity) index = 0;

                if (count == capacity) {
                    sum += v - old;
                    y[i] = sum / capacity;
                } else {
                    sum += v;
                    ++count;
                    if (count == capacity) {
                        y[i] = sum / capacity;
                    } else if (policy == detail::StartPolicy::Strict) {
                        y[i] = std::numeric_limits<double>::quiet_NaN();
                    } else if (policy == detail::StartPolicy::Expanding) {
                        y[i] = sum / count;      // grows with the samples seen
                    } else {
                        y[i] = sum / capacity;   // zero: pad the missing past
                    }
                }
            }
            state.restore(index, count, sum);
        }

        // NOTE: previous fast-path overrides used the sliding-sum recurrence,
        // which cannot honor the "ignore" NaN policy. See the corresponding
        // note in rolling_sum.h.

    private:
        screamer::detail::RollingMean rolling_mean_;

    }; // end of class

} // end of namespace

#endif // end of include guards
