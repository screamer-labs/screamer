#ifndef SCREAMER_DETAIL_ROLLING_MEAN_H
#define SCREAMER_DETAIL_ROLLING_MEAN_H

#include <vector>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include "screamer/common/float_info.h"
#include "screamer/detail/start_policy.h"
#include <cstddef>
#include <string>

namespace screamer {
namespace detail {


// One step of a trailing-window mean on plain state: the ring buffer `buf` of
// `cap` values, the next slot `index`, the samples seen `count` (up to cap)
// and their `sum`. Returns the mean after adding v. NaN policy "ignore": a NaN
// v leaves the state unchanged and returns NaN (see docs/nan_policy.md). The
// one step of RollingMean::append, the RollingMean batch loop and the
// operators built on it (AmihudIlliquidity), which keep the state in locals.
inline double rolling_mean_step(double* buf, std::size_t cap, std::size_t& index,
                                std::size_t& count, double& sum, StartPolicy policy, double v) {
    if (screamer::isnan2(v)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double old = buf[index];
    buf[index] = v;
    if (++index == cap) {
        index = 0;
    }
    // the most common case, we are past the start period
    if (count == cap) {
        sum += v - old;
        return sum / cap;
    }
    // all other cases: add the new element to the sum, increment count
    sum += v;
    ++count;
    // if we reached capacity then we have a valid sum
    if (count == cap) {
        return sum / cap;
    }
    // if we haven't reached capacity yet then it depends on the policy
    if (policy == StartPolicy::Strict) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (policy == StartPolicy::Expanding) {
        return sum / count;
    }
    return sum / cap;   // StartPolicy::Zero: pad the missing past
}

class RollingMean {
public:
    RollingMean(size_t size, const std::string& start_policy = "strict") 
        : 
        capacity_(size),
        start_policy_(parse_start_policy(start_policy)),
        index_(0),
        size_(0),
        sum_(0)
    {
        if (size < 1) {
            throw std::invalid_argument("Size must be at least 1.");
        }

        buffer_.resize(size);
        reset();
    }

    void reset() 
    {
        sum_ = 0;
        size_ = 0;
        index_ = 0;
        std::fill(buffer_.begin(), buffer_.end(), 0.0);
    }

    double append(double newValue)
    {
        return rolling_mean_step(buffer_.data(), capacity_, index_, size_, sum_,
                                 start_policy_, newValue);
    }

    const size_t size() {
        return size_;
    }

    const size_t capacity() {
        return capacity_;
    }

    const StartPolicy start_policy() {
        return start_policy_;
    }

    // Accessors for a hoisted batch loop: see screamer/rolling_mean.h. The
    // per-sample path keeps this state behind `this`, which the compiler must
    // reload each iteration because it cannot prove the caller's output buffer
    // does not alias the operator.
    double* buffer_data() { return buffer_.data(); }
    size_t capacity() const { return capacity_; }
    size_t index() const { return index_; }
    size_t count() const { return size_; }
    double sum() const { return sum_; }
    StartPolicy start_policy() const { return start_policy_; }
    void restore(size_t index, size_t count, double sum) {
        index_ = index;
        size_ = count;
        sum_ = sum;
    }

private:
    const size_t capacity_;
    const StartPolicy start_policy_; 
    size_t index_;
    size_t size_;
    double sum_;
    std::vector<double> buffer_;

}; // class

} // namespace detail
} // namespace screamer
#endif // include guards