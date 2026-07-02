#ifndef SCREAMER_STREAMS_COMBINE_LATEST_H
#define SCREAMER_STREAMS_COMBINE_LATEST_H

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace screamer { namespace streams {

// As-of latest-value join state for N sources fed a tagged event stream (in
// key order). on_event() updates the emitting source's latest value and marks
// it seen; it returns whether an aligned row should be emitted now:
//   when_all: only once every source has produced at least one value
//   on_any:   always (not-yet-seen sources read as NaN)
// latest() is the N-wide aligned row valid immediately after on_event().
class CombineLatest {
public:
    CombineLatest(std::size_t n, bool when_all)
        : latest_(n, std::numeric_limits<double>::quiet_NaN()),
          seen_(n, 0), n_(n), seen_count_(0), when_all_(when_all) {}

    bool on_event(std::uint32_t source, double value) {
        assert(source < n_);
        if (!seen_[source]) { seen_[source] = 1; ++seen_count_; }
        latest_[source] = value;
        if (when_all_ && seen_count_ < n_) return false;
        return true;
    }

    const std::vector<double>& latest() const { return latest_; }

private:
    std::vector<double> latest_;
    std::vector<char> seen_;
    std::size_t n_;
    std::size_t seen_count_;
    bool when_all_;
};

}} // namespace screamer::streams
#endif
