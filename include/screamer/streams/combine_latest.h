#ifndef SCREAMER_STREAMS_COMBINE_LATEST_H
#define SCREAMER_STREAMS_COMBINE_LATEST_H

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace screamer { namespace streams {

// As-of latest-value join state for N sources fed a tagged event stream (in
// index order). Source i carries widths[i] values per event (its fields); the
// aligned row concatenates every source's latest fields in source order.
// on_event() updates the emitting source and returns whether a row is due:
//   when_all: only once every source has produced at least one event
//   on_any:   always (not-yet-seen sources read as NaN)
class CombineLatest {
public:
    CombineLatest(std::vector<std::size_t> widths, bool when_all)
        : widths_(std::move(widths)), seen_(widths_.size(), 0),
          seen_count_(0), when_all_(when_all) { layout(); }

    CombineLatest(std::size_t n, bool when_all)
        : CombineLatest(std::vector<std::size_t>(n, 1), when_all) {}

    bool on_event(std::uint32_t source, const double* values) {
        assert(source < widths_.size());
        if (!seen_[source]) { seen_[source] = 1; ++seen_count_; }
        const std::size_t w = widths_[source];
        if (w == 1) latest_[offset_[source]] = values[0];
        else std::copy(values, values + w, latest_.begin() + offset_[source]);
        return !(when_all_ && seen_count_ < widths_.size());
    }

    // A one-column source.
    bool on_event(std::uint32_t source, double value) {
        assert(widths_[source] == 1);
        return on_event(source, &value);
    }

    // Change the width of a source that has no value yet; the others keep theirs.
    void set_width(std::size_t source, std::size_t w) {
        if (seen_[source])
            throw std::logic_error("CombineLatest: a source's width is fixed by its first event");
        const std::vector<double> old = latest_;
        const std::vector<std::size_t> old_offset = offset_, old_width = widths_;
        widths_[source] = w;
        layout();
        for (std::size_t i = 0; i < widths_.size(); ++i)
            if (seen_[i])
                std::copy(old.begin() + old_offset[i], old.begin() + old_offset[i] + old_width[i],
                          latest_.begin() + offset_[i]);
    }

    std::size_t width(std::size_t source) const { return widths_[source]; }
    std::size_t total_width() const { return latest_.size(); }
    const std::vector<double>& latest() const { return latest_; }

    // reset() with the source widths `widths` (as many as before). Storage
    // is reused, so a reset to the widths of an earlier run allocates nothing.
    void reset(const std::vector<std::size_t>& widths) {
        if (widths != widths_) {
            widths_.assign(widths.begin(), widths.end());
            layout();
        }
        reset();
    }

    void reset() {
        std::fill(latest_.begin(), latest_.end(), std::numeric_limits<double>::quiet_NaN());
        std::fill(seen_.begin(), seen_.end(), static_cast<char>(0));
        seen_count_ = 0;
    }

private:
    void layout() {
        offset_.resize(widths_.size());
        std::size_t total = 0;
        for (std::size_t i = 0; i < widths_.size(); ++i) { offset_[i] = total; total += widths_[i]; }
        latest_.assign(total, std::numeric_limits<double>::quiet_NaN());
    }

    std::vector<std::size_t> widths_, offset_;
    std::vector<double> latest_;
    std::vector<char> seen_;
    std::size_t seen_count_;
    bool when_all_;
};

}} // namespace screamer::streams
#endif
