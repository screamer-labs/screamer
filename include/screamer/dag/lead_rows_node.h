#ifndef SCREAMER_DAG_LEAD_ROWS_NODE_H
#define SCREAMER_DAG_LEAD_ROWS_NODE_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>
#include "screamer/dag/frame.h"

namespace screamer { namespace dag {

// Pairs the features of a row with the target `count` rows later. The input
// rows are [features..., target]. The row emitted for event t is
// [features of t..., target of t + count], stamped with the index of t, and it
// is emitted when event t + count arrives. flush() emits the rows still
// waiting, in order, with a NaN target. The width is set by the first event
// after a reset and is at least 2.
//
// The forwarded watermark is the index of the oldest waiting row (no later
// row can carry a lower stamp), or the incoming watermark when no row waits.
//
// The ring holds min(count, events seen) rows with their indexes: it grows
// geometrically during the warm-up, so a large count costs memory only as
// events arrive, and it does not reallocate once full. A row's last value is
// scratch: the emitted target is written there.
template <class Index>
class LeadRowsNode : public Sink<Index> {
public:
    LeadRowsNode(std::size_t count, Sink<Index>& downstream)
        : count_(count), downstream_(downstream) {}

    void push(const Frame<Index>& f) override {
        if (filled_ == count_) {                       // steady state
            check_width(f.width);
            double* slot = ring_.data() + pos_ * width_;
            // The slot holds the row of `count` events ago: emit its features
            // with this event's target, then store this event in its place.
            slot[width_ - 1] = f.values[width_ - 1];
            downstream_.push(Frame<Index>{index_[pos_], slot, width_});
            std::memcpy(slot, f.values, width_ * sizeof(double));
            index_[pos_] = f.index;
            if (++pos_ == count_) pos_ = 0;
            return;
        }
        // Warm-up: store the event; nothing is emitted yet.
        const bool first = width_ == 0;                // the first event after a reset
        if (first) {
            if (f.width < 2)
                throw std::invalid_argument(
                    "a lead takes rows of features and a target (at least 2 values), got " +
                    std::to_string(f.width));
            width_ = f.width;
        } else {
            check_width(f.width);
        }
        try {
            keep(f.index, f.values);
        } catch (...) {
            if (first) width_ = 0;
            throw;
        }
    }

    // Emits the waiting rows, oldest first, with a NaN target; the node is
    // then empty.
    void flush() override {
        const std::size_t start = filled_ == count_ ? pos_ : 0;
        for (std::size_t k = 0; k < filled_; ++k) {
            std::size_t s = start + k;
            if (s >= count_) s -= count_;
            double* slot = ring_.data() + s * width_;
            slot[width_ - 1] = std::numeric_limits<double>::quiet_NaN();
            downstream_.push(Frame<Index>{index_[s], slot, width_});
        }
        empty();
        downstream_.flush();
    }

    void on_watermark(Index w) override {
        Index out = w;
        if (filled_ != 0) out = std::min(out, index_[filled_ == count_ ? pos_ : 0]);
        if (out > last_wm_) {
            last_wm_ = out;
            downstream_.on_watermark(out);
        }
    }

    void reset() override {
        width_ = 0;
        empty();
        last_wm_ = std::numeric_limits<Index>::min();
    }

private:
    void empty() {
        pos_ = 0;
        filled_ = 0;
        ring_.clear();               // keeps its capacity for the next run
        index_.clear();
    }

    void check_width(std::size_t w) const {
        if (w != width_)
            throw std::invalid_argument("a lead takes rows of " + std::to_string(width_) +
                                        " values, got " + std::to_string(w));
    }

    // Appends a warm-up row, growing the ring geometrically up to count rows.
    void keep(Index index, const double* row) {
        if (filled_ == index_.capacity() || ring_.size() + width_ > ring_.capacity()) {
            // At least filled_ + 1 rows, since filled_ < count_.
            const std::size_t rows = std::min(std::max<std::size_t>(filled_ * 2, 64), count_);
            try {
                if (rows > SIZE_MAX / width_) throw std::length_error("lead ring");
                ring_.reserve(rows * width_);
                index_.reserve(rows);
            } catch (const std::bad_alloc&) {
                throw out_of_memory();
            } catch (const std::length_error&) {
                throw out_of_memory();
            }
        }
        ring_.insert(ring_.end(), row, row + width_);
        index_.push_back(index);
        ++filled_;
    }

    std::runtime_error out_of_memory() const {
        return std::runtime_error("a lead of " + std::to_string(count_) +
                                  " events ran out of memory after " +
                                  std::to_string(filled_) + " rows");
    }

    std::size_t count_;
    Sink<Index>& downstream_;
    std::size_t width_ = 0;          // 0 = not set since the last reset
    std::size_t pos_ = 0;            // the slot of the oldest row, once full
    std::size_t filled_ = 0;         // rows kept, up to count_
    std::vector<double> ring_;       // min(count_, events) rows of width_ values
    std::vector<Index> index_;       // the index of each kept row
    Index last_wm_ = std::numeric_limits<Index>::min();   // last forwarded watermark
};

}} // namespace screamer::dag
#endif
