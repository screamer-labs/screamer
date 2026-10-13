#ifndef SCREAMER_DAG_LAG_ROWS_NODE_H
#define SCREAMER_DAG_LAG_ROWS_NODE_H

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

// Lags rows of any width by `count` events: the row emitted at event t is the
// row of event t - count, or a NaN row for the first `count` events (Lag's
// strict start). The width is set by the first event after a reset. The index
// and the watermarks pass through.
//
// The ring holds min(count, events seen) rows: it grows geometrically during
// the warm-up, so a large count costs memory only as events arrive, and it
// does not reallocate once full.
template <class Index>
class LagRowsNode : public Sink<Index> {
public:
    LagRowsNode(std::size_t count, Sink<Index>& downstream)
        : count_(count), downstream_(downstream) {}

    void push(const Frame<Index>& f) override {
        if (filled_ == count_) {                       // steady state
            if (f.width == 1 && width_ == 1) {
                double& slot = ring_[pos_];
                const double out = slot;
                slot = f.values[0];
                if (++pos_ == count_) pos_ = 0;
                downstream_.push(Frame<Index>{f.index, &out, 1});
                return;
            }
            check_width(f.width);
            double* slot = ring_.data() + pos_ * width_;
            // The slot holds the row of `count` events ago: emit it, then
            // store this event's row in its place.
            downstream_.push(Frame<Index>{f.index, slot, width_});
            std::memcpy(slot, f.values, width_ * sizeof(double));
            if (++pos_ == count_) pos_ = 0;
            return;
        }
        if (width_ == 0) {                             // the first event after a reset
            if (f.width == 0)
                throw std::invalid_argument("a lag takes rows of at least one value");
            width_ = f.width;
            nan_row_.assign(width_, std::numeric_limits<double>::quiet_NaN());
        } else {
            check_width(f.width);
        }
        downstream_.push(Frame<Index>{f.index, nan_row_.data(), width_});
        keep(f.values);
    }

    void flush() override { downstream_.flush(); }
    void on_watermark(Index w) override { downstream_.on_watermark(w); }

    void reset() override {
        width_ = 0;
        pos_ = 0;
        filled_ = 0;
        ring_.clear();               // keeps its capacity for the next run
    }

private:
    void check_width(std::size_t w) const {
        if (w != width_)
            throw std::invalid_argument("a lag takes rows of " + std::to_string(width_) +
                                        " values, got " + std::to_string(w));
    }

    // Appends a warm-up row, growing the ring geometrically up to count rows.
    void keep(const double* row) {
        const std::size_t need = ring_.size() + width_;
        if (need > ring_.capacity()) {
            const std::size_t full = width_ > SIZE_MAX / count_ ? SIZE_MAX : count_ * width_;
            std::size_t cap = std::max<std::size_t>(ring_.capacity() * 2, 64 * width_);
            cap = std::min(std::max(cap, need), full);
            try {
                ring_.reserve(cap);
            } catch (const std::bad_alloc&) {
                throw std::runtime_error("a lag of " + std::to_string(count_) +
                                         " events ran out of memory after " +
                                         std::to_string(filled_) + " rows");
            } catch (const std::length_error&) {
                throw std::runtime_error("a lag of " + std::to_string(count_) +
                                         " events ran out of memory after " +
                                         std::to_string(filled_) + " rows");
            }
        }
        ring_.insert(ring_.end(), row, row + width_);
        ++filled_;
    }

    std::size_t count_;
    Sink<Index>& downstream_;
    std::size_t width_ = 0;          // 0 = not set since the last reset
    std::size_t pos_ = 0;            // the slot of the oldest row, once full
    std::size_t filled_ = 0;         // rows kept, up to count_
    std::vector<double> ring_;       // min(count_, events) rows of width_ values
    std::vector<double> nan_row_;    // the warm-up row
};

}} // namespace screamer::dag
#endif
