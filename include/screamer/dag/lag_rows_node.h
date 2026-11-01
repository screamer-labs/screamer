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
#include "screamer/dag/node_state.h"

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
        // Warm-up: store the event, then emit the NaN row, so an event that
        // the ring cannot store (out of memory), or that downstream rejects,
        // leaves the node as it was.
        const bool first = width_ == 0;                // the first event after a reset
        if (first) {
            if (f.width == 0)
                throw std::invalid_argument("a lag takes rows of at least one value");
            nan_row_.assign(f.width, std::numeric_limits<double>::quiet_NaN());
            width_ = f.width;
        } else {
            check_width(f.width);
        }
        try {
            keep(f.values);
        } catch (...) {
            if (first) width_ = 0;
            throw;
        }
        // A downstream that rejects the row: take the event back out, so the
        // node is as it was, as in the steady state.
        try {
            downstream_.push(Frame<Index>{f.index, nan_row_.data(), width_});
        } catch (...) {
            ring_.resize(ring_.size() - width_);
            --filled_;
            if (first) width_ = 0;
            throw;
        }
    }

    void flush() override { downstream_.flush(); }
    void on_watermark(Index w) override { downstream_.on_watermark(w); }

    void reset() override {
        width_ = 0;
        pos_ = 0;
        filled_ = 0;
        ring_.clear();               // keeps its capacity for the next run
    }

    static constexpr int kStateVersion = 1;
    bool saveable() const override { return true; }
    void write_state(json::Value& out) const override { write_node_state(*this, out); }
    void load_into(const json::Value& in, const std::string& path, serial::detail::Loader& ar) override {
        load_node_state(*this, in, path, ar);
    }

    // What reset() touches: the width and the kept rows, oldest first.
    template <class Ar>
    void state(Ar& ar) {
        ar("width", width_);
        std::vector<double> rows;
        if constexpr (Ar::kSaving) {
            const std::size_t first = filled_ == count_ ? pos_ : 0;
            for (std::size_t k = 0; k < filled_; ++k) {
                const std::size_t s = (first + k) % filled_;
                rows.insert(rows.end(), ring_.begin() + s * width_, ring_.begin() + (s + 1) * width_);
            }
        }
        ar.local("rows", rows);
        if constexpr (Ar::kLoading) {
            ar.check(width_ != 0 || rows.empty(), "rows are kept before the width is known");
            const std::size_t n = width_ ? rows.size() / width_ : 0;
            ar.check(n * width_ == rows.size(), "the kept rows do not fit the width");
            ar.check(n <= count_, "more rows are kept than the lag count");
            ar.apply([this, rows, n] {
                         ring_ = rows;
                         filled_ = n;
                         pos_ = 0;
                         nan_row_.assign(width_, std::numeric_limits<double>::quiet_NaN());
                     },
                     [this, ring = ring_, filled = filled_, pos = pos_, nan_row = nan_row_] {
                         ring_ = ring; filled_ = filled; pos_ = pos; nan_row_ = nan_row;
                     });
        }
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
