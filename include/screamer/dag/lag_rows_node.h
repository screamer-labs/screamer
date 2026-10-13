#ifndef SCREAMER_DAG_LAG_ROWS_NODE_H
#define SCREAMER_DAG_LAG_ROWS_NODE_H

#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "screamer/dag/frame.h"

namespace screamer { namespace dag {

// Lags rows of any width by `count` events: the row emitted at event t is the
// row of event t - count, or a NaN row for the first `count` events (Lag's
// strict start). The width is set by the first event after a reset. The index
// and the watermarks pass through.
template <class Index>
class LagRowsNode : public Sink<Index> {
public:
    LagRowsNode(std::size_t count, Sink<Index>& downstream)
        : count_(count), downstream_(downstream) {}

    void push(const Frame<Index>& f) override {
        if (width_ == 0) {
            width_ = f.width;
            ring_.assign(count_ * width_, std::numeric_limits<double>::quiet_NaN());
        } else if (f.width != width_) {
            throw std::invalid_argument("a lag takes rows of " + std::to_string(width_) +
                                        " values, got " + std::to_string(f.width));
        }
        double* slot = ring_.data() + pos_ * width_;
        // The slot holds the row of `count` events ago: emit it, then store
        // this event's row in its place.
        downstream_.push(Frame<Index>{f.index, slot, width_});
        std::memcpy(slot, f.values, width_ * sizeof(double));
        if (++pos_ == count_) pos_ = 0;
    }

    void flush() override { downstream_.flush(); }
    void on_watermark(Index w) override { downstream_.on_watermark(w); }

    void reset() override {
        width_ = 0;
        pos_ = 0;
        ring_.clear();
    }

private:
    std::size_t count_;
    Sink<Index>& downstream_;
    std::size_t width_ = 0;          // 0 = not set since the last reset
    std::size_t pos_ = 0;            // the slot of the oldest row
    std::vector<double> ring_;       // count_ rows of width_ values
};

}} // namespace screamer::dag
#endif
