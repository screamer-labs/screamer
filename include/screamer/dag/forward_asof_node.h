#ifndef SCREAMER_DAG_FORWARD_ASOF_NODE_H
#define SCREAMER_DAG_FORWARD_ASOF_NODE_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "screamer/dag/frame.h"
#include "screamer/dag/max_pending.h"
#include "screamer/dag/resettable.h"

namespace screamer { namespace dag {

// A forward as-of join on the clock of X. Port 0 takes the rows of X (any
// width), port 1 the events of y (one value). The row emitted for the X row
// at index t is [X..., the last y at or before t], stamped t, in X order; for
// several y events at one index, the last of them counts.
//
// The row for t waits until y has passed t: a y event above t, or a y event
// at t and a y watermark above t. flush() emits the rows still waiting: with
// the last y at or before t when a y event at or after t was seen, else with
// a NaN target (y ended before t). The target is NaN too when no y event is
// at or before t.
//
// forecast_pairs feeds it y delayed by -duration, so "y at or before t" is
// "y at or before t + duration" on y's own index.
//
// The node keeps the waiting X rows (as many as y lags behind them) and the
// y events no waiting or later X row has used yet (above the X frontier: the
// highest X index or watermark). While one merge feeds the graph in index
// order, an X port that is in order and a y port that is never ahead make
// every y event at or below the frontier, so none is kept; otherwise (a live
// session with y pushed ahead of X) at most `max_pending` are, as in the
// reorder buffer of CombineLatest.
//
// The forwarded watermark is the index of the oldest waiting row, or the X
// frontier when no row waits.
template <class Index>
class ForwardAsOfNode : public Resettable {
public:
    explicit ForwardAsOfNode(Sink<Index>& downstream,
                             std::size_t max_pending = kDefaultMaxPending)
        : downstream_(downstream), max_pending_(max_pending) {
        ports_.reserve(2);
        for (std::size_t i = 0; i < 2; ++i) ports_.emplace_back(*this, i);
    }

    // Non-movable/copyable: the ports hold a reference back to this node.
    ForwardAsOfNode(const ForwardAsOfNode&) = delete;
    ForwardAsOfNode& operator=(const ForwardAsOfNode&) = delete;
    ForwardAsOfNode(ForwardAsOfNode&&) = delete;
    ForwardAsOfNode& operator=(ForwardAsOfNode&&) = delete;

    Sink<Index>& port(std::size_t i) { return ports_[i]; }

    // How the ports relate to the merge of the graph's inputs (CompiledGraph
    // sets it at wiring, as for a clock resample): X never comes below an
    // index the merge has passed, y never comes above the merge.
    void set_ports(bool x_in_order, bool y_never_ahead) {
        y_bounds_x_ = x_in_order && y_never_ahead;
    }

    // Whether one merge feeds the graph's inputs in index order.
    void set_merge_order(bool on) { merge_order_ = on; }

    void reset() override {
        xw_ = 0;
        head_ = 0;
        size_ = 0;
        yev_.clear();                // keeps its capacity for the next run
        yhead_ = 0;
        base_ = std::numeric_limits<double>::quiet_NaN();
        have_y_ = false;
        y_last_ = kMin;
        y_wm_ = kMin;
        x_floor_ = kMin;
        last_wm_ = kMin;
        flushed_[0] = flushed_[1] = false;
    }

private:
    static constexpr Index kMin = std::numeric_limits<Index>::min();
    static constexpr Index kMax = std::numeric_limits<Index>::max();

    void on_x(const Frame<Index>& f) {
        const bool first = xw_ == 0;                   // the first X row after a reset
        if (first) {
            if (f.width == 0)
                throw std::invalid_argument("a forward as-of join takes X rows of at least "
                                            "one value");
            xw_ = f.width;
        } else if (f.width != xw_) {
            throw std::invalid_argument("a forward as-of join takes X rows of " +
                                        std::to_string(xw_) + " values, got " +
                                        std::to_string(f.width));
        }
        try {
            append(f.index, f.values);
        } catch (...) {
            if (first) xw_ = 0;
            throw;
        }
        if (x_floor_ < f.index) x_floor_ = f.index;
        settle();
    }

    void on_y(const Frame<Index>& f) {
        if (f.width != 1)
            throw std::invalid_argument("a forward as-of join takes y events of one value, "
                                        "got " + std::to_string(f.width));
        yev_.emplace_back(f.index, f.values[0]);
        y_last_ = f.index;
        have_y_ = true;
        // The merge has reached f.index: no later X row comes below it.
        if (merge_order_ && y_bounds_x_ && x_floor_ < f.index) x_floor_ = f.index;
        settle();
        if (yev_.size() - yhead_ > max_pending_)
            throw std::runtime_error(
                "forecast_pairs: the target is more than " + std::to_string(max_pending_) +
                " events ahead of the features; a stream is stalled. Call advance(now) "
                "or check the feed.");
    }

    void on_watermark(std::size_t port, Index w) {
        if (port == 0) { if (x_floor_ < w) x_floor_ = w; }
        else if (y_wm_ < w) y_wm_ = w;
        settle();
        const Index out = size_ ? stamp_[head_] : x_floor_;
        if (out > last_wm_) {
            last_wm_ = out;
            downstream_.on_watermark(out);
        }
    }

    // Called once per port at end of input; the second port drains the
    // waiting rows and flushes downstream once.
    void on_flush(std::size_t port) {
        if (port == 0) x_floor_ = kMax;
        else y_wm_ = kMax;
        flushed_[port] = true;
        settle();
        if (!(flushed_[0] && flushed_[1])) return;
        while (size_) {
            const Index t = stamp_[head_];
            take_y(t);
            emit(have_y_ && y_last_ >= t ? base_ : std::numeric_limits<double>::quiet_NaN());
        }
        downstream_.flush();
        flushed_[0] = flushed_[1] = false;            // re-armed: a repeat flush emits nothing
    }

    // Emits the waiting rows whose target is known, oldest first, then drops
    // the y events that no waiting or later X row needs.
    void settle() {
        while (size_) {
            const Index t = stamp_[head_];
            if (!(have_y_ && (y_last_ > t || (y_last_ == t && y_wm_ > t)))) break;
            take_y(t);
            emit(base_);
        }
        // A waiting row t has seen no y above t, so every kept y is at or
        // below it; a later X row comes at or above the frontier.
        take_y(size_ ? stamp_[head_] : x_floor_);
    }

    // Moves the y events at or before t into base_ (the last of them).
    void take_y(Index t) {
        while (yhead_ < yev_.size() && yev_[yhead_].first <= t) base_ = yev_[yhead_++].second;
        if (yhead_ == yev_.size()) { yev_.clear(); yhead_ = 0; }
        else if (yhead_ > 64 && yhead_ * 2 > yev_.size()) {
            yev_.erase(yev_.begin(), yev_.begin() + static_cast<std::ptrdiff_t>(yhead_));
            yhead_ = 0;
        }
    }

    // Emits the oldest waiting row with `target`, then removes it. A
    // downstream that rejects the row leaves it waiting.
    void emit(double target) {
        double* row = rows_.get() + head_ * (xw_ + 1);
        row[xw_] = target;
        downstream_.push(Frame<Index>{stamp_[head_], row, xw_ + 1});
        if (++head_ == cap_) head_ = 0;
        if (--size_ == 0) head_ = 0;
    }

    // Appends an X row to the ring of waiting rows, doubling it when full.
    // The new storage is not zero-filled: only the copied rows are read.
    void append(Index t, const double* x) {
        const std::size_t rw = xw_ + 1;
        if (size_ == cap_ || rows_len_ / rw < cap_) {
            // Full: double; a wider X after a reset: the same number of rows.
            const std::size_t cap = std::max<std::size_t>(
                size_ == cap_ ? (cap_ > SIZE_MAX / 2 ? SIZE_MAX : cap_ * 2) : cap_, 16);
            if (cap > SIZE_MAX / rw || cap > SIZE_MAX / sizeof(double) / rw)
                throw std::runtime_error("forecast_pairs: " + std::to_string(size_) +
                                         " rows waiting for their target ran out of memory");
            std::unique_ptr<double[]> rows(new double[cap * rw]);
            std::unique_ptr<Index[]> stamp(new Index[cap]);
            for (std::size_t k = 0; k < size_; ++k) {
                const std::size_t s = (head_ + k) % cap_;
                std::memcpy(rows.get() + k * rw, rows_.get() + s * rw, rw * sizeof(double));
                stamp[k] = stamp_[s];
            }
            rows_ = std::move(rows);
            stamp_ = std::move(stamp);
            rows_len_ = cap * rw;
            cap_ = cap;
            head_ = 0;
        }
        std::size_t s = head_ + size_;
        if (s >= cap_) s -= cap_;
        std::memcpy(rows_.get() + s * rw, x, xw_ * sizeof(double));
        stamp_[s] = t;
        ++size_;
    }

    struct Port : Sink<Index> {
        ForwardAsOfNode& node;
        std::size_t idx;
        Port(ForwardAsOfNode& n, std::size_t i) : node(n), idx(i) {}
        void push(const Frame<Index>& f) override {
            if (idx == 0) node.on_x(f); else node.on_y(f);
        }
        void flush() override { node.on_flush(idx); }
        void on_watermark(Index w) override { node.on_watermark(idx, w); }
    };
    friend struct Port;

    Sink<Index>& downstream_;
    std::size_t max_pending_;
    std::vector<Port> ports_;
    bool y_bounds_x_ = false;        // set_ports
    bool merge_order_ = false;       // set_merge_order

    // The waiting X rows: a ring of cap_ rows of xw_ + 1 values (the last is
    // the target, written when the row is emitted) and their indexes.
    std::size_t xw_ = 0;             // 0 = not set since the last reset
    std::unique_ptr<double[]> rows_;
    std::unique_ptr<Index[]> stamp_;
    std::size_t rows_len_ = 0;       // the values rows_ holds
    std::size_t cap_ = 0;
    std::size_t head_ = 0;
    std::size_t size_ = 0;

    // The y events above the last point taken (yev_[yhead_..]), and the last
    // y at or before it.
    std::vector<std::pair<Index, double>> yev_;
    std::size_t yhead_ = 0;
    double base_ = std::numeric_limits<double>::quiet_NaN();
    bool have_y_ = false;
    Index y_last_ = kMin;            // the index of the last y event
    Index y_wm_ = kMin;              // the y port's watermark
    Index x_floor_ = kMin;           // no later X row comes below it
    Index last_wm_ = kMin;           // the last forwarded watermark
    bool flushed_[2] = {false, false};
};

}} // namespace screamer::dag
#endif
