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
#include "screamer/dag/max_pending.h"
#include "screamer/dag/resettable.h"

namespace screamer { namespace dag {

// Pairs the features of a row with the target `count` rows later. Port 0
// takes the rows of X (any width), port 1 the events of y (one value), on
// one clock: the k-th X event and the k-th y event have the same index. The
// row emitted for event t is [X of t..., y of t + count], stamped with the
// index of t, once event t + count has arrived. flush() emits the rows still
// waiting, in order, with a NaN target.
//
// An index at which only one of X and y has an event raises, as soon as it
// is known: the other input has passed it (a later event or a watermark
// above it), or has ended. While one merge feeds the graph in index order,
// an event on one port bounds the other port as well (the relations of a
// clock resample, set by set_ports), so a batch run raises at the first
// event past the gap.
//
// The forwarded watermark is the index of the oldest waiting row, else the
// lowest index a later pair can have.
//
// The ring holds min(count, rows seen) rows [X..., y] with their indexes: it
// grows geometrically during the warm-up, so a large count costs memory only
// as events arrive, and it does not reallocate once full. A row's last value
// is scratch: the emitted target is written there. An X or y event waits for
// its partner in a queue, at most `max_pending` events (one, in a batch run).
template <class Index>
class LeadRowsNode : public Resettable {
public:
    LeadRowsNode(std::size_t count, Sink<Index>& downstream,
                 std::size_t max_pending = kDefaultMaxPending)
        : count_(count), downstream_(downstream), max_pending_(max_pending) {
        ports_.reserve(2);
        for (std::size_t i = 0; i < 2; ++i) ports_.emplace_back(*this, i);
    }

    // Non-movable/copyable: the ports hold a reference back to this node.
    LeadRowsNode(const LeadRowsNode&) = delete;
    LeadRowsNode& operator=(const LeadRowsNode&) = delete;
    LeadRowsNode(LeadRowsNode&&) = delete;
    LeadRowsNode& operator=(LeadRowsNode&&) = delete;

    Sink<Index>& port(std::size_t i) { return ports_[i]; }

    // How the ports relate to the merge of the graph's inputs (as for a clock
    // resample): `in_order`, the port never comes below an index the merge
    // has passed; `never_ahead`, the port never comes above the merge.
    void set_ports(bool x_in_order, bool x_never_ahead, bool y_in_order, bool y_never_ahead) {
        y_bounds_x_ = x_in_order && y_never_ahead;
        x_bounds_y_ = y_in_order && x_never_ahead;
    }

    // Whether one merge feeds the graph's inputs in index order.
    void set_merge_order(bool on) { merge_order_ = on; }

    void reset() override {
        xw_ = 0;
        empty();
        xq_.clear();
        xqi_.clear();
        xhead_ = 0;
        yq_.clear();
        yqi_.clear();
        yhead_ = 0;
        lo_[0] = lo_[1] = kMin;
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
                throw std::invalid_argument("a lead takes X rows of at least one value");
            xw_ = f.width;
        } else if (f.width != xw_) {
            throw std::invalid_argument("a lead takes X rows of " + std::to_string(xw_) +
                                        " values, got " + std::to_string(f.width));
        }
        try {
            xq_.insert(xq_.end(), f.values, f.values + xw_);
            xqi_.push_back(f.index);
        } catch (...) {
            xq_.resize(xqi_.size() * xw_);
            if (first) xw_ = 0;
            throw;
        }
        bound(0, f.index);
        match();
    }

    void on_y(const Frame<Index>& f) {
        if (f.width != 1)
            throw std::invalid_argument("a lead takes y events of one value, got " +
                                        std::to_string(f.width));
        yq_.push_back(f.values[0]);
        try {
            yqi_.push_back(f.index);
        } catch (...) {
            yq_.pop_back();
            throw;
        }
        bound(1, f.index);
        match();
    }

    // An event at `index` on `port`: no later event of that port comes below
    // it, and in a merge-ordered run none of the other port either.
    void bound(std::size_t port, Index index) {
        if (lo_[port] < index) lo_[port] = index;
        if (merge_order_ && (port == 0 ? x_bounds_y_ : y_bounds_x_) && lo_[1 - port] < index)
            lo_[1 - port] = index;
    }

    void on_watermark(std::size_t port, Index w) {
        if (lo_[port] < w) lo_[port] = w;
        match();
        Index out;
        if (filled_ != 0) {
            out = index_[filled_ == count_ ? pos_ : 0];
        } else {
            const Index x = xhead_ < xqi_.size() ? xqi_[xhead_] : lo_[0];
            const Index y = yhead_ < yqi_.size() ? yqi_[yhead_] : lo_[1];
            out = std::max(x, y);                    // a pair needs both
        }
        if (out > last_wm_) {
            last_wm_ = out;
            downstream_.on_watermark(out);
        }
    }

    // Called once per port at end of input; the second port drains the
    // waiting rows and flushes downstream once.
    void on_flush(std::size_t port) {
        lo_[port] = kMax;
        flushed_[port] = true;
        match();
        if (!(flushed_[0] && flushed_[1])) return;
        // Oldest first; each emitted row leaves the ring at once, so a
        // downstream that rejects one does not see the earlier ones again.
        const std::size_t n = index_.size();          // the ring's rows
        if (!draining_ && filled_ != count_) pos_ = 0; // warm-up: the oldest is row 0
        draining_ = true;
        while (filled_ != 0) {
            double* slot = ring_.data() + pos_ * (xw_ + 1);
            slot[xw_] = std::numeric_limits<double>::quiet_NaN();
            downstream_.push(Frame<Index>{index_[pos_], slot, xw_ + 1});
            if (++pos_ == n) pos_ = 0;
            --filled_;
        }
        empty();
        downstream_.flush();
        flushed_[0] = flushed_[1] = false;            // re-armed: a repeat flush emits nothing
    }

    // Pairs the queued X and y events, oldest first; raises for an index
    // that only one of them has.
    void match() {
        while (xhead_ < xqi_.size() && yhead_ < yqi_.size()) {
            const Index tx = xqi_[xhead_], ty = yqi_[yhead_];
            if (tx != ty) throw_one_clock(tx < ty ? 0 : 1, std::min(tx, ty));
            pair(tx, xq_.data() + xhead_ * xw_, yq_[yhead_]);
            ++xhead_;
            ++yhead_;
        }
        if (xhead_ < xqi_.size() && lo_[1] > xqi_[xhead_]) throw_one_clock(0, xqi_[xhead_]);
        if (yhead_ < yqi_.size() && lo_[0] > yqi_[yhead_]) throw_one_clock(1, yqi_[yhead_]);
        compact();
        if (xqi_.size() - xhead_ > max_pending_ || yqi_.size() - yhead_ > max_pending_)
            throw std::runtime_error(
                "forecast_pairs: one of X and y is more than " + std::to_string(max_pending_) +
                " events ahead of the other; a stream is stalled. Call advance(now) or "
                "check the feed.");
    }

    [[noreturn]] static void throw_one_clock(std::size_t port, Index index) {
        throw std::invalid_argument(
            std::string("forecast_pairs: count= with index=\"features\" needs X and y on one "
                        "clock; at index ") + std::to_string(index) + " only " +
            (port == 0 ? "X" : "y") + " has an event; use duration= for streams on "
            "different clocks");
    }

    // Drops the paired events from the queues; their storage is reused.
    void compact() {
        if (xhead_ == xqi_.size()) { xq_.clear(); xqi_.clear(); xhead_ = 0; }
        else if (xhead_ > 64 && xhead_ * 2 > xqi_.size()) {
            xq_.erase(xq_.begin(), xq_.begin() + static_cast<std::ptrdiff_t>(xhead_ * xw_));
            xqi_.erase(xqi_.begin(), xqi_.begin() + static_cast<std::ptrdiff_t>(xhead_));
            xhead_ = 0;
        }
        if (yhead_ == yqi_.size()) { yq_.clear(); yqi_.clear(); yhead_ = 0; }
        else if (yhead_ > 64 && yhead_ * 2 > yqi_.size()) {
            yq_.erase(yq_.begin(), yq_.begin() + static_cast<std::ptrdiff_t>(yhead_));
            yqi_.erase(yqi_.begin(), yqi_.begin() + static_cast<std::ptrdiff_t>(yhead_));
            yhead_ = 0;
        }
    }

    // A matched row: emits the oldest row with this target once the ring is
    // full, and keeps this row in its place.
    void pair(Index t, const double* x, double y) {
        const std::size_t w = xw_ + 1;
        if (filled_ == count_) {                       // steady state
            double* slot = ring_.data() + pos_ * w;
            slot[xw_] = y;
            downstream_.push(Frame<Index>{index_[pos_], slot, w});
            std::memcpy(slot, x, xw_ * sizeof(double));
            index_[pos_] = t;
            if (++pos_ == count_) pos_ = 0;
            return;
        }
        keep(t, x, y);                                 // warm-up: nothing is emitted yet
    }

    void empty() {
        pos_ = 0;
        filled_ = 0;
        draining_ = false;
        ring_.clear();               // keeps its capacity for the next run
        index_.clear();
    }

    // Appends a warm-up row, growing the ring geometrically up to count rows.
    void keep(Index index, const double* x, double y) {
        const std::size_t w = xw_ + 1;
        if (filled_ == index_.capacity() || ring_.size() + w > ring_.capacity()) {
            // At least filled_ + 1 rows, since filled_ < count_.
            const std::size_t rows = std::min(std::max<std::size_t>(filled_ * 2, 64), count_);
            try {
                if (rows > SIZE_MAX / w) throw std::length_error("lead ring");
                ring_.reserve(rows * w);
                index_.reserve(rows);
            } catch (const std::bad_alloc&) {
                throw out_of_memory();
            } catch (const std::length_error&) {
                throw out_of_memory();
            }
        }
        ring_.insert(ring_.end(), x, x + xw_);
        ring_.push_back(y);
        index_.push_back(index);
        ++filled_;
    }

    std::runtime_error out_of_memory() const {
        return std::runtime_error("a lead of " + std::to_string(count_) +
                                  " events ran out of memory after " +
                                  std::to_string(filled_) + " rows");
    }

    struct Port : Sink<Index> {
        LeadRowsNode& node;
        std::size_t idx;
        Port(LeadRowsNode& n, std::size_t i) : node(n), idx(i) {}
        void push(const Frame<Index>& f) override {
            if (idx == 0) node.on_x(f); else node.on_y(f);
        }
        void flush() override { node.on_flush(idx); }
        void on_watermark(Index w) override { node.on_watermark(idx, w); }
    };
    friend struct Port;

    std::size_t count_;
    Sink<Index>& downstream_;
    std::size_t max_pending_;
    std::vector<Port> ports_;
    bool y_bounds_x_ = false;        // set_ports
    bool x_bounds_y_ = false;
    bool merge_order_ = false;       // set_merge_order

    // The ring of kept rows [X..., y]; once full, pos_ is the oldest slot.
    std::size_t xw_ = 0;             // 0 = not set since the last reset
    std::size_t pos_ = 0;
    std::size_t filled_ = 0;         // rows kept, up to count_
    bool draining_ = false;          // flush is emitting the rows from pos_ on
    std::vector<double> ring_;
    std::vector<Index> index_;

    // The X rows and y values waiting for their partner (from *head_ on).
    std::vector<double> xq_;
    std::vector<Index> xqi_;
    std::size_t xhead_ = 0;
    std::vector<double> yq_;
    std::vector<Index> yqi_;
    std::size_t yhead_ = 0;

    Index lo_[2] = {kMin, kMin};     // no later event of port i comes below lo_[i]
    Index last_wm_ = kMin;           // the last forwarded watermark
    bool flushed_[2] = {false, false};
};

}} // namespace screamer::dag
#endif
