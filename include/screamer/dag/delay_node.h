#ifndef SCREAMER_DAG_DELAY_NODE_H
#define SCREAMER_DAG_DELAY_NODE_H

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include "screamer/common/checked_arith.h"
#include "screamer/dag/frame.h"
#include <cstddef>

namespace screamer { namespace dag {

// Stateless index re-stamp. Forwards each frame with index shifted by `duration`,
// values untouched. Lossless, 1:1, order-preserving (a constant positive shift keeps
// events monotonic), no warmup. When this node feeds a CombineLatest merge inside a
// Pipeline, the merge holds future-dated frames in a bounded reorder buffer and
// releases them in index order as watermarks advance. See Delay.md for details.
template <class Index>
class DelayNode : public Sink<Index> {
public:
    DelayNode(std::int64_t duration, Sink<Index>& downstream)
        : duration_(duration), downstream_(downstream) {}

    // A delayed index outside the int64 range raises before the event goes on.
    void push(const Frame<Index>& f) override {
        std::int64_t k;
        if (!screamer::checked_add(static_cast<std::int64_t>(f.index), duration_, k))
            throw std::invalid_argument(
                "a delay by duration=" + std::to_string(duration_) + " moves index " +
                std::to_string(f.index) + " outside the int64 range");
        Frame<Index> out{ static_cast<Index>(k), f.values, f.width };
        downstream_.push(out);
    }

    void flush() override { downstream_.flush(); }
    // The watermark moves with the rows; it saturates at the int64 range (the
    // end-of-input watermark stays the int64 maximum).
    void on_watermark(Index w) override {
        std::int64_t out = screamer::saturating_add(static_cast<std::int64_t>(w), duration_);
        if (w == std::numeric_limits<Index>::max()) out = w;
        downstream_.on_watermark(static_cast<Index>(out));
    }
    void reset() override {}                      // stateless

    std::size_t n_in()  const override { return 1; }
    std::size_t n_out() const override { return 1; }

private:
    std::int64_t duration_;
    Sink<Index>& downstream_;
};

}} // namespace screamer::dag

#endif // SCREAMER_DAG_DELAY_NODE_H
