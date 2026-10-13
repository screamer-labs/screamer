#ifndef SCREAMER_DAG_DELAY_NODE_H
#define SCREAMER_DAG_DELAY_NODE_H

#include <cstdint>
#include <limits>
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

    void push(const Frame<Index>& f) override {
        Frame<Index> out{ static_cast<Index>(f.index + duration_), f.values, f.width };
        downstream_.push(out);
    }

    void flush() override { downstream_.flush(); }
    // The watermark moves with the rows; it saturates at the int64 range (the
    // end-of-input watermark stays the int64 maximum).
    void on_watermark(Index w) override {
        std::int64_t out;
        if (__builtin_add_overflow(static_cast<std::int64_t>(w), duration_, &out))
            out = duration_ > 0 ? std::numeric_limits<std::int64_t>::max()
                                : std::numeric_limits<std::int64_t>::min();
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
