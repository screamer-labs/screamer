#ifndef SCREAMER_DAG_DELAY_NODE_H
#define SCREAMER_DAG_DELAY_NODE_H

#include <cstdint>
#include "screamer/dag/frame.h"

namespace screamer { namespace dag {

// Stateless index re-stamp. Forwards each frame with index shifted by `duration`,
// values untouched. Lossless, 1:1, order-preserving (a constant positive shift keeps
// events monotonic), no warmup. See Delay.md for the live-merge-fusion limitation.
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
    void reset() override {}                      // stateless

    std::size_t n_in()  const override { return 1; }
    std::size_t n_out() const override { return 1; }

private:
    std::int64_t duration_;
    Sink<Index>& downstream_;
};

}} // namespace screamer::dag

#endif // SCREAMER_DAG_DELAY_NODE_H
