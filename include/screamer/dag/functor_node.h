#ifndef SCREAMER_DAG_FUNCTOR_NODE_H
#define SCREAMER_DAG_FUNCTOR_NODE_H

#include <stdexcept>
#include <string>
#include <vector>
#include "screamer/common/eval_op.h"
#include "screamer/dag/frame.h"
#include "screamer/dag/checked_index.h"   // SCREAMER_NOINLINE
#include <cstddef>

namespace screamer { namespace dag {

// Drives exactly one EvalOp. On each frame it evaluates op into its OWN reused
// output buffer and emits a frame downstream. Shape-preserving: index passes
// through; output width is op.n_out().
template <class Index>
class FunctorNode : public Sink<Index> {
public:
    FunctorNode(EvalOp& op, Sink<Index>& downstream)
        : op_(op), downstream_(downstream), n_in_(op.n_in()), out_(op.n_out()) {}

    void push(const Frame<Index>& f) override {
        if (f.width != n_in_) throw_width(f.width);
        op_.eval(f.values, out_.data());
        downstream_.push(Frame<Index>{f.index, out_.data(), out_.size()});
    }

    void flush() override { downstream_.flush(); }
    void on_watermark(Index w) override { downstream_.on_watermark(w); }

    void reset() override { op_.reset(); }

    std::size_t n_in()  const override { return n_in_; }
    std::size_t n_out() const override { return op_.n_out(); }

private:
    // Out of line, so the per-event push keeps its size.
    [[noreturn]] SCREAMER_NOINLINE void throw_width(std::size_t width) const {
        throw std::invalid_argument(
            "an operator takes " + std::to_string(n_in_) +
            " column(s), but its input has " + std::to_string(width));
    }

    EvalOp& op_;
    Sink<Index>& downstream_;
    // The op's widths are fixed at construction (a row operator, whose width
    // follows its data, is refused in a Pipeline), so the per-event width
    // check reads a member instead of a virtual call.
    const std::size_t n_in_;
    std::vector<double> out_;   // reused every event; zero per-event allocation
};

}} // namespace screamer::dag
#endif
