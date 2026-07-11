#ifndef SCREAMER_DAG_FUNCTOR_NODE_H
#define SCREAMER_DAG_FUNCTOR_NODE_H

#include <stdexcept>
#include <vector>
#include "screamer/common/eval_op.h"
#include "screamer/dag/frame.h"

namespace screamer { namespace dag {

// Drives exactly one EvalOp. On each frame it evaluates op into its OWN reused
// output buffer and emits a frame downstream. Shape-preserving: index passes
// through; output width is op.n_out().
template <class Index>
class FunctorNode : public Sink<Index> {
public:
    FunctorNode(EvalOp& op, Sink<Index>& downstream)
        : op_(op), downstream_(downstream), out_(op.n_out()) {}

    void push(const Frame<Index>& f) override {
        if (f.width != op_.n_in()) {
            throw std::runtime_error(
                "dag::FunctorNode: frame width does not match op n_in");
        }
        op_.eval(f.values, out_.data());
        downstream_.push(Frame<Index>{f.index, out_.data(), out_.size()});
    }

    void flush() override { downstream_.flush(); }

    void reset() override { op_.reset(); }

    std::size_t n_in()  const override { return op_.n_in(); }
    std::size_t n_out() const override { return op_.n_out(); }

private:
    EvalOp& op_;
    Sink<Index>& downstream_;
    std::vector<double> out_;   // reused every event; zero per-event allocation
};

}} // namespace screamer::dag
#endif
