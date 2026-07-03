#ifndef SCREAMER_DAG_FUNCTOR_NODE_H
#define SCREAMER_DAG_FUNCTOR_NODE_H

#include <stdexcept>
#include <vector>
#include "screamer/common/eval_op.h"
#include "screamer/dag/frame.h"

namespace screamer { namespace dag {

// Drives exactly one EvalOp. On each frame it evaluates op into its OWN reused
// output buffer and emits a frame downstream. Shape-preserving: key passes
// through; output width is op.n_out().
template <class Key>
class FunctorNode : public Sink<Key> {
public:
    FunctorNode(EvalOp& op, Sink<Key>& downstream)
        : op_(op), downstream_(downstream), out_(op.n_out()) {}

    void push(const Frame<Key>& f) override {
        if (f.width != op_.n_in()) {
            throw std::runtime_error(
                "dag::FunctorNode: frame width does not match op n_in");
        }
        op_.eval(f.values, out_.data());
        downstream_.push(Frame<Key>{f.key, out_.data(), out_.size()});
    }

    void flush() override { downstream_.flush(); }

private:
    EvalOp& op_;
    Sink<Key>& downstream_;
    std::vector<double> out_;   // reused every event; zero per-event allocation
};

}} // namespace screamer::dag
#endif
