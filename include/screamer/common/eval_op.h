#ifndef SCREAMER_EVAL_OP_H
#define SCREAMER_EVAL_OP_H

#include <cstddef>

namespace screamer {

// The uniform op interface the DAG engine holds for ANY functor, regardless of
// arity. ScreamerBase (1->1) and FunctorBase<_,N,M> (N->M) both implement it in
// terms of their existing process_scalar/call. eval() processes exactly one
// event: it reads n_in() inputs from `in` and writes n_out() outputs to `out`.
struct EvalOp {
    virtual ~EvalOp() = default;
    virtual std::size_t n_in() const = 0;
    virtual std::size_t n_out() const = 0;
    virtual void eval(const double* in, double* out) = 0;
    virtual void reset() = 0;
};

} // namespace screamer
#endif
