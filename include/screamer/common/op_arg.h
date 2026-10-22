#ifndef SCREAMER_COMMON_OP_ARG_H
#define SCREAMER_COMMON_OP_ARG_H

// OpArg: an operator given as a setting of another operator, such as the op
// that Segment runs. `op` is the operator; `keep` owns it (or whatever owns
// it), so the operator lives as long as the OpArg and every copy of `keep`.
//
// A binding builds the OpArg from its own handle to a fresh copy of the
// caller's operator (the Python object, the JavaScript Embind handle), so the
// wrapper never shares state with the caller's instance. C++ code that owns
// its operator passes it with OpArg::owning.

#include <memory>
#include "screamer/common/eval_op.h"

namespace screamer {

struct OpArg {
    EvalOp* op = nullptr;
    std::shared_ptr<void> keep;

    // An OpArg that owns `op`.
    static OpArg owning(std::unique_ptr<EvalOp> op) {
        OpArg a;
        a.op = op.get();
        a.keep = std::shared_ptr<EvalOp>(std::move(op));
        return a;
    }
};

}  // namespace screamer
#endif
