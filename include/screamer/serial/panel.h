#ifndef SCREAMER_SERIAL_PANEL_H
#define SCREAMER_SERIAL_PANEL_H

// Panel: the per-column state of an operator resumed on a (T, K) array. Each
// column runs on its own copy of the operator (EvalOp::clone_fresh), kept
// between resume calls and saved as the state's "columns" list. EvalOp holds
// one in panel_ (common/eval_op.h has the rules).

#include <memory>
#include <vector>

namespace screamer {

struct EvalOp;

namespace serial {

struct Panel {
    std::vector<std::unique_ptr<EvalOp>> cols;
};

}  // namespace serial
}  // namespace screamer

#endif  // SCREAMER_SERIAL_PANEL_H
