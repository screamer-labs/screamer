#ifndef SCREAMER_SERIAL_PANEL_H
#define SCREAMER_SERIAL_PANEL_H

// Panel: the per-column state of an operator resumed on a (T, K) array, or on
// a (T, *A) array with the K columns of its axes A in row-major order. Each
// column runs on its own copy of the operator (EvalOp::clone_fresh), kept
// between resume calls and saved as the state's "columns" list; `shape`
// holds A, which a later resume must give again. EvalOp holds one in panel_
// (common/eval_op.h has the rules).

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace screamer {

struct EvalOp;

namespace serial {

struct Panel {
    std::vector<std::unique_ptr<EvalOp>> cols;
    std::vector<std::size_t> shape;   // the axes after time; their product is cols.size()
};

// A shape as Python writes a tuple: "(2, 3)", "(6,)".
inline std::string shape_text(const std::vector<std::size_t>& shape) {
    std::string t = "(";
    for (std::size_t d = 0; d < shape.size(); ++d) t += (d ? ", " : "") + std::to_string(shape[d]);
    return t + (shape.size() == 1 ? ",)" : ")");
}

}  // namespace serial
}  // namespace screamer

#endif  // SCREAMER_SERIAL_PANEL_H
