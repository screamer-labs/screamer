#ifndef SCREAMER_SERIAL_RESUME_H
#define SCREAMER_SERIAL_RESUME_H

// The operators a resume call runs its columns on, shared by the Python
// binding (common/dispatch.cpp) and the JavaScript one (wasm/shape_bindings.cpp):
// the object itself for one series, the columns of its panel for a (T, K)
// call, made on the first such call (serial/panel.h, common/eval_op.h).

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "screamer/common/eval_op.h"
#include "screamer/serial/panel.h"

namespace screamer { namespace serial {

// A resume the operator cannot run (it cannot keep per-column state): a
// TypeError in Python and JavaScript.
struct ResumeTypeError : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

struct ResumeColumns {
    std::vector<EvalOp*> ops;      // the operator of each column, in order
    std::unique_ptr<Panel> made;   // a new panel, installed by finish_resume
};

// The columns of a resume of `self` on `cols` columns; `panel` is true for a
// (T, K) call. `who` names the operator in errors. Another column count than
// the held panel's, or one series on an object with a panel, raises
// std::invalid_argument (ValueError, RangeError).
inline ResumeColumns resume_columns(EvalOp& self, bool panel, std::size_t cols, const std::string& who) {
    ResumeColumns r;
    if (!panel) {
        if (self.has_panel())
            throw std::invalid_argument("resume: object holds " + std::to_string(self.panel_columns()) +
                                        " columns, got a single series");
        r.ops.push_back(&self);
        return r;
    }
    if (self.has_panel()) {
        if (self.panel_columns() != cols)
            throw std::invalid_argument("resume: object holds " + std::to_string(self.panel_columns()) +
                                        " columns, got " + std::to_string(cols));
        for (const auto& c : self.panel_->cols) r.ops.push_back(c.get());
        return r;
    }
    if (cols == 0)
        throw std::invalid_argument(who + ".resume: a (T, 0) array has no columns to keep state for");
    r.made = std::make_unique<Panel>();
    r.made->cols.reserve(cols);
    for (std::size_t c = 0; c < cols; ++c) {
        std::unique_ptr<EvalOp> op = self.clone_fresh();
        if (!op)
            throw ResumeTypeError(who + ".resume: this operator cannot keep per-column state yet; "
                                        "resume one series at a time");
        r.ops.push_back(op.get());
        r.made->cols.push_back(std::move(op));
    }
    return r;
}

// A resume of a row operator or PortfolioReport (one object for k assets,
// no panel): an object that holds state for `held` assets (0 when it holds
// none) continues only on rows of that many. std::invalid_argument
// (ValueError, RangeError).
inline void check_resume_assets(std::size_t held, std::size_t k) {
    if (held != 0 && held != k)
        throw std::invalid_argument("resume: object holds " + std::to_string(held) + " assets, got " +
                                    std::to_string(k));
}

// After the run: a new panel becomes the object's, and the object resets. A
// throw during the run leaves the columns that ran advanced (an existing
// panel or the object itself); a new panel is installed only after the run.
inline void finish_resume(EvalOp& self, ResumeColumns& r) {
    if (!r.made) return;
    self.reset();
    self.panel_ = std::move(r.made);
}

}}  // namespace screamer::serial

#endif  // SCREAMER_SERIAL_RESUME_H
