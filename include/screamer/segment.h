#ifndef SCREAMER_SEGMENT_H
#define SCREAMER_SEGMENT_H

// Segment(op): runs `op` on each segment of a series from a fresh state.
// It takes 1 + op.n_in() inputs, (label, *op inputs), and returns op.n_out()
// outputs per row.
//
// A segment is a run of rows with the same label. At a row whose label
// differs from the last non-NaN label, Segment calls op.reset(); then, at
// every row, it evaluates op on the row's inputs and returns op's outputs.
// So a label that returns to an earlier value starts a new segment, which is
// pandas `groupby` over the runs of a sorted key.
//
// A NaN label keeps the current segment: the row is evaluated and nothing is
// reset. Rows before the first non-NaN label belong to no segment and give
// NaN without reaching op. NaN in op's inputs is op's business.
//
// The batch path finds the segment boundaries on the label column and runs
// op.eval_column once per segment (eval_column resets op first), so a column
// costs one pass over the labels plus op's own batch loop; the per-event path
// runs op.eval. Both leave the same state.

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include "screamer/common/column_driver.h"
#include "screamer/common/float_info.h"
#include "screamer/common/op_arg.h"
#include "screamer/common/row_op.h"
#include "screamer/common/var_arity_op.h"

namespace screamer {

class Segment : public VarArityOp<Segment> {
public:
    explicit Segment(OpArg op)
        : VarArityOp<Segment>(checked_n_in(op.op), checked_n_out(op.op)),
          op_(op.op), keep_(std::move(op.keep)) {}

    // An owned op, for C++ callers: Segment(std::make_unique<CumSum>()).
    explicit Segment(std::unique_ptr<EvalOp> op) : Segment(OpArg::owning(std::move(op))) {}

    // The op is held by pointer; a copy would share its state.
    Segment(const Segment&) = delete;
    Segment& operator=(const Segment&) = delete;

    void reset() override {
        op_->reset();
        started_ = false;
        label_ = 0.0;
    }

    // One row: in[0] is the label, in[1..] are op's inputs.
    void step(const double* in, double* out) {
        const double label = in[0];
        if (isnan2(label)) {
            if (!started_) {
                fill_nan(out);
                return;
            }
        } else if (!started_ || label != label_) {
            op_->reset();
            label_ = label;
            started_ = true;
        }
        op_->eval(in + 1, out);
    }

    bool tight_column() const override { return op_->tight_column(); }

    // A column: rows before the first label are NaN, then op.eval_column runs
    // once per segment. A NaN label never ends a segment.
    void eval_column(const double* const* in, const std::ptrdiff_t* in_stride,
                     std::size_t T, double* out, std::ptrdiff_t out_stride) override {
        reset();
        const std::size_t n = n_in() - 1;
        const double* label = in[0];
        const std::ptrdiff_t ls = in_stride[0];
        std::size_t t = 0;
        for (; t < T && isnan2(*label); ++t, label += ls, out += out_stride) fill_nan(out);
        std::array<const double*, kMaxColumnInputs> p{};
        for (std::size_t i = 0; i < n; ++i) p[i] = in[i + 1] + static_cast<std::ptrdiff_t>(t) * in_stride[i + 1];
        while (t < T) {
            const double cur = *label;
            const std::size_t start = t;
            for (++t, label += ls; t < T; ++t, label += ls) {
                const double l = *label;
                if (l != cur && !isnan2(l)) break;
            }
            const std::size_t len = t - start;
            op_->eval_column(p.data(), in_stride + 1, len, out, out_stride);
            for (std::size_t i = 0; i < n; ++i) p[i] += static_cast<std::ptrdiff_t>(len) * in_stride[i + 1];
            out += static_cast<std::ptrdiff_t>(len) * out_stride;
            label_ = cur;
            started_ = true;
        }
    }

private:
    static std::size_t checked_n_in(const EvalOp* op) {
        if (!op) throw std::invalid_argument("Segment: op must be an operator");
        if (dynamic_cast<const RowOp*>(op))
            throw std::invalid_argument(
                "Segment: op must take one value per input and row; a cross-sectional "
                "operator (a row of assets per event) cannot be segmented");
        const std::size_t n = op->n_in() + 1;
        if (n > kMaxColumnInputs)
            throw std::invalid_argument(
                "Segment: op takes " + std::to_string(op->n_in()) + " inputs; with the label "
                "that exceeds the limit of " + std::to_string(kMaxColumnInputs) + " inputs");
        if (op->n_out() > kMaxVarArityOutputs)
            throw std::invalid_argument(
                "Segment: op returns " + std::to_string(op->n_out()) + " outputs, more than "
                "the limit of " + std::to_string(kMaxVarArityOutputs));
        return n;
    }

    static std::size_t checked_n_out(const EvalOp* op) {
        checked_n_in(op);
        return op->n_out();
    }

    void fill_nan(double* out) const {
        const std::size_t m = n_out();
        for (std::size_t j = 0; j < m; ++j) out[j] = std::numeric_limits<double>::quiet_NaN();
    }

    EvalOp* op_;
    std::shared_ptr<void> keep_;   // owns op_ (or its binding handle)
    double label_ = 0.0;           // the current segment's label
    bool started_ = false;         // a non-NaN label has been seen
};

}  // namespace screamer
#endif
