#ifndef SCREAMER_ROW_OP_H
#define SCREAMER_ROW_OP_H

#include <cstddef>
#include <stdexcept>
#include "screamer/common/eval_op.h"

namespace screamer {

// RowOp: the base of cross-sectional operators. An event is one row of K
// values, one per asset, and the output is a row of K values. K is a property
// of the data, not of the operator: it is fixed by the first row after
// reset(), and a later row of another width is an error, so a changed universe
// is never silently mixed with the old one. Assets that are not available on a
// row (not yet listed, halted) are NaN, and the operators leave them out of the
// cross-section and return NaN for them.
//
// A (T, K) array is T events. Derived classes implement map_row(); the
// operators in this family keep no state across rows, but the contract allows
// it (reset_state()).
class RowOp : public EvalOp {
public:
    std::size_t n_in() const override { return width_; }
    std::size_t n_out() const override { return width_; }

    void eval(const double* in, double* out) override { call_row(in, out, width_); }

    void reset() override {
        width_ = 0;
        reset_state();
    }

    // One event: read k values from `in`, write k values to `out`.
    void call_row(const double* in, double* out, std::size_t k) {
        if (k == 0)
            throw std::invalid_argument("a cross-sectional row needs at least one value.");
        if (width_ != 0 && k != width_)
            throw std::invalid_argument(
                "the number of assets changed; call reset() before a new universe.");
        width_ = k;
        map_row(in, out, k);
    }

protected:
    virtual void map_row(const double* in, double* out, std::size_t k) = 0;
    virtual void reset_state() {}

private:
    std::size_t width_ = 0;
};

} // namespace screamer
#endif
