#ifndef SCREAMER_ROW_OP_H
#define SCREAMER_ROW_OP_H

#include <cstddef>
#include <stdexcept>
#include "screamer/common/eval_op.h"

namespace screamer {

// RowOp: the base of operators whose event is one row of K assets, such as
// the cross-sectional operators and the portfolio backtest engines. An event
// holds `fields_in` values per asset and returns `fields_out` values per asset.
//
// K is a property of the data, not of the operator: it is fixed by the first
// row after reset(), and a later row of another width is an error, so a
// changed universe is never silently mixed with the old one. Assets that are
// not available on a row (not yet listed, halted) are NaN.
//
// Layout of one event: the input is field-major, `fields_in` blocks of K
// values (all assets' first field, then all assets' second field, ...), which
// is a row of each of the operator's (T, K) inputs laid end to end. The output
// is asset-major, K blocks of `fields_out` values, which is one row of a
// (T, K, fields_out) result. With one field in and out both are one row of K.
class RowOp : public EvalOp {
public:
    explicit RowOp(std::size_t fields_in = 1, std::size_t fields_out = 1)
        : fields_in_(fields_in), fields_out_(fields_out) {}

    std::size_t fields_in() const { return fields_in_; }
    std::size_t fields_out() const { return fields_out_; }
    std::size_t width() const { return width_; }

    std::size_t n_in() const override { return fields_in_ * width_; }
    std::size_t n_out() const override { return fields_out_ * width_; }

    void eval(const double* in, double* out) override { call_row(in, out, width_); }

    void reset() override {
        width_ = 0;
        reset_state();
    }

    // One event of k assets: read fields_in * k values, write k * fields_out.
    void call_row(const double* in, double* out, std::size_t k) {
        if (k == 0)
            throw std::invalid_argument("a row needs at least one asset.");
        if (width_ != 0 && k != width_)
            throw std::invalid_argument(
                "the number of assets changed; call reset() before a new universe.");
        if (width_ == 0) {
            width_ = k;
            on_width(k);
        }
        map_row(in, out, k);
    }

protected:
    virtual void map_row(const double* in, double* out, std::size_t k) = 0;
    // Called once per reset, when the first row fixes the number of assets.
    virtual void on_width(std::size_t /*k*/) {}
    virtual void reset_state() {}

private:
    std::size_t fields_in_, fields_out_;
    std::size_t width_ = 0;
};

} // namespace screamer
#endif
