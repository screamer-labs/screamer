#ifndef SCREAMER_ROW_OP_H
#define SCREAMER_ROW_OP_H

#include <cstddef>
#include <stdexcept>
#include <vector>
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
//
// A row reduction (`per_asset_out` false, such as CrossStd) instead returns
// `fields_out` values for the whole row, one row of a (T,) or (T, fields_out)
// result.
class RowOp : public EvalOp {
public:
    explicit RowOp(std::size_t fields_in = 1, std::size_t fields_out = 1,
                   bool per_asset_out = true)
        : fields_in_(fields_in), fields_out_(fields_out), per_asset_out_(per_asset_out) {}

    std::size_t fields_in() const { return fields_in_; }
    std::size_t fields_out() const { return fields_out_; }
    bool per_asset_out() const { return per_asset_out_; }
    std::size_t width() const { return width_; }
    // Values one event of k assets writes.
    std::size_t out_width(std::size_t k) const { return per_asset_out_ ? fields_out_ * k : fields_out_; }

    std::size_t n_in() const override { return fields_in_ * width_; }
    std::size_t n_out() const override { return width_ ? out_width(width_) : 0; }

    void eval(const double* in, double* out) override { call_row(in, out, width_); }

    // A row op's event width is fixed by its first row, so it has no column
    // form: its batch entries are call_row over rows (mapRowsInto in WASM).
    // eval_column, and so evalBatchInto, throw for a row op; they threw before
    // too, since a reset row op has no width and eval() rejects a 0-asset row.
    void eval_column(const double* const*, const std::ptrdiff_t*,
                     std::size_t, double*, std::ptrdiff_t) override {
        throw std::logic_error("a row operator runs by rows, not by columns.");
    }

    void reset() override {
        width_ = 0;
        reset_state();
    }

    // One event of k assets: read fields_in * k values, write out_width(k).
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

    // `rows` events of k assets laid end to end, the rows of a batch call.
    // Field-major rows (`packed` false) are fields_in blocks of k values, the
    // rows of the separate (T, K) inputs laid end to end. Packed rows are k
    // assets of fields_in values each, the rows of one packed (T, K, F)
    // array, F last (rule 4): each is gathered into field-major order. Both
    // layouts give the same events, so the same results bit for bit.
    void call_rows(const double* in, double* out, std::size_t rows, std::size_t k, bool packed) {
        const std::size_t fi = fields_in_, step_in = fi * k, step_out = out_width(k);
        if (!packed || fi == 1) {
            for (std::size_t r = 0; r < rows; ++r) call_row(in + r * step_in, out + r * step_out, k);
            return;
        }
        std::vector<double> event(step_in);
        for (std::size_t r = 0; r < rows; ++r) {
            const double* row = in + r * step_in;
            for (std::size_t j = 0; j < k; ++j)
                for (std::size_t f = 0; f < fi; ++f) event[f * k + j] = row[j * fi + f];
            call_row(event.data(), out + r * step_out, k);
        }
    }

protected:
    virtual void map_row(const double* in, double* out, std::size_t k) = 0;
    // Called once per reset, when the first row fixes the number of assets.
    virtual void on_width(std::size_t /*k*/) {}
    virtual void reset_state() {}

private:
    std::size_t fields_in_, fields_out_;
    bool per_asset_out_;
    std::size_t width_ = 0;
};

} // namespace screamer
#endif
