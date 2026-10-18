#ifndef SCREAMER_COMMON_VAR_ARITY_OP_H
#define SCREAMER_COMMON_VAR_ARITY_OP_H

// VarArityOp: the base of an operator whose number of inputs and outputs is a
// setting, fixed when it is built, such as RollingOLS (one input per
// regressor). FunctorBase fixes the counts in the type; here they are
// n_in() and n_out() of the instance. Every event still takes exactly n_in()
// values and returns n_out(), so an operator built with given settings
// behaves like a FunctorBase operator of those counts in both bindings.
//
// Derived implements
//     void step(const double* in, double* out);   // one event
//     void reset();
// eval() and the batch column loop both call step(), so batch and per-event
// results are the same.

#include <array>
#include <cstddef>
#include <stdexcept>
#include "screamer/common/column_driver.h"
#include "screamer/common/eval_op.h"
#include "screamer/common/functor_base.h"

namespace screamer {

// The most outputs a VarArityOp can have.
constexpr std::size_t kMaxVarArityOutputs = 32;

template <class Derived>
class VarArityOp : public EvalOp {
public:
    VarArityOp(std::size_t n_in, std::size_t n_out) : n_in_(n_in), n_out_(n_out) {
        if (n_in < 1 || n_in > kMaxColumnInputs || n_out < 1 || n_out > kMaxVarArityOutputs)
            throw std::invalid_argument("unsupported number of inputs or outputs.");
    }

    std::size_t n_in() const override { return n_in_; }
    std::size_t n_out() const override { return n_out_; }

    void eval(const double* in, double* out) override {
        static_cast<Derived&>(*this).Derived::step(in, out);
    }

    bool tight_column() const override { return true; }

    SCREAMER_NOINLINE
    void eval_column(const double* const* in, const std::ptrdiff_t* in_stride,
                     std::size_t T, double* out, std::ptrdiff_t out_stride) override {
        Derived& self = static_cast<Derived&>(*this);
        self.Derived::reset();
        const std::size_t n = n_in_;
        std::array<const double*, kMaxColumnInputs> p{};
        std::array<double, kMaxColumnInputs> row{};
        for (std::size_t i = 0; i < n; ++i) p[i] = in[i];
        for (std::size_t t = 0; t < T; ++t, out += out_stride) {
            for (std::size_t i = 0; i < n; ++i) {
                row[i] = *p[i];
                p[i] += in_stride[i];
            }
            SCREAMER_INLINE_CALL self.Derived::step(row.data(), out);
        }
    }

private:
    std::size_t n_in_, n_out_;
};

}  // namespace screamer
#endif
