#ifndef SCREAMER_COMMON_VAR_ARITY_DISPATCH_H
#define SCREAMER_COMMON_VAR_ARITY_DISPATCH_H

// The Python call of a VarArityOp (var_arity_op.h): an operator whose numbers
// of inputs and outputs are settings. It runs the N-input dispatch of
// dispatch.h on a VarArityView, whose kN == 0 and kM == 0 make the dispatch
// read the counts from the instance: every input form (numbers, arrays,
// broadcasting, packed arrays, lists, iterators, async iterators, Pipeline
// nodes) behaves as for a FunctorBase operator of those counts. The dispatch
// is instantiated once for all VarArityOp operators.

#include <array>
#include <cstddef>
#include <utility>
#include <nanobind/nanobind.h>
#include "screamer/common/dispatch.h"
#include "screamer/common/var_arity_op.h"

namespace screamer {
namespace detail {
// The outputs of one event of a VarArityOp, returned to Python as a tuple
// (a float for one output).
struct VarRow {
    std::array<double, kMaxVarArityOutputs> v;
    std::size_t n = 0;
};
}  // namespace detail
}  // namespace screamer

namespace nanobind {
namespace detail {
template <>
struct type_caster<screamer::detail::VarRow> {
    NB_TYPE_CASTER(screamer::detail::VarRow, const_name("float | tuple[float, ...]"))

    bool from_python(handle, uint8_t, cleanup_list*) noexcept { return false; }

    // One output is a float, as a one-output operator returns.
    static handle from_cpp(const screamer::detail::VarRow& r, rv_policy, cleanup_list*) noexcept {
        if (r.n == 1) return PyFloat_FromDouble(r.v[0]);
        PyObject* t = PyTuple_New(static_cast<Py_ssize_t>(r.n));
        if (!t) return handle();
        for (std::size_t i = 0; i < r.n; ++i) {
            PyObject* f = PyFloat_FromDouble(r.v[i]);
            if (!f) {
                Py_DECREF(t);
                return handle();
            }
            PyTuple_SetItem(t, static_cast<Py_ssize_t>(i), f);   // steals f
        }
        return t;
    }
};
}  // namespace detail
}  // namespace nanobind

namespace screamer {
namespace detail {

// A view of a VarArityOp for the dispatch of dispatch.h. kN == 0 and kM == 0
// say that the input and output counts are the instance's n_in() and n_out();
// an event holds at most kMaxColumnInputs inputs.
struct VarArityView {
    static constexpr std::size_t kN = 0;
    static constexpr std::size_t kM = 0;
    using InputArray = std::array<double, kMaxColumnInputs>;
    using ResultTuple = VarRow;

    EvalOp& op;
    nb::handle owner;

    ResultTuple call(const InputArray& in) {
        VarRow r;
        r.n = op.n_out();
        op.eval(in.data(), r.v.data());
        return r;
    }
    void reset() { op.reset(); }
    std::size_t n_in() const { return op.n_in(); }
    std::size_t n_out() const { return op.n_out(); }
    bool tight_column() const { return op.tight_column(); }
    void eval_column(const double* const* in, const std::ptrdiff_t* in_stride, std::size_t T,
                     double* out, std::ptrdiff_t out_stride) {
        op.eval_column(in, in_stride, T, out, out_stride);
    }
    nb::object py_owner() const { return nb::borrow(owner); }
};

// The Python call of any VarArityOp, by its base.
inline nb::object var_arity_call_impl(EvalOp& op, nb::handle owner, nb::args args) {
    VarArityView view{op, owner};
    return functor_call<VarArityView>(view, std::move(args));
}

}  // namespace detail

// __call__ of a VarArityOp operator.
template <class Op>
nb::object var_arity_call(Op& self, nb::args args) {
    nb::object owner = nb::find(self);
    return detail::var_arity_call_impl(self, owner, std::move(args));
}

}  // namespace screamer
#endif
