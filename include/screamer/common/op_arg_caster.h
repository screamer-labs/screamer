#ifndef SCREAMER_COMMON_OP_ARG_CASTER_H
#define SCREAMER_COMMON_OP_ARG_CASTER_H

// The Python conversion of an OpArg (op_arg.h): an operator setting, such as
// the op of Segment(op). It accepts a screamer operator (an EvalOp built from
// the screamer namespace, which can copy itself with _fresh()) and holds a
// fresh copy of it, so the operator that takes it never shares state with the
// caller's instance. Anything else fails the conversion; the Python class
// (screamer/_functor_params.py) names the operator and the setting.

#include <memory>
#include <nanobind/nanobind.h>
#include "screamer/common/eval_op.h"
#include "screamer/common/op_arg.h"

namespace nanobind {
namespace detail {

template <>
struct type_caster<screamer::OpArg> {
    NB_TYPE_CASTER(screamer::OpArg, const_name("EvalOp"))

    bool from_python(handle src, uint32_t, cleanup_list*) noexcept {
        if (!isinstance<screamer::EvalOp>(src) || !hasattr(src, "_fresh")) return false;
        try {
            object fresh = src.attr("_fresh")();
            screamer::EvalOp* op = cast<screamer::EvalOp*>(fresh);
            value.op = op;
            // The deleter drops the copy's reference, under the GIL.
            value.keep = std::shared_ptr<void>(new object(std::move(fresh)), [](void* p) {
                gil_scoped_acquire gil;
                delete static_cast<object*>(p);
            });
            return true;
        } catch (python_error& e) {
            e.restore();
            PyErr_Clear();
            return false;
        } catch (...) {
            return false;
        }
    }

    static handle from_cpp(const screamer::OpArg&, rv_policy, cleanup_list*) noexcept {
        return none().release();
    }
};

}  // namespace detail
}  // namespace nanobind
#endif
