#include "screamer/common/dispatch.h"
#include "screamer/common/lazy_eval_iterator.h"
#include <nanobind/stl/vector.h>
#include <vector>

namespace screamer {

bool is_dag_node(const nb::object& obj) {
    // A graph node is a Python class, so a built-in type (float, list,
    // ndarray, a NumPy scalar) is never one and skips the attribute lookup.
    if (!(PyType_GetFlags(Py_TYPE(obj.ptr())) & Py_TPFLAGS_HEAPTYPE)) return false;
    return nb::hasattr(obj, "is_node") &&
           nb::cast<bool>(obj.attr("is_node")) == true;
}

nb::object make_dag_functor_node(nb::object self, nb::object args_tuple) {
    nb::object mod = nb::module_::import_("screamer.dag");
    return mod.attr("make_functor_node")(self, args_tuple);
}

nb::object screamer_call(ScreamerBase& self, nb::object raw) {
    nb::object obj = normalize_input(raw, self);
    if (can_cast_to_double(obj)) {
        double value;
        // An int too large for a double fails here; name it, not a bad_cast.
        if (!nb::try_cast<double>(obj, value)) throw_unsupported_input(nb::find(self), obj);
        return nb::float_(self.process_scalar(value));
    }

    if (is_async_iterable(obj))
        return nb::cast(LazyAsyncIterator(nb::find(self), { obj }));

    reject_array_tuple(obj);
    if (is_event_list(obj, false))      // a list of asset events
        return collect_events(nb::find(self), { obj });

    if (nb::isinstance<nb::list>(obj) || nb::isinstance<nb::tuple>(obj)) {
        EagerRun<ScreamerBase> run(self);
        nb::list out;
        std::ptrdiff_t k = 0;
        for (nb::handle item : obj) {
            double d;
            if (!nb::try_cast<double>(item, d))
                throw_not_a_number(op_display_name(nb::find(self)), item, where_text("event", k));
            out.append(nb::float_(self.process_scalar(d)));
            ++k;
        }
        return out;
    }

    if (detail::is_ndarray(obj)) {
        // Container/rank preservation (Rule A): an ndarray input returns an
        // ndarray of the same shape, whatever its length. A length-1 array is a
        // time series of one, not a scalar; only an actual scalar returns a
        // scalar.
        nb::ndarray<nb::ro> arr = nb::cast<nb::ndarray<nb::ro>>(obj);
        if (arr.ndim() == 0) {
            // Rank 0: a 0-d array carries a single sample with no time axis, so
            // it behaves like a scalar - one event in, one scalar out.
            nb::dlpack::dtype dt = arr.dtype();
            return nb::float_(self.process_scalar(detail::load_elem(arr.data(), 0, dt)));
        }
        return process_python_array(self, arr);
    }

    if (detail::is_unsupported_dtype_array(obj)) {
        // A numpy array whose dtype nanobind's nb::ndarray<nb::ro> cannot represent
        // (e.g. longdouble/float128). Coerce to a contiguous float64 array so
        // the normal ndarray path handles it, instead of falling through to the
        // iterable branch below. Checked after is_ndarray so no supported dtype
        // pays for this.
        nb::ndarray<nb::ro> arr = nb::cast<nb::ndarray<nb::ro>>(detail::coerce_to_f64(obj));
        return process_python_array(self, arr);
    }

    if (nb::hasattr(obj, "__iter__")) {
        std::vector<nb::object> sources{ obj };      // a single iterable of scalars (n_in==1)
        return nb::cast(LazyEvalIterator(nb::find(self), std::move(sources)));
    }

    if (is_dag_node(obj)) {
        nb::object self_obj = nb::find(self);
        return make_dag_functor_node(self_obj, nb::make_tuple(obj));
    }

    throw_unsupported_input(nb::find(self), obj);
}

nb::object process_python_array(ScreamerBase& self, nb::ndarray<nb::ro> input) {
    size_t nd = input.ndim();

    std::vector<size_t> shape(nd);
    size_t total = 1;
    for (size_t i = 0; i < nd; ++i) { shape[i] = input.shape(i); total *= shape[i]; }

    // Materialise a C-contiguous double view of the input BEFORE allocating the
    // output buffer. read_contig_double / load_elem can throw on an
    // unsupported dtype; doing this first means no raw `new`-owned output
    // buffer is ever live across a throwing call, so nothing leaks.
    nb::dlpack::dtype dt = input.dtype();
    bool f64 = (dt.code == (uint8_t) nb::dlpack::dtype_code::Float && dt.bits == 64);
    detail::ContigDouble tmp;
    const double* in = nullptr;
    if (total > 0) {
        if (f64 && detail::is_c_contiguous(input)) {
            in = (const double*) input.data();
        } else {
            tmp = detail::read_contig_double(input);
            in = tmp.data.data();
        }
    }

    auto out = detail::new_output(total);
    if (total == 0) {
        return detail::make_owned_array(std::move(out), shape::consume_unit_field(shape));
    }

    size_t size = shape[0];
    size_t rest = total / size;

    // One column per asset; for a 1-D input the single column has stride 1.
    const std::size_t stride = rest;
    {
        EagerRun<ScreamerBase> run(self);   // resets the op afterwards, on a throw too
        run_columns([&self](const double* const* x, const std::ptrdiff_t* xs, std::size_t T,
                            double* y, std::ptrdiff_t ys) { self.eval_column(x, xs, T, y, ys); },
                    1, 1, size, rest, &in, &stride, [](std::size_t, std::size_t c) { return c; },
                    out.get(), false);
    }
    return detail::make_owned_array(std::move(out), shape::consume_unit_field(shape));
}

}  // namespace screamer
