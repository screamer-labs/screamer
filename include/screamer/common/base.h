#ifndef SCREAMER_BASE_H
#define SCREAMER_BASE_H

#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <iostream>
#include <sstream>
#include "screamer/common/cast_double.h"
#include "screamer/common/async_generator.h"
#include "screamer/common/eval_op.h"

namespace py = pybind11;

namespace screamer {

// Returns true if obj is a screamer.dag.Node (duck-typed: has is_node True).
bool is_dag_node(const py::object& obj);
// Build a graph node from a callable `self` and its argument objects.
py::object make_dag_functor_node(py::object self, py::object args_tuple);

class ScreamerBase : public EvalOp {
public:
    virtual ~ScreamerBase() = default;

    virtual void reset() {}

    std::size_t n_in() const override { return 1; }
    std::size_t n_out() const override { return 1; }
    void eval(const double* in, double* out) override { out[0] = process_scalar(in[0]); }

    py::object operator()(py::object obj);

    virtual double process_scalar(double value) = 0;

    virtual void process_array_no_stride(double* result_data, const double* input_data, size_t size);

    virtual void process_array_stride(
        double* result_data, 
        size_t result_stride,
        const double* input_data, 
        size_t input_stride,
        size_t size
    );

protected:
    py::array_t<double> process_python_array(py::array_t<double> input_array);
};

} // namespace screamer

#endif // SCREAMER_BASE_H
