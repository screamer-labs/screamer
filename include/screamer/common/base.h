#ifndef SCREAMER_BASE_H
#define SCREAMER_BASE_H

#include <cstddef>
#include "screamer/common/eval_op.h"

namespace screamer {

class ScreamerBase : public EvalOp {
public:
    virtual ~ScreamerBase() = default;

    void reset() override {}
    std::size_t n_in() const override { return 1; }
    std::size_t n_out() const override { return 1; }
    void eval(const double* in, double* out) override { out[0] = process_scalar(in[0]); }

    virtual double process_scalar(double value) = 0;

    virtual void process_array_no_stride(double* result_data, const double* input_data, size_t size);
    virtual void process_array_stride(
        double* result_data, size_t result_stride,
        const double* input_data, size_t input_stride, size_t size);
};

}  // namespace screamer
#endif
