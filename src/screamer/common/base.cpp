#include "screamer/common/base.h"
#include <cstddef>

namespace screamer {

void ScreamerBase::process_array_no_stride(double* result_data, const double* input_data, size_t size) {
    for (size_t i = 0; i < size; i++) {
        result_data[i] = process_scalar(input_data[i]);
    }
}

void ScreamerBase::process_array_stride(
    double* result_data,
    size_t result_stride,
    const double* input_data,
    size_t input_stride,
    size_t size
) {
    size_t result_start = 0;
    size_t input_start = 0;

    for (size_t i = 0; i < size; i++) {
        result_data[result_start] = process_scalar(input_data[input_start]);
        result_start += result_stride;
        input_start += input_stride;
    }
}

}  // namespace screamer
