#ifndef SCREAMER_CLIP_H
#define SCREAMER_CLIP_H

#include <limits>
#include "screamer/common/base.h"
#include <algorithm>
namespace screamer {

    class Clip : public ScreamerBase {
    public:
        Clip(std::optional<double> lower = std::nullopt, std::optional<double> upper = std::nullopt)
            : lower_(lower), upper_(upper) {}

    private:
        double process_scalar(double newValue) override {
            if (lower_.has_value() && newValue < lower_.value()) {
                newValue = lower_.value();
            }
            if (upper_.has_value() && newValue > upper_.value()) {
                newValue = upper_.value();
            }
            return newValue;
        }

        void process_array_no_stride(double* y, const double* x, size_t size) override {
            const double lower_bound = lower_.value_or(std::numeric_limits<double>::lowest());
            const double upper_bound = upper_.value_or(std::numeric_limits<double>::max());

            // std::min(std::max(...)) is branchless and matches numpy's
            // minimum(maximum(x, lo), hi), so the compiler vectorizes it to SIMD
            // min/max. std::clamp's three-way branch does not vectorize. NaN is
            // preserved either way (the comparisons are false for NaN).
            for (size_t i = 0; i < size; ++i) {
                y[i] = std::min(std::max(x[i], lower_bound), upper_bound);
            }
        }

        void process_array_stride(double* y, size_t dyi, const double* x, size_t dxi, size_t size) override {
            size_t yi = 0;
            size_t xi = 0;

            double lower_bound = lower_.value_or(std::numeric_limits<double>::lowest());
            double upper_bound = upper_.value_or(std::numeric_limits<double>::max());

            for (size_t i = 0; i < size; i++) {
                y[yi] = std::min(std::max(x[xi], lower_bound), upper_bound);
                xi += dxi;
                yi += dyi;
            }
        }

    private:
        std::optional<double> lower_;
        std::optional<double> upper_;
    };

} // namespace screamer

#endif // SCREAMER_CLIP_H
