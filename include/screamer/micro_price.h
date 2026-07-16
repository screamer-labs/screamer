#ifndef SCREAMER_MICRO_PRICE_H
#define SCREAMER_MICRO_PRICE_H

#include <limits>
#include "screamer/common/functor_base.h"
#include "screamer/common/float_info.h"

namespace screamer {

    // Micro-price (Stoikov 2018, first-order): the imbalance-weighted mid, a fair
    // value that leans toward the thinner side of the book,
    //     I = bid_size / (bid_size + ask_size),
    //     micro = I * ask + (1 - I) * bid.
    // When the bid queue is heavier (I -> 1) the price is pulled toward the ask
    // (upward pressure), and vice versa; a balanced book gives the plain mid. An
    // empty book (both sizes 0) also falls back to the mid. Stateless and
    // elementwise; a NaN on any input yields NaN (nan_policy: ignore). This is the
    // widely-used weighted-mid form; the full Stoikov micro-price with a
    // calibrated adjustment function is not modelled here.
    class MicroPrice : public FunctorBase<MicroPrice, 4, 1> {
    public:
        MicroPrice() = default;

        ResultTuple call(const InputArray& inputs) override {
            const double bid = inputs[0];
            const double ask = inputs[1];
            const double bid_size = inputs[2];
            const double ask_size = inputs[3];
            if (isnan2(bid) || isnan2(ask) || isnan2(bid_size) || isnan2(ask_size)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            const double denom = bid_size + ask_size;
            if (denom == 0.0) {
                return 0.5 * (bid + ask);          // empty book -> plain mid
            }
            const double imbalance = bid_size / denom;
            return imbalance * ask + (1.0 - imbalance) * bid;
        }
    };

} // namespace screamer

#endif // SCREAMER_MICRO_PRICE_H
