#ifndef SCREAMER_REALIZED_SPREAD_H
#define SCREAMER_REALIZED_SPREAD_H

#include <limits>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include "screamer/common/functor_base.h"
#include "screamer/common/float_info.h"

namespace screamer {

    // Realized spread: the part of the effective spread that the liquidity
    // provider keeps once the price has moved. Using the quote-based trade
    // direction (a print above the mid is a buy, below is a sell),
    //     D_t = sign(price_t - mid_t),
    //     realized_t = 2 * D_{t-lag} * (price_{t-lag} - mid_t),
    // it compares a past trade's price to the mid `lag` steps later. What
    // effective spread does not keep (`effective - realized`) is the price-impact
    // / adverse-selection component. This is causal: the value at t references a
    // trade that happened `lag` steps earlier, never a future one. The first
    // `lag` samples are NaN (warmup); per nan_policy "ignore" a NaN input yields
    // NaN and does not advance the delay.
    class RealizedSpread : public FunctorBase<RealizedSpread, 2, 1> {
    public:
        RealizedSpread(int lag = 1)
            : lag_(lag), dir_(lag > 0 ? lag : 1, 0.0), price_(lag > 0 ? lag : 1, 0.0)
        {
            if (lag_ < 1) {
                throw std::invalid_argument("lag must be 1 or more.");
            }
            reset();
        }

        void reset() override {
            std::fill(dir_.begin(), dir_.end(), 0.0);
            std::fill(price_.begin(), price_.end(), 0.0);
            idx_ = 0;
            count_ = 0;
        }

        ResultTuple call(const InputArray& inputs) override {
            const double price = inputs[0];
            const double mid = inputs[1];
            if (isnan2(price) || isnan2(mid)) {
                return std::numeric_limits<double>::quiet_NaN();   // ignore
            }
            const double d = (price > mid) ? 1.0 : (price < mid ? -1.0 : 0.0);

            // Before overwriting, dir_/price_ at idx_ hold the sample from `lag`
            // steps back (idx_ == t mod lag_).
            double rs;
            if (count_ >= lag_) {
                rs = 2.0 * dir_[idx_] * (price_[idx_] - mid);
            } else {
                rs = std::numeric_limits<double>::quiet_NaN();     // warmup
            }
            dir_[idx_] = d;
            price_[idx_] = price;
            idx_ = (idx_ + 1) % lag_;
            if (count_ < lag_) count_++;
            return rs;
        }

    private:
        int lag_;
        std::vector<double> dir_;
        std::vector<double> price_;
        int idx_ = 0;
        int count_ = 0;
    };

} // namespace screamer

#endif // SCREAMER_REALIZED_SPREAD_H
