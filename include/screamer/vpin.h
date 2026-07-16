#ifndef SCREAMER_VPIN_H
#define SCREAMER_VPIN_H

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include "screamer/common/functor_base.h"
#include "screamer/common/float_info.h"

namespace screamer {

    // VPIN - Volume-Synchronized Probability of Informed Trading (Easley,
    // Lopez de Prado, O'Hara 2012): a measure of order-flow toxicity. Trades are
    // packed into equal-volume buckets of size `bucket_volume` (the volume
    // clock); a trade that straddles a bucket boundary is split proportionally
    // across the two buckets. Each closed bucket contributes its absolute order
    // imbalance |V_buy - V_sell|, and VPIN is the mean of that imbalance over the
    // last `n_buckets` buckets, normalized by the bucket volume:
    //     vpin = mean(|V_buy - V_sell|) / bucket_volume,   in [0, 1].
    // A high value means flow is one-sided (toxic / informed). The output is NaN
    // until `n_buckets` buckets have closed (warmup); per nan_policy "ignore" a
    // NaN on either input yields NaN with the state left untouched.
    class VPIN : public FunctorBase<VPIN, 2, 1> {
    public:
        VPIN(double bucket_volume = 1.0, int n_buckets = 50)
            : bucket_volume_(bucket_volume), n_buckets_(n_buckets),
              imbalances_(n_buckets_ > 0 ? n_buckets_ : 1, 0.0)
        {
            if (bucket_volume_ <= 0.0) {
                throw std::invalid_argument("bucket_volume must be positive.");
            }
            if (n_buckets_ < 1) {
                throw std::invalid_argument("n_buckets must be 1 or more.");
            }
            reset();
        }

        void reset() override {
            cur_buy_ = 0.0;
            cur_sell_ = 0.0;
            std::fill(imbalances_.begin(), imbalances_.end(), 0.0);
            head_ = 0;
            filled_ = 0;
            sum_ = 0.0;
        }

        ResultTuple call(const InputArray& inputs) override {
            double buy = inputs[0];
            double sell = inputs[1];
            if (isnan2(buy) || isnan2(sell)) {
                return std::numeric_limits<double>::quiet_NaN();   // ignore
            }
            // Volumes are non-negative; clamp defensively so the volume clock and
            // the [0, 1] range hold even on a stray negative input.
            if (buy < 0.0) buy = 0.0;
            if (sell < 0.0) sell = 0.0;

            // Pack the trade's volume into buckets, splitting across boundaries.
            double total = buy + sell;
            while (total > 0.0) {
                const double space = bucket_volume_ - (cur_buy_ + cur_sell_);
                if (total < space) {                 // fits in the current bucket
                    cur_buy_ += buy;
                    cur_sell_ += sell;
                    break;
                }
                // Fill the bucket with a proportional slice of the trade, close
                // it, and carry the remainder into the next bucket.
                const double frac = space / total;
                const double add_buy = buy * frac;
                const double add_sell = sell * frac;
                cur_buy_ += add_buy;
                cur_sell_ += add_sell;
                close_bucket();
                buy -= add_buy;
                sell -= add_sell;
                total -= space;
            }
            return current_vpin();
        }

    private:
        void close_bucket() {
            const double imb = std::abs(cur_buy_ - cur_sell_);
            sum_ -= imbalances_[head_];              // evict the oldest
            imbalances_[head_] = imb;
            sum_ += imb;
            head_ = (head_ + 1) % n_buckets_;
            if (filled_ < n_buckets_) filled_++;
            cur_buy_ = 0.0;
            cur_sell_ = 0.0;
        }

        double current_vpin() const {
            if (filled_ < n_buckets_) {
                return std::numeric_limits<double>::quiet_NaN();   // warmup
            }
            return (sum_ / n_buckets_) / bucket_volume_;
        }

        double bucket_volume_;
        int n_buckets_;
        std::vector<double> imbalances_;   // ring of the last n bucket imbalances
        double cur_buy_ = 0.0;
        double cur_sell_ = 0.0;
        int head_ = 0;
        int filled_ = 0;
        double sum_ = 0.0;                 // running sum of the ring
    };

} // namespace screamer

#endif // SCREAMER_VPIN_H
