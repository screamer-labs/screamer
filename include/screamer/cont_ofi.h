#ifndef SCREAMER_CONT_OFI_H
#define SCREAMER_CONT_OFI_H

#include <limits>
#include "screamer/common/functor_base.h"
#include "screamer/common/float_info.h"

namespace screamer {

    // Cont-Kukanov-Stoikov (2014) order-flow imbalance from L1 book events. For
    // each top-of-book update (bid, ask, bid_size, ask_size) versus the previous
    // one it accumulates the signed change in resting depth:
    //     e^b = bid_size * 1(bid >= prev_bid) - prev_bid_size * 1(bid <= prev_bid)
    //     e^a = ask_size * 1(ask <= prev_ask) - prev_ask_size * 1(ask >= prev_ask)
    //     OFI = e^b - e^a.
    // A bid that ticks up adds its full size (buyers stepping in), a bid that ticks
    // down removes the old size, and an unchanged bid contributes the size change;
    // the ask side is symmetric. This is the canonical order-book OFI (distinct
    // from the trade-flow `OFI`) and is a strong short-horizon price predictor.
    // The first event has no baseline and returns NaN; per nan_policy "ignore" a
    // NaN on any input yields NaN with the previous quote left untouched.
    class ContOFI : public FunctorBase<ContOFI, 4, 1> {
    public:
        // The state is a few plain values: the batch loop runs on a local
        // copy, which stays in registers (functor_base.h).
        static constexpr bool kColumnCopy = true;

        static constexpr const char* kTypeName = "ContOFI";
        static constexpr int kStateVersion = 1;

        template <class Ar>
        void settings(Ar&) {}

        // The previous quote.
        template <class Ar>
        void state(Ar& ar) {
            ar("has_prev", has_prev_);
            ar("prev_bid", prev_bid_);
            ar("prev_ask", prev_ask_);
            ar("prev_bid_size", prev_bid_size_);
            ar("prev_ask_size", prev_ask_size_);
        }

        ContOFI() = default;

        void reset() override {
            has_prev_ = false;
            prev_bid_ = 0.0;
            prev_ask_ = 0.0;
            prev_bid_size_ = 0.0;
            prev_ask_size_ = 0.0;
        }

        ResultTuple call(const InputArray& inputs) override {
            const double bid = inputs[0];
            const double ask = inputs[1];
            const double bid_size = inputs[2];
            const double ask_size = inputs[3];
            if (isnan2(bid) || isnan2(ask) || isnan2(bid_size) || isnan2(ask_size)) {
                return std::numeric_limits<double>::quiet_NaN();   // ignore
            }
            double ofi;
            if (!has_prev_) {
                ofi = std::numeric_limits<double>::quiet_NaN();    // no baseline yet
            } else {
                const double eb = (bid >= prev_bid_ ? bid_size : 0.0)
                                - (bid <= prev_bid_ ? prev_bid_size_ : 0.0);
                const double ea = (ask <= prev_ask_ ? ask_size : 0.0)
                                - (ask >= prev_ask_ ? prev_ask_size_ : 0.0);
                ofi = eb - ea;
            }
            prev_bid_ = bid;
            prev_ask_ = ask;
            prev_bid_size_ = bid_size;
            prev_ask_size_ = ask_size;
            has_prev_ = true;
            return ofi;
        }

    private:
        bool has_prev_ = false;
        double prev_bid_ = 0.0;
        double prev_ask_ = 0.0;
        double prev_bid_size_ = 0.0;
        double prev_ask_size_ = 0.0;
    };

} // namespace screamer

#endif // SCREAMER_CONT_OFI_H
