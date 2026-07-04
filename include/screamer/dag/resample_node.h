#ifndef SCREAMER_DAG_RESAMPLE_NODE_H
#define SCREAMER_DAG_RESAMPLE_NODE_H

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
#include "screamer/common/float_info.h"
#include "screamer/dag/frame.h"
#include "screamer/dag/resample_params.h"

namespace screamer { namespace dag {

// Single-pass O(1) NaN-ignore accumulator. add() folds one value; emit() writes
// the reducer result. `has` marks that any event (NaN or not) fell in the bucket.
struct ResampleAccum {
    std::int64_t count = 0;
    double sum = 0.0, mn = 0.0, mx = 0.0, first = 0.0, last = 0.0;
    bool has = false;

    void reset() { count = 0; sum = 0.0; mn = mx = first = last = 0.0; has = false; }

    void add(double v) {
        has = true;
        if (screamer::isnan2(v)) return;      // ignore policy
        if (count == 0) { mn = mx = first = last = v; }
        else {
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            last = v;
        }
        sum += v;
        ++count;
    }

    void emit(ResampleAgg agg, double* out) const {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        switch (agg) {
        case ResampleAgg::First: out[0] = count ? first : nan; break;
        case ResampleAgg::Last:  out[0] = count ? last  : nan; break;
        case ResampleAgg::Min:   out[0] = count ? mn    : nan; break;
        case ResampleAgg::Max:   out[0] = count ? mx    : nan; break;
        case ResampleAgg::Sum:   out[0] = sum; break;
        case ResampleAgg::Count: out[0] = static_cast<double>(count); break;
        case ResampleAgg::Mean:  out[0] = count ? sum / static_cast<double>(count) : nan; break;
        case ResampleAgg::Ohlc:
            out[0] = count ? first : nan;
            out[1] = count ? mx    : nan;
            out[2] = count ? mn    : nan;
            out[3] = count ? last  : nan;
            break;
        }
    }
};

// Stateful windowing push-node. Buckets a width-1 stream by index-interval or event
// count, reduces each bucket with a fixed C++ reducer, and emits the bucket on the
// causal boundary (an index crossing the bucket end / the Nth event). flush() emits
// the trailing partial bucket and is idempotent (emit-then-clear).
template <class Index>
class ResampleNode : public Sink<Index> {
public:
    ResampleNode(ResampleParams p, Sink<Index>& downstream)
        : p_(p), downstream_(downstream), out_(resample_width(p.agg)) { reset(); }

    void push(const Frame<Index>& f) override {
        if (f.width != 1)
            throw std::runtime_error("dag::ResampleNode: expects a width-1 input stream");
        if (p_.mode == ResampleMode::ByIndex) push_by_index(f.index, f.values[0]);
        else                                push_by_count(f.index, f.values[0]);
    }

    void flush() override {
        if (p_.mode == ResampleMode::ByIndex) {
            if (acc_.has) emit(cur_label_);
        } else {
            if (count_in_bucket_ > 0) emit(p_.label == ResampleLabel::Left ? first_index_ : last_index_);
        }
        // idempotent: clear so a repeat flush emits nothing
        started_ = false;
        count_in_bucket_ = 0;
        acc_.reset();
        downstream_.flush();
    }

    void reset() {
        acc_.reset();
        started_ = false;
        bucket_ = 0;
        cur_label_ = Index{};
        count_in_bucket_ = 0;
        first_index_ = last_index_ = Index{};
    }

private:
    void push_by_index(Index k, double v) {
        std::int64_t nb = floordiv(static_cast<std::int64_t>(k) - p_.origin, p_.width);
        if (!started_) {
            started_ = true; bucket_ = nb; acc_.reset(); set_index_label(nb);
        } else if (nb != bucket_) {
            if (acc_.has) emit(cur_label_);
            bucket_ = nb; acc_.reset(); set_index_label(nb);
        }
        acc_.add(v);
    }

    void set_index_label(std::int64_t nb) {
        std::int64_t start = p_.origin + nb * p_.width;
        cur_label_ = static_cast<Index>(p_.label == ResampleLabel::Left ? start : start + p_.width);
    }

    void push_by_count(Index k, double v) {
        if (count_in_bucket_ == 0) first_index_ = k;
        last_index_ = k;
        acc_.add(v);
        ++count_in_bucket_;
        if (count_in_bucket_ == p_.count) {
            emit(p_.label == ResampleLabel::Left ? first_index_ : last_index_);
            acc_.reset();
            count_in_bucket_ = 0;
        }
    }

    void emit(Index label) {
        acc_.emit(p_.agg, out_.data());
        downstream_.push(Frame<Index>{label, out_.data(), out_.size()});
    }

    static std::int64_t floordiv(std::int64_t a, std::int64_t b) {
        std::int64_t q = a / b, r = a % b;
        if (r != 0 && ((r < 0) != (b < 0))) --q;
        return q;
    }

    ResampleParams p_;
    Sink<Index>& downstream_;
    std::vector<double> out_;
    ResampleAccum acc_;
    bool started_ = false;
    std::int64_t bucket_ = 0;
    Index cur_label_{};
    std::int64_t count_in_bucket_ = 0;
    Index first_index_{}, last_index_{};
};

}} // namespace screamer::dag
#endif
