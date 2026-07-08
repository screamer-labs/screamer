#ifndef SCREAMER_DAG_RESAMPLE_GENERIC_NODE_H
#define SCREAMER_DAG_RESAMPLE_GENERIC_NODE_H

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
#include "screamer/common/eval_op.h"
#include "screamer/common/float_info.h"
#include "screamer/dag/frame.h"
#include "screamer/dag/resample_params.h"

namespace screamer { namespace dag {

// Stateful windowing push-node whose per-bucket reducer is an arbitrary screamer
// functor (an EvalOp), rather than the fixed builtin ResampleAgg. Buckets a
// width-1 stream exactly like ResampleNode (index-interval or event count), but
// each in-bar sample is fed to reducer_->eval(); the reducer's latest output is
// remembered and emitted at the causal bucket boundary, then reducer_->reset()
// starts the next bar clean. NaN-ignore matches ResampleAccum: NaN samples are
// not fed to the reducer, a bucket that saw only NaN (or no finite) samples emits
// an all-NaN row, and a bucket with any event still emits (empty buckets do not).
// flush() emits the trailing partial bucket at end-of-input and is idempotent.
template <class Index>
class GenericResampleNode : public Sink<Index> {
public:
    GenericResampleNode(ResampleParams p, Sink<Index>& downstream)
        : p_(p), downstream_(downstream), reducer_(p.reducer),
          out_(reducer_ ? reducer_->n_out() : 0) {
        if (!reducer_)
            throw std::runtime_error(
                "dag::GenericResampleNode: null reducer");
        reset();
    }

    void push(const Frame<Index>& f) override {
        if (f.width != 1)
            throw std::runtime_error(
                "dag::GenericResampleNode: expects a width-1 input stream");
        if (p_.mode == ResampleMode::ByIndex) push_by_index(f.index, f.values[0]);
        else                                  push_by_count(f.index, f.values[0]);
    }

    void flush() override {
        if (p_.mode == ResampleMode::ByIndex) {
            if (has_) emit(cur_label_);
        } else {
            if (count_in_bucket_ > 0) emit(p_.label == ResampleLabel::Left ? first_index_ : last_index_);
        }
        // idempotent: clear so a repeat flush emits nothing
        clear_bucket();
        started_ = false;
        count_in_bucket_ = 0;
        downstream_.flush();
    }

    void reset() {
        clear_bucket();
        started_ = false;
        bucket_ = 0;
        cur_label_ = Index{};
        count_in_bucket_ = 0;
        first_index_ = last_index_ = Index{};
    }

private:
    void add(double v) {
        has_ = true;                       // any event, even NaN
        if (screamer::isnan2(v)) return;   // ignore policy: do not feed NaN
        reducer_->eval(&v, out_.data());
        fed_ = true;
    }

    void clear_bucket() {
        reducer_->reset();
        has_ = false;
        fed_ = false;
    }

    void push_by_index(Index k, double v) {
        std::int64_t nb = floordiv(static_cast<std::int64_t>(k) - p_.origin, p_.width);
        if (!started_) {
            started_ = true; bucket_ = nb; clear_bucket(); set_index_label(nb);
        } else if (nb != bucket_) {
            if (has_) emit(cur_label_);
            bucket_ = nb; clear_bucket(); set_index_label(nb);
        }
        add(v);
    }

    void set_index_label(std::int64_t nb) {
        std::int64_t start = p_.origin + nb * p_.width;
        cur_label_ = static_cast<Index>(p_.label == ResampleLabel::Left ? start : start + p_.width);
    }

    void push_by_count(Index k, double v) {
        if (count_in_bucket_ == 0) first_index_ = k;
        last_index_ = k;
        add(v);
        ++count_in_bucket_;
        if (count_in_bucket_ == p_.count) {
            emit(p_.label == ResampleLabel::Left ? first_index_ : last_index_);
            clear_bucket();
            count_in_bucket_ = 0;
        }
    }

    void emit(Index label) {
        // A bucket with events but no finite sample emits an all-NaN row (matches
        // ResampleAccum, which returns NaN when its finite count is 0).
        if (!fed_)
            std::fill(out_.begin(), out_.end(),
                      std::numeric_limits<double>::quiet_NaN());
        downstream_.push(Frame<Index>{label, out_.data(), out_.size()});
    }

    static std::int64_t floordiv(std::int64_t a, std::int64_t b) {
        std::int64_t q = a / b, r = a % b;
        if (r != 0 && ((r < 0) != (b < 0))) --q;
        return q;
    }

    ResampleParams p_;
    Sink<Index>& downstream_;
    EvalOp* reducer_;
    std::vector<double> out_;   // reducer's latest output (width n_out())
    bool has_ = false;          // any event fell in the current bucket
    bool fed_ = false;          // reducer got >=1 finite sample this bucket
    bool started_ = false;
    std::int64_t bucket_ = 0;
    Index cur_label_{};
    std::int64_t count_in_bucket_ = 0;
    Index first_index_{}, last_index_{};
};

}} // namespace screamer::dag
#endif
