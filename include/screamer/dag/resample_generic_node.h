#ifndef SCREAMER_DAG_RESAMPLE_GENERIC_NODE_H
#define SCREAMER_DAG_RESAMPLE_GENERIC_NODE_H

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
#include "screamer/common/eval_op.h"
#include "screamer/common/checked_arith.h"
#include "screamer/common/float_info.h"
#include "screamer/dag/frame.h"
#include "screamer/dag/resample_params.h"
#include <cstddef>
#include <string>

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
          out_(reducer_ ? reducer_->n_out() : 0),
          last_emitted_(reducer_ ? reducer_->n_out() : 0),
          nan_row_(reducer_ ? reducer_->n_out() : 0,
                   std::numeric_limits<double>::quiet_NaN()) {
        if (!reducer_)
            throw std::runtime_error(
                "dag::GenericResampleNode: null reducer");
        if (reducer_->n_in() != 1)
            throw std::invalid_argument(
                "a reducer resample takes a reducer of 1 input, got one of " +
                std::to_string(reducer_->n_in()));
        reset();
    }

    void push(const Frame<Index>& f) override {
        if (f.width != 1)
            throw std::invalid_argument(
                "a reducer resample takes 1 column, but its input has " +
                std::to_string(f.width));
        if (p_.mode == ResampleMode::ByIndex) push_by_index(f.index, f.values[0]);
        else                                  push_by_count(f.index, f.values[0]);
    }

    void flush() override {
        if (p_.mode == ResampleMode::ByIndex) {
            if (!closed_ && has_) emit(cur_label_);
        } else {
            if (count_in_bucket_ > 0) emit(p_.label == ResampleLabel::Left ? first_index_ : last_index_);
        }
        // idempotent: clear so a repeat flush emits nothing
        clear_bucket();
        started_ = false;
        closed_ = false;
        count_in_bucket_ = 0;
        downstream_.flush();
    }

    void reset() override {
        clear_bucket();
        started_ = false;
        closed_ = false;
        bucket_ = 0;
        cur_label_ = Index{};
        count_in_bucket_ = 0;
        first_index_ = last_index_ = Index{};
        have_emitted_ = false;
    }

    std::size_t n_in()  const override { return 1; }
    std::size_t n_out() const override { return out_.size(); }

    // Event-time watermark: the same rule as ResampleNode::on_watermark. An open
    // bucket still emits at cur_label_; otherwise the next bar is a fill bar for
    // the empty bucket after the last closed one (fill nan or carry), or the bar
    // of the bucket that holds the next event. The end-of-input watermark (the
    // int64 maximum) forwards unchanged.
    void on_watermark(Index w) override {
        advance(w);
        Index fwd = w;
        if (p_.mode == ResampleMode::ByIndex) {
            if (started_ && !closed_) { if (cur_label_ < fwd) fwd = cur_label_; }
            else if (w != std::numeric_limits<Index>::max()) {
                Index next = label_for(bucket_of(w));
                if (started_ && p_.fill != ResampleFill::Skip) {
                    const Index gap = label_for(bucket_ + 1);
                    if (gap < next) next = gap;
                }
                if (next < fwd) fwd = next;
            }
        } else {
            // Count mode: advance() is a no-op; clamp to the pending emit-index of
            // the open bucket (first_index_ for label=left, last_index_ for right).
            if (count_in_bucket_ > 0) {
                Index pending = (p_.label == ResampleLabel::Left
                                 ? first_index_ : last_index_);
                if (pending < fwd) fwd = pending;
            }
        }
        downstream_.on_watermark(fwd);
    }

    // Time-driven close, the same rule as ResampleNode::advance: once `now` has
    // left the open bucket, emit it and mark it closed. Empty buckets are filled
    // only between two buckets with data, when the later one's event arrives, so
    // a watermark changes when a bar is emitted, never which bars. `now` at the
    // int64 maximum is the end of input. No-op for count mode.
    void advance(Index now) {
        if (p_.mode != ResampleMode::ByIndex) return;   // count mode: time has no meaning
        if (!started_) return;                           // no bucket open yet
        if (now == std::numeric_limits<Index>::max()) {
            if (!closed_ && has_) emit(cur_label_);
            clear_bucket();
            started_ = false;
            closed_ = false;
            return;
        }
        if (closed_ || bucket_of(now) <= bucket_) return;
        if (has_) emit(cur_label_);
        clear_bucket();
        closed_ = true;
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
        } else if (nb != bucket_ || closed_) {
            // Emit the bucket left behind (unless advance() closed it), then
            // fill the empty buckets strictly between it and nb.
            if (!closed_ && has_) emit(cur_label_);
            if (p_.fill != ResampleFill::Skip)
                for (std::int64_t b = bucket_ + 1; b < nb; ++b)
                    emit_fill(label_for(b));
            bucket_ = nb; clear_bucket(); set_index_label(nb);
        }
        closed_ = false;
        add(v);
    }

    Index label_for(std::int64_t nb) const {
        std::int64_t start = p_.origin + nb * p_.width;
        return static_cast<Index>(p_.label == ResampleLabel::Left ? start : start + p_.width);
    }

    void set_index_label(std::int64_t nb) { cur_label_ = label_for(nb); }

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
        if (p_.fill == ResampleFill::Carry) {
            last_emitted_.assign(out_.begin(), out_.end());
            have_emitted_ = true;
        }
        downstream_.push(Frame<Index>{label, out_.data(), out_.size()});
    }

    // Emit a filler row for an internal empty bucket. Nan -> an all-NaN row;
    // Carry -> the previous emitted row verbatim (skipped if nothing emitted yet).
    void emit_fill(Index label) {
        if (p_.fill == ResampleFill::Nan) {
            downstream_.push(Frame<Index>{label, nan_row_.data(), nan_row_.size()});
        } else if (have_emitted_) {  // Carry
            downstream_.push(Frame<Index>{label, last_emitted_.data(), last_emitted_.size()});
        }
    }

    // The bucket that holds index k. k - origin is taken without overflow (a
    // watermark can be any int64).
    std::int64_t bucket_of(Index k) const {
        return floordiv(screamer::saturating_sub(static_cast<std::int64_t>(k), p_.origin),
                        p_.width);
    }

    static std::int64_t floordiv(std::int64_t a, std::int64_t b) {
        std::int64_t q = a / b, r = a % b;
        if (r != 0 && ((r < 0) != (b < 0))) --q;
        return q;
    }

    ResampleParams p_;
    Sink<Index>& downstream_;
    EvalOp* reducer_;
    std::vector<double> out_;            // reducer's latest output (width n_out())
    std::vector<double> last_emitted_;   // Carry: last emitted row
    std::vector<double> nan_row_;        // Nan: reusable all-NaN row
    bool has_ = false;          // any event fell in the current bucket
    bool fed_ = false;          // reducer got >=1 finite sample this bucket
    bool have_emitted_ = false; // Carry: any real row emitted yet
    bool started_ = false;
    bool closed_ = false;       // ByIndex: advance() emitted the bucket; no event since
    std::int64_t bucket_ = 0;
    Index cur_label_{};
    std::int64_t count_in_bucket_ = 0;
    Index first_index_{}, last_index_{};
};

}} // namespace screamer::dag
#endif
