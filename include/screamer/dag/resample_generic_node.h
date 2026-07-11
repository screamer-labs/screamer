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
          out_(reducer_ ? reducer_->n_out() : 0),
          last_emitted_(reducer_ ? reducer_->n_out() : 0),
          nan_row_(reducer_ ? reducer_->n_out() : 0,
                   std::numeric_limits<double>::quiet_NaN()) {
        if (!reducer_)
            throw std::runtime_error(
                "dag::GenericResampleNode: null reducer");
        if (reducer_->n_in() != 1)
            throw std::runtime_error(
                "dag::GenericResampleNode: reducer must have exactly 1 input "
                "(this is a single-value stream reducer); got n_in=" +
                std::to_string(reducer_->n_in()));
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

    void reset() override {
        clear_bucket();
        started_ = false;
        bucket_ = 0;
        cur_label_ = Index{};
        count_in_bucket_ = 0;
        first_index_ = last_index_ = Index{};
        have_emitted_ = false;
    }

    std::size_t n_in()  const override { return 1; }
    std::size_t n_out() const override { return out_.size(); }

    // Close every window whose end boundary has passed by logical time `now`, even
    // when empty. Emits the current bucket (real row if it has data) and then any
    // trailing empty buckets up to but NOT including the bucket that contains `now`
    // (that bucket is still open). No-op for count mode (windows are event-counted,
    // not timed) and when the node has not started (no event has anchored it yet).
    void advance(Index now) {
        if (p_.mode != ResampleMode::ByIndex) return;   // count mode: time has no meaning
        if (!started_) return;                           // nothing to anchor trailing empties
        std::int64_t target =
            floordiv(static_cast<std::int64_t>(now) - p_.origin, p_.width);
        if (target <= bucket_) return;                   // still inside the current bucket
        // Close the current bucket: real row if it saw events, else a fill row.
        if (has_) emit(cur_label_);
        else if (p_.fill != ResampleFill::Skip) emit_fill(cur_label_);
        // Trailing empty buckets strictly between the current one and `target`.
        if (p_.fill != ResampleFill::Skip)
            for (std::int64_t b = bucket_ + 1; b < target; ++b) emit_fill(label_for(b));
        // Move to the (still open, empty) bucket that contains `now`.
        bucket_ = target; clear_bucket(); set_index_label(target);
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
            // A parked empty current bucket (e.g. advance() left it open with no
            // event) is filled too, so advance()+fill agrees with MultiResampleNode.
            else if (p_.fill != ResampleFill::Skip) emit_fill(cur_label_);
            // Fill internal gaps (empty buckets strictly between bucket_ and nb);
            // trailing empties are handled by advance()/flush(), not here.
            if (p_.fill != ResampleFill::Skip)
                for (std::int64_t b = bucket_ + 1; b < nb; ++b)
                    emit_fill(label_for(b));
            bucket_ = nb; clear_bucket(); set_index_label(nb);
        }
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
    std::int64_t bucket_ = 0;
    Index cur_label_{};
    std::int64_t count_in_bucket_ = 0;
    Index first_index_{}, last_index_{};
};

}} // namespace screamer::dag
#endif
