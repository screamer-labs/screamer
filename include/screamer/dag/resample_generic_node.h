#ifndef SCREAMER_DAG_RESAMPLE_GENERIC_NODE_H
#define SCREAMER_DAG_RESAMPLE_GENERIC_NODE_H

#include <algorithm>
#include <cmath>
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
        forward_open_bar(f.index);
    }

    void flush() override {
        if (p_.mode == ResampleMode::ByIndex) {
            if (has_) emit(cur_label());
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
        label_ok_ = true;
        start_ok_ = false;
        has_bar_wm_ = false;
        count_in_bucket_ = 0;
        first_index_ = last_index_ = Index{};
        have_emitted_ = false;
        std::fill(last_emitted_.begin(), last_emitted_.end(),
                  std::numeric_limits<double>::quiet_NaN());
    }

    std::size_t n_in()  const override { return 1; }
    std::size_t n_out() const override { return out_.size(); }

    // Event-time watermark: close windows up to `w` then forward a watermark
    // clamped to the smallest index this node might STILL emit, so a merge or
    // combinator downstream does not stall AND never receives `w` before a lower-
    // indexed frame it will later emit. advance(w) leaves the bucket CONTAINING w
    // open with emit-label cur_label_. For label="left" that label is the bucket's
    // left edge (<= w), so a later frame at index < w is still coming; we must
    // forward min(w, cur_label_). For label="right" the label is the right edge
    // (> w), so min(w, cur_label_) == w and forwarding w is already safe. Before
    // any event has anchored a bucket (not started_), the first event can still open
    // the bucket that holds w, so the forward is at most that bucket's label (a
    // label-left bucket starts at or below w). The end-of-input watermark (the
    // int64 maximum, sent at a flush) emits the current bucket only if it has
    // data, as flush() does: trailing empty buckets are never filled.
    void on_watermark(Index w) override {
        advance(w);
        Index fwd = w;
        if (p_.mode == ResampleMode::ByIndex) {
            // Index mode: advance() may have left an open bucket whose emit-label
            // (cur_label_) is below w. Forward at most cur_label_ so the downstream
            // merge is not told "no frame < w" while one is still pending.
            if (started_) { if (cur_label_ < fwd) fwd = cur_label_; }
            else if (w != std::numeric_limits<Index>::max()) {
                const Index first = static_cast<Index>(
                    resample_label_bound(resample_bucket_bound(w, p_), p_));
                if (first < fwd) fwd = first;
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

    // Close every window whose end boundary has passed by logical time `now`, even
    // when empty. Emits the current bucket (real row if it has data) and then any
    // trailing empty buckets up to but NOT including the bucket that contains `now`
    // (that bucket is still open). No-op for count mode (windows are event-counted,
    // not timed) and when the node has not started (no event has anchored it yet).
    void advance(Index now) {
        if (p_.mode != ResampleMode::ByIndex) return;   // count mode: time has no meaning
        if (!started_) return;                           // nothing to anchor trailing empties
        // The end-of-input watermark emits the current bucket if it has data
        // and fills no empty bucket, as flush() does: they are all trailing.
        const bool end = now == std::numeric_limits<Index>::max();
        std::int64_t target = end ? bucket_ + 1 : resample_bucket_bound(now, p_);
        if (target <= bucket_) return;                   // still inside the current bucket
        if (p_.fill != ResampleFill::Skip && !end)
            check_fill_gap(bucket_, target, !has_, p_.max_fill);
        // Close the current bucket: real row if it saw events, else a fill row.
        // At the end of input an empty bucket is trailing: as flush(), no fill.
        if (has_) emit(cur_label());
        else if (p_.fill != ResampleFill::Skip && !end) emit_fill(cur_label());
        // Trailing empty buckets strictly between the current one and `target`.
        if (p_.fill != ResampleFill::Skip && !end)
            for (std::int64_t b = bucket_ + 1; b < target; ++b) emit_fill(label_for(b));
        // Move to the (still open, empty) bucket that contains `now`.
        bucket_ = target; clear_bucket(); set_index_label(target);
        if (end) started_ = false;     // as after a flush: the next event anchors
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
        const std::int64_t nb = resample_bucket(static_cast<std::int64_t>(k), p_);
        check_label(nb);
        if (!started_) {
            started_ = true; bucket_ = nb; clear_bucket(); set_index_label(nb);
        } else if (nb != bucket_) {
            if (p_.fill != ResampleFill::Skip)
                check_fill_gap(bucket_, nb, !has_, p_.max_fill);
            if (has_) emit(cur_label());
            // A parked empty current bucket (e.g. advance() left it open with no
            // event) is filled too, matching advance()+fill semantics.
            else if (p_.fill != ResampleFill::Skip) emit_fill(cur_label());
            // Fill internal gaps (empty buckets strictly between bucket_ and nb);
            // trailing empties are handled by advance()/flush(), not here.
            if (p_.fill != ResampleFill::Skip)
                for (std::int64_t b = bucket_ + 1; b < nb; ++b)
                    emit_fill(label_for(b));
            bucket_ = nb; clear_bucket(); set_index_label(nb);
        }
        add(v);
    }

    // The label of bar nb (an emitted row); raises outside the int64 range.
    Index label_for(std::int64_t nb) const {
        return static_cast<Index>(resample_label_checked(nb, p_));
    }

    // An event at bar nb: its label must be in the int64 range, checked
    // before the event changes any state.
    void check_label(std::int64_t nb) const {
        if (started_ && nb == bucket_) return;
        std::int64_t l;
        if (!resample_label(nb, p_, l)) throw_resample_label(p_);
    }

    // The open bar's label. advance(now) can open a bar whose label is
    // outside the int64 range: it is kept clamped (a bound for watermarks)
    // and raises if the bar is ever emitted.
    void set_index_label(std::int64_t nb) {
        std::int64_t l;
        label_ok_ = resample_label(nb, p_, l);
        cur_label_ = static_cast<Index>(label_ok_ ? l : resample_label_bound(nb, p_));
        start_ok_ = resample_start(nb, p_, l);
        if (start_ok_) cur_start_ = static_cast<Index>(l);
    }

    // After an event at k: the open bar emits no row below its first index
    // (its start for every=; the first event of a count= or threshold= bar,
    // else k), so that is a watermark. It moves once per bar, and it is never
    // above k, so the node stays never ahead of the input merge. A consumer
    // (an output alignment, a join) then holds at most the rows of one open
    // bar, not of two.
    void forward_open_bar(Index k) {
        Index w;
        if (p_.mode == ResampleMode::ByIndex) {
            if (!started_ || !start_ok_) return;
            w = cur_start_;
        } else {
            w = count_in_bucket_ > 0 ? first_index_ : k;
        }
        if (has_bar_wm_ && w <= bar_wm_) return;
        bar_wm_ = w;
        has_bar_wm_ = true;
        downstream_.on_watermark(w);
    }
    Index cur_label() const {
        if (!label_ok_) throw_resample_label(p_);
        return cur_label_;
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
        // fill="carry" under freq=: each NaN cell takes its column's last
        // non-NaN value (as ResampleNode::carry_cells).
        if (p_.fill == ResampleFill::Carry && p_.mode == ResampleMode::ByIndex) {
            for (std::size_t i = 0; i < out_.size(); ++i) {
                if (std::isnan(out_[i])) out_[i] = last_emitted_[i];
                else last_emitted_[i] = out_[i];
            }
            have_emitted_ = true;
        }
        downstream_.push(Frame<Index>{label, out_.data(), out_.size()});
    }

    // Emit a filler row for an internal empty bucket. Nan -> an all-NaN row;
    // Carry -> each column's last non-NaN value (skipped if nothing emitted yet).
    void emit_fill(Index label) {
        if (p_.fill == ResampleFill::Nan) {
            downstream_.push(Frame<Index>{label, nan_row_.data(), nan_row_.size()});
        } else if (have_emitted_) {  // Carry
            downstream_.push(Frame<Index>{label, last_emitted_.data(), last_emitted_.size()});
        }
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
    bool label_ok_ = true;       // cur_label_ is the open bar's label (not clamped)
    Index cur_start_{};          // the open bar's start (every=), when start_ok_
    bool start_ok_ = false;
    Index bar_wm_{};             // the last forward_open_bar watermark
    bool has_bar_wm_ = false;
    std::int64_t count_in_bucket_ = 0;
    Index first_index_{}, last_index_{};
};

}} // namespace screamer::dag
#endif
