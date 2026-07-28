#ifndef SCREAMER_DAG_RESAMPLE_NODE_H
#define SCREAMER_DAG_RESAMPLE_NODE_H

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
#include "screamer/common/float_info.h"
#include "screamer/dag/frame.h"
#include "screamer/dag/resample_params.h"

namespace screamer { namespace dag {

// Single-pass O(1) NaN-ignore accumulator for one input column.
// add() folds one value; emit_one() writes the reducer result for the given
// ResampleAgg kind. `has` marks that any event (NaN or not) fell in the bucket.
struct ResampleAccum {
    std::int64_t count = 0;
    double sum = 0.0, sum_pos = 0.0, sum_neg = 0.0;
    double mn = 0.0, mx = 0.0, first = 0.0, last = 0.0;
    bool has = false;

    void reset() {
        count = 0;
        sum = 0.0; sum_pos = 0.0; sum_neg = 0.0;
        mn = mx = first = last = 0.0;
        has = false;
    }

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
        if (v > 0.0) sum_pos += v;
        if (v < 0.0) sum_neg += v;
        ++count;
    }

    // Emit a single output value for the given reducer kind.
    void emit_one(ResampleAgg agg, double* out) const {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        switch (agg) {
        case ResampleAgg::First:  out[0] = count ? first : nan; break;
        case ResampleAgg::Last:   out[0] = count ? last  : nan; break;
        case ResampleAgg::Min:    out[0] = count ? mn    : nan; break;
        case ResampleAgg::Max:    out[0] = count ? mx    : nan; break;
        case ResampleAgg::Sum:    out[0] = sum;                 break;
        case ResampleAgg::Count:  out[0] = static_cast<double>(count); break;
        case ResampleAgg::Mean:
            out[0] = count ? sum / static_cast<double>(count) : nan; break;
        // SumPos: sum of the positive part of each value (= sum of max(v, 0)).
        // SumNeg: sum of the magnitude of the negative part (= sum of |min(v, 0)|).
        //   Stored as sum_neg (negative), returned negated so sell_vol is positive,
        //   consistent with NegPart() + sum used in the pre-C++-plan ohlcv2 path.
        case ResampleAgg::SumPos: out[0] = sum_pos;  break;
        case ResampleAgg::SumNeg: out[0] = -sum_neg; break;
        case ResampleAgg::Ohlc:
            // Ohlc in the plan context: emit 4 values for one input column.
            // (Used for the legacy single-agg path; plan entries use single-value aggs.)
            out[0] = count ? first : nan;
            out[1] = count ? mx    : nan;
            out[2] = count ? mn    : nan;
            out[3] = count ? last  : nan;
            break;
        }
    }

    // Legacy emit for the builtin single-agg path (writes 1 or 4 doubles for Ohlc).
    void emit(ResampleAgg agg, double* out) const {
        emit_one(agg, out);
    }
};

// Stateful windowing push-node. Buckets a stream by index-interval or event
// count, reduces each bucket with a C++ reducer (plan or single builtin agg),
// and emits the bucket on the causal boundary.
//
// Single-column mode (plan empty): input width must be 1. Behaves exactly as
// before - one ResampleAccum for the single input.
//
// Multi-column plan mode (plan non-empty): input width must equal the number
// of distinct input columns in the plan. Each plan entry maps an output column
// to a (reducer_kind, input_col_index) pair. Output width = plan.size().
//
// flush() emits the trailing partial bucket and is idempotent.
template <class Index>
class ResampleNode : public Sink<Index> {
public:
    ResampleNode(ResampleParams p, Sink<Index>& downstream)
        : p_(p), downstream_(downstream),
          out_width_(resample_output_width(p)),
          out_(out_width_),
          last_emitted_(out_width_),
          nan_row_(out_width_, std::numeric_limits<double>::quiet_NaN()),
          // Multi-column plan: one accumulator per distinct input column.
          // Single-column (plan empty): one accumulator for the single input.
          // OhlcvBars with empty plan: num_accums_ and accums_ are initialized
          // lazily on first push() when the real input width becomes known.
          num_accums_(p.plan.empty() ? 1u : resample_plan_input_cols(p.plan)),
          accums_(p.agg == ResampleAgg::OhlcvBars && p.plan.empty() ? 0u : num_accums_)
    { reset(); }

    void push(const Frame<Index>& f) override {
        // OhlcvBars with deferred plan: build the plan from the first frame's width.
        if (p_.agg == ResampleAgg::OhlcvBars && p_.plan.empty()) {
            p_.plan = make_ohlcv_bars_plan(f.width);
            num_accums_ = resample_plan_input_cols(p_.plan);
            out_width_  = p_.plan.size();
            out_.assign(out_width_, 0.0);
            last_emitted_.assign(out_width_, std::numeric_limits<double>::quiet_NaN());
            nan_row_.assign(out_width_, std::numeric_limits<double>::quiet_NaN());
            accums_.resize(num_accums_);
        }
        if (p_.plan.empty()) {
            // Single-column path: width must be 1.
            if (f.width != 1)
                throw std::runtime_error(
                    "dag::ResampleNode: expects a width-1 input stream");
            if (p_.mode == ResampleMode::ByIndex)
                push_by_index(f.index, f.values[0]);
            else
                push_by_count(f.index, f.values[0]);
        } else {
            // Multi-column plan path: input width must match num_accums_.
            if (f.width != num_accums_)
                throw std::runtime_error(
                    "dag::ResampleNode: input frame width (" +
                    std::to_string(f.width) + ") does not match number of "
                    "plan input columns (" + std::to_string(num_accums_) + ")");
            if (p_.mode == ResampleMode::ByIndex)
                push_by_index_multi(f.index, f.values);
            else
                push_by_count_multi(f.index, f.values);
        }
    }

    void flush() override {
        if (p_.mode == ResampleMode::ByIndex) {
            if (has_any()) emit(cur_label_);
        } else {
            if (count_in_bucket_ > 0)
                emit(p_.label == ResampleLabel::Left ? first_index_ : last_index_);
        }
        // idempotent: clear so a repeat flush emits nothing
        started_ = false;
        count_in_bucket_ = 0;
        reset_accums();
        downstream_.flush();
    }

    void reset() override {
        reset_accums();
        started_ = false;
        bucket_ = 0;
        cur_label_ = Index{};
        count_in_bucket_ = 0;
        first_index_ = last_index_ = Index{};
        have_emitted_ = false;
    }

    std::size_t n_in()  const override { return p_.plan.empty() ? 1u : num_accums_; }
    std::size_t n_out() const override { return out_width_; }

    // Event-time watermark: close windows up to `w` then forward a watermark
    // clamped to the smallest index this node might STILL emit, so a merge or
    // combinator downstream does not stall AND never receives `w` before a lower-
    // indexed frame it will later emit. advance(w) leaves the bucket CONTAINING w
    // open with emit-label cur_label_. For label="left" that label is the bucket's
    // left edge (<= w), so a later frame at index < w is still coming; we must
    // forward min(w, cur_label_). For label="right" the label is the right edge
    // (> w), so min(w, cur_label_) == w and forwarding w is already safe. Before
    // any event has anchored a bucket (not started_) there is nothing pending, so
    // forward w unchanged.
    void on_watermark(Index w) override {
        advance(w);
        Index fwd = w;
        if (p_.mode == ResampleMode::ByIndex) {
            if (started_ && cur_label_ < fwd) fwd = cur_label_;
        } else {
            if (count_in_bucket_ > 0) {
                Index pending = (p_.label == ResampleLabel::Left
                                 ? first_index_ : last_index_);
                if (pending < fwd) fwd = pending;
            }
        }
        downstream_.on_watermark(fwd);
    }

    // Close every window whose end boundary has passed by logical time `now`, even
    // when empty. No-op for count mode. Emits the current bucket and any trailing
    // empty buckets up to but NOT including the bucket containing `now`.
    void advance(Index now) {
        if (p_.mode != ResampleMode::ByIndex) return;
        if (!started_) return;
        std::int64_t target =
            floordiv(static_cast<std::int64_t>(now) - p_.origin, p_.width);
        if (target <= bucket_) return;
        if (has_any()) emit(cur_label_);
        else if (p_.fill != ResampleFill::Skip) emit_fill(cur_label_);
        if (p_.fill != ResampleFill::Skip)
            for (std::int64_t b = bucket_ + 1; b < target; ++b) emit_fill(label_for(b));
        bucket_ = target; reset_accums(); set_index_label(target);
    }

private:
    bool has_any() const {
        for (const auto& a : accums_) if (a.has) return true;
        return false;
    }

    void reset_accums() {
        for (auto& a : accums_) a.reset();
    }

    void push_by_index(Index k, double v) {
        std::int64_t nb = floordiv(static_cast<std::int64_t>(k) - p_.origin, p_.width);
        if (!started_) {
            started_ = true; bucket_ = nb; reset_accums(); set_index_label(nb);
        } else if (nb != bucket_) {
            if (has_any()) emit(cur_label_);
            else if (p_.fill != ResampleFill::Skip) emit_fill(cur_label_);
            if (p_.fill != ResampleFill::Skip)
                for (std::int64_t b = bucket_ + 1; b < nb; ++b)
                    emit_fill(label_for(b));
            bucket_ = nb; reset_accums(); set_index_label(nb);
        }
        accums_[0].add(v);
    }

    void push_by_count(Index k, double v) {
        if (count_in_bucket_ == 0) first_index_ = k;
        last_index_ = k;
        accums_[0].add(v);
        ++count_in_bucket_;
        if (count_in_bucket_ == p_.count) {
            emit(p_.label == ResampleLabel::Left ? first_index_ : last_index_);
            reset_accums();
            count_in_bucket_ = 0;
        }
    }

    void push_by_index_multi(Index k, const double* vals) {
        std::int64_t nb = floordiv(static_cast<std::int64_t>(k) - p_.origin, p_.width);
        if (!started_) {
            started_ = true; bucket_ = nb; reset_accums(); set_index_label(nb);
        } else if (nb != bucket_) {
            if (has_any()) emit(cur_label_);
            else if (p_.fill != ResampleFill::Skip) emit_fill(cur_label_);
            if (p_.fill != ResampleFill::Skip)
                for (std::int64_t b = bucket_ + 1; b < nb; ++b)
                    emit_fill(label_for(b));
            bucket_ = nb; reset_accums(); set_index_label(nb);
        }
        for (std::size_t c = 0; c < num_accums_; ++c)
            accums_[c].add(vals[c]);
    }

    void push_by_count_multi(Index k, const double* vals) {
        if (count_in_bucket_ == 0) first_index_ = k;
        last_index_ = k;
        for (std::size_t c = 0; c < num_accums_; ++c)
            accums_[c].add(vals[c]);
        ++count_in_bucket_;
        if (count_in_bucket_ == p_.count) {
            emit(p_.label == ResampleLabel::Left ? first_index_ : last_index_);
            reset_accums();
            count_in_bucket_ = 0;
        }
    }

    Index label_for(std::int64_t nb) const {
        std::int64_t start = p_.origin + nb * p_.width;
        return static_cast<Index>(p_.label == ResampleLabel::Left ? start : start + p_.width);
    }

    void set_index_label(std::int64_t nb) { cur_label_ = label_for(nb); }

    void emit(Index label) {
        if (p_.plan.empty()) {
            // Single-column / single-agg path (writes 1 or 4 doubles for Ohlc).
            accums_[0].emit(p_.agg, out_.data());
        } else {
            // Multi-column plan: each output column emits via its reducer from
            // the designated input column's accumulator.
            for (std::size_t i = 0; i < p_.plan.size(); ++i)
                accums_[p_.plan[i].input_col].emit_one(p_.plan[i].agg, &out_[i]);
        }
        if (p_.fill == ResampleFill::Carry) {
            last_emitted_.assign(out_.begin(), out_.end());
            have_emitted_ = true;
        }
        downstream_.push(Frame<Index>{label, out_.data(), out_.size()});
    }

    void emit_fill(Index label) {
        if (p_.fill == ResampleFill::Nan) {
            downstream_.push(Frame<Index>{label, nan_row_.data(), nan_row_.size()});
        } else if (have_emitted_) {
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
    std::size_t out_width_;
    std::vector<double> out_;
    std::vector<double> last_emitted_;
    std::vector<double> nan_row_;
    std::size_t num_accums_;          // distinct input columns = accums_.size()
    std::vector<ResampleAccum> accums_;
    bool started_ = false;
    bool have_emitted_ = false;
    std::int64_t bucket_ = 0;
    Index cur_label_{};
    std::int64_t count_in_bucket_ = 0;
    Index first_index_{}, last_index_{};
};

}} // namespace screamer::dag
#endif
