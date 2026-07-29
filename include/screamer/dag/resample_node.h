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
        case ResampleAgg::OhlcvBars:
            // OhlcvBars is a node-level sentinel decomposed into a plan before any
            // emit_one call; reaching this case is a bug in the caller.
            throw std::logic_error("emit_one: OhlcvBars must be resolved to a plan before emission");
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
            // ByCumulative: the frame is [value, driver]; value width = 1.
            // All other modes: value width = frame.width.
            std::size_t val_width = f.width;
            if (p_.mode == ResampleMode::ByCumulative)
                val_width = (f.width > 0) ? f.width - 1 : 0;
            p_.plan = make_ohlcv_bars_plan(val_width);
            num_accums_ = resample_plan_input_cols(p_.plan);
            out_width_  = p_.plan.size();
            out_.assign(out_width_, 0.0);
            last_emitted_.assign(out_width_, std::numeric_limits<double>::quiet_NaN());
            nan_row_.assign(out_width_, std::numeric_limits<double>::quiet_NaN());
            accums_.resize(num_accums_);
        }
        if (p_.mode == ResampleMode::ByCumulative) {
            // ByCumulative: input frame must be width 2 (value col + driver col).
            // The driver is always the LAST column; value columns are all but last.
            // In the current implementation value is width 1 (single value column)
            // so the frame is always [value, driver] (width 2).
            if (f.width != 2)
                throw std::runtime_error(
                    "dag::ResampleNode(ByCumulative): expects width-2 input "
                    "[value, driver]; got width " + std::to_string(f.width));
            push_by_cumulative(f.index, f.values[0], f.values[1]);
            return;
        }
        // ByClock is always wired via 2-port (ClockPort) in the compiled graph;
        // push() is never called for ByClock mode in practice.
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
        } else if (p_.mode == ResampleMode::ByClock) {
            // ByClock: if a clock tick was deferred (pending_clock_tick_), emit it
            // now — end-of-stream means no future events can arrive at the same index.
            // After that, any remaining buffered value events are discarded (they
            // arrived AFTER the last clock tick and have no bucket to close them).
            if (pending_clock_tick_) {
                do_clock_tick(pending_clock_tick_index_);
                pending_clock_tick_ = false;
            }
            // Trailing partial bucket after the last clock tick is NOT emitted
            // (the clock drives emission; no clock tick = no bar).
        } else if (p_.mode == ResampleMode::ByCumulative) {
            // Emit trailing partial bucket if any events landed in it.
            if (count_in_bucket_ > 0)
                emit(p_.label == ResampleLabel::Left ? first_index_ : last_index_);
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
        have_last_finite_ = false;
        last_finite_index_ = Index{};
        cum_driver_ = 0.0;
        prev_clock_tick_set_ = false;
        prev_clock_tick_ = Index{};
        clock_value_buf_.clear();
        clock_port_flush_count_ = 0;
        pending_clock_tick_ = false;
        pending_clock_tick_index_ = Index{};
    }

    std::size_t n_in()  const override {
        // ByCumulative and ByClock return 2 here for graph-validation purposes,
        // but both modes are wired via a single width-N input (a CombineLatest
        // upstream collapses value + driver/clock into one frame); the 2u is
        // advisory, not a port count.
        if (p_.mode == ResampleMode::ByCumulative) return 2u;
        if (p_.mode == ResampleMode::ByClock) return 2u;
        return p_.plan.empty() ? 1u : num_accums_;
    }
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

    // ByClock 2-port interface: port(0) = value stream, port(1) = clock stream.
    // Used by the compiled graph to wire two separate input nodes directly into
    // the ResampleNode without combining them into a single frame first.
    // Events from both ports arrive in global index order (the compiled graph's
    // MergeSource guarantees this), so no reorder buffer is needed.
    //
    // The port objects are created lazily on first call to port(). Non-movable:
    // if the node is moved after port() was called the Port's back-references
    // would dangle. The compiled graph creates nodes via shared_ptr and never
    // moves them after creation, so this is safe in practice.
    struct ClockPort : Sink<Index> {
        ResampleNode& node;
        std::size_t port_idx;  // 0 = value, 1 = clock
        ClockPort(ResampleNode& n, std::size_t i) : node(n), port_idx(i) {}
        void push(const Frame<Index>& f) override {
            if (port_idx == 0) {
                // Value port: accumulate all value columns.
                node.push_by_clock_value(f.index, f.values, f.width);
            } else {
                // Clock port: close the bucket at this index (clock value ignored).
                node.push_by_clock_tick(f.index);
            }
        }
        void flush() override {
            node.clock_port_flush_count_++;
            if (node.clock_port_flush_count_ >= 2) {
                // Both ports flushed: run the node's own flush() which emits any
                // deferred pending clock tick and then propagates flush downstream.
                node.clock_port_flush_count_ = 0;
                node.flush();
            }
        }
        void on_watermark(Index /*w*/) override {}
        std::size_t n_in()  const override { return 1; }
        std::size_t n_out() const override { return 0; }
    };
    friend struct ClockPort;

    // Returns the clock port for ByClock 2-port wiring. Creates ports on first call.
    Sink<Index>& port(std::size_t i) {
        if (clock_ports_.empty()) {
            clock_ports_.emplace_back(*this, 0u);
            clock_ports_.emplace_back(*this, 1u);
        }
        return clock_ports_[i];
    }
    friend class CompiledGraph;

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

    // ByCumulative: value is the bar value; driver is the cumulative driver.
    // NaN driver is ignored (does not advance the cumulative sum).
    // The value is always added to the accumulator (even on a NaN driver row),
    // consistent with nan_policy=ignore: NaN driver does not close the bar, but
    // the observation's value DOES contribute to the bar's reducer.
    void push_by_cumulative(Index k, double value, double driver) {
        if (count_in_bucket_ == 0) first_index_ = k;
        last_index_ = k;
        // Always add the value (the price) to the accumulator.
        accums_[0].add(value);
        ++count_in_bucket_;
        // Only advance the cumulative driver if it is not NaN.
        if (!screamer::isnan2(driver)) {
            cum_driver_ += driver;
            if (cum_driver_ >= p_.threshold) {
                emit(p_.label == ResampleLabel::Left ? first_index_ : last_index_);
                reset_accums();
                count_in_bucket_ = 0;
                cum_driver_ = 0.0;
            }
        }
    }

    bool clock_value_is_fresh(Index k) const {
        const auto tick = static_cast<std::uint64_t>(
            static_cast<std::int64_t>(k));
        const auto last = static_cast<std::uint64_t>(
            static_cast<std::int64_t>(last_finite_index_));
        return tick - last <= static_cast<std::uint64_t>(p_.max_age);
    }

    // ByClock: buffer a value event for deferred accumulation.
    // Events are buffered (with their timeline index k) so that clock ticks can
    // drain only the events whose index falls within the closing bucket.
    // This supports the case where value events arrive with a shifted index
    // (e.g., from a Delay node inside the same compiled graph), which can cause
    // events to arrive in original-index order but out of timeline-index order.
    // vals is the value portion of the frame (all but the last is_clock column).
    // num_val_cols is the count of value columns (frame.width - 1).
    //
    // Tie-order fix (C1): before buffering, check if a pending clock tick at a
    // STRICTLY EARLIER index can now be emitted safely (the current value event
    // arrives at a higher index, proving no same-index value events remain for
    // the pending tick). This makes the output independent of whether the value
    // or clock input is listed first in Pipeline([...]).
    void push_by_clock_value(Index k, const double* vals, std::size_t num_val_cols) {
        if (p_.max_age >= 0 && num_val_cols != 1) {
            throw std::runtime_error(
                "dag::ResampleNode(ByClock): max_age requires a width-1 value stream");
        }
        if (p_.plan.empty() && num_val_cols > 1) {
            // Multi-column single-agg ByClock: build a trivial plan on first push
            // (one entry per input column, all using the same agg kind).
            // This matches resample(multi_col, clock=..., agg='last') semantics:
            // each column is reduced independently with the same reducer.
            for (std::size_t c = 0; c < num_val_cols; ++c)
                p_.plan.push_back({p_.agg, c});
            num_accums_ = num_val_cols;
            out_width_  = num_val_cols;
            out_.assign(out_width_, 0.0);
            last_emitted_.assign(out_width_, std::numeric_limits<double>::quiet_NaN());
            nan_row_.assign(out_width_, std::numeric_limits<double>::quiet_NaN());
            accums_.resize(num_accums_);
        }
        // Deferred-tick: if there is a pending clock tick at a strictly smaller
        // index than the incoming value, emit it now before buffering the value.
        // Equal index: do NOT emit — the value still belongs in the pending bucket.
        if (pending_clock_tick_ &&
                static_cast<std::int64_t>(k) > static_cast<std::int64_t>(pending_clock_tick_index_)) {
            do_clock_tick(pending_clock_tick_index_);
            pending_clock_tick_ = false;
        }
        // Buffer the event (copy values) so the clock tick can drain it correctly.
        // Buffering handles the case where events arrive out of timeline order
        // (e.g., Delay shifting indices inside the same compiled graph).
        std::size_t nc = p_.plan.empty() ? (num_val_cols >= 1 ? 1u : 0u) : num_accums_;
        clock_value_buf_.push_back({k, std::vector<double>(vals, vals + std::min(nc, num_val_cols))});
    }

    // ByClock: a clock event registers a pending tick at index k. The actual
    // bucket is emitted only when a STRICTLY LATER event arrives (value or clock),
    // ensuring that same-index value events are included regardless of Pipeline
    // input order (MergeSource tie-breaking by source index).
    //
    // do_clock_tick(k) performs the immediate emission: drains buffered value
    // events with index <= k into the accumulator, emits/fills, and resets.
    void push_by_clock_tick(Index k) {
        // Deferred-tick: if there is already a pending tick, the new tick at k
        // proves a strictly later event has arrived — emit the earlier pending tick.
        if (pending_clock_tick_ &&
                static_cast<std::int64_t>(k) > static_cast<std::int64_t>(pending_clock_tick_index_)) {
            do_clock_tick(pending_clock_tick_index_);
            pending_clock_tick_ = false;
        }
        // Register k as the new pending tick (replace if same index, defer if later).
        pending_clock_tick_ = true;
        pending_clock_tick_index_ = k;
    }

    // Immediately emit the clock bucket at index k. Drains value events with
    // index <= k from the buffer, accumulates them, emits the reducer output or
    // the fill value, then resets the accumulator state.
    void do_clock_tick(Index k) {
        // Drain buffered value events with index <= k (they belong to this bucket).
        auto rem_it = clock_value_buf_.begin();
        for (auto it = clock_value_buf_.begin(); it != clock_value_buf_.end(); ++it) {
            if (static_cast<std::int64_t>(it->first) <= static_cast<std::int64_t>(k)) {
                // This event belongs to the closing bucket: accumulate it.
                const auto& row = it->second;
                if (!p_.plan.empty()) {
                    for (std::size_t c = 0; c < num_accums_ && c < row.size(); ++c)
                        accums_[c].add(row[c]);
                } else {
                    if (!row.empty()) accums_[0].add(row[0]);
                }
                // max_age is intentionally restricted to scalar last-value as-of
                // joins. A finite source observation updates its event-time stamp;
                // NaNs do not refresh freshness.
                if (p_.max_age >= 0 && !row.empty() &&
                        !screamer::isnan2(row[0]) &&
                        (!have_last_finite_ || it->first > last_finite_index_)) {
                    have_last_finite_ = true;
                    last_finite_index_ = it->first;
                }
                // count_in_bucket_ is not used for ByClock decision logic (has_real
                // checks accum.count instead); reset happens at the end of each tick.
            } else {
                // Future event: keep in buffer. Guard against self-move: if rem_it
                // and it point to the same element (no elements were drained yet),
                // skip the assignment to avoid std::vector self-move UB.
                if (rem_it != it) *rem_it = std::move(*it);
                ++rem_it;
            }
        }
        clock_value_buf_.erase(rem_it, clock_value_buf_.end());

        // I1 fix (ByClock only): for fill='carry' as-of semantics, a bucket whose
        // only value events are NaN should carry the last real value rather than
        // emit NaN. Use count > 0 (non-NaN events) rather than has_any() to decide
        // between emit and carry. Other modes (ByIndex/ByCount/ByCumulative) use
        // has_any() and are unaffected (do_clock_tick is ByClock-exclusive).
        bool has_real = false;
        for (const auto& a : accums_) if (a.count > 0) { has_real = true; break; }

        // Check after draining the current bucket: an observation in (prev, k]
        // can still be too old at a sparse clock tick. Preserve the leading
        // carry behavior before the first finite observation; only an observed
        // value can expire.
        const bool stale = p_.max_age >= 0 && have_last_finite_ &&
            !clock_value_is_fresh(k);
        if (stale) {
            downstream_.push(Frame<Index>{k, nan_row_.data(), nan_row_.size()});
        } else if (has_real) {
            emit(k);
        } else {
            // Empty bucket (no events, or all-NaN events): apply fill policy.
            if (p_.fill == ResampleFill::Carry && have_emitted_) {
                downstream_.push(Frame<Index>{k, last_emitted_.data(), last_emitted_.size()});
            } else if (p_.fill == ResampleFill::Nan) {
                downstream_.push(Frame<Index>{k, nan_row_.data(), nan_row_.size()});
            }
            // fill='skip': no emission.
        }
        reset_accums();
        count_in_bucket_ = 0;
        prev_clock_tick_set_ = true;
        prev_clock_tick_ = k;
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
    bool have_last_finite_ = false;
    Index last_finite_index_{};
    std::int64_t bucket_ = 0;
    Index cur_label_{};
    std::int64_t count_in_bucket_ = 0;
    Index first_index_{}, last_index_{};
    double cum_driver_ = 0.0;         // ByCumulative: running driver accumulator
    bool prev_clock_tick_set_ = false;  // ByClock: whether a clock tick has fired
    Index prev_clock_tick_{};           // ByClock: index of the last clock tick
    // ByClock value-event buffer: stores (index, values) pairs for deferred
    // accumulation. Clock ticks drain events with index <= tick; the rest stay.
    // Enables correct operation when value events arrive with shifted indices
    // (e.g., from a Delay node in the same compiled graph causing out-of-order
    // event delivery at the ResampleNode).
    std::vector<std::pair<Index, std::vector<double>>> clock_value_buf_;
    // ByClock 2-port wiring (created lazily on first port() call).
    std::vector<ClockPort> clock_ports_;
    std::size_t clock_port_flush_count_ = 0;
    // ByClock deferred-tick state (C1 tie-order fix): a clock tick at
    // pending_clock_tick_index_ is held until a strictly later event arrives,
    // so same-index value events (regardless of Pipeline input order) are always
    // included in the bucket before it closes.
    bool pending_clock_tick_ = false;
    Index pending_clock_tick_index_{};
};

}} // namespace screamer::dag
#endif
