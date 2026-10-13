#ifndef SCREAMER_DAG_RESAMPLE_NODE_H
#define SCREAMER_DAG_RESAMPLE_NODE_H

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
#include "screamer/common/checked_arith.h"
#include "screamer/common/float_info.h"
#include "screamer/dag/frame.h"
#include "screamer/dag/resample_params.h"
#include <algorithm>
#include <cstddef>
#include <deque>
#include <string>
#include <utility>

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
// Single-column mode (plan empty): a width-1 stream uses one accumulator. A wider
// stream switches to a per-column plan on its first frame (make_per_column_plan)
// until reset().
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
        : p_(p), plan0_(p.plan), downstream_(downstream),
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
            learn_plan(make_ohlcv_bars_plan(val_width));
        }
        if (p_.mode == ResampleMode::ByCumulative) {
            // ByCumulative: input frame must be width 2 (value col + driver col).
            // The driver is always the LAST column; value columns are all but last.
            // In the current implementation value is width 1 (single value column)
            // so the frame is always [value, driver] (width 2).
            if (f.width != 2)
                throw std::invalid_argument(
                    "a cumulative resample takes [value, driver] (2 columns), "
                    "but its input has " + std::to_string(f.width));
            push_by_cumulative(f.index, f.values[0], f.values[1]);
            return;
        }
        // ByClock is always wired via 2-port (ClockPort) in the compiled graph;
        // push() is never called for ByClock mode in practice.
        if (p_.plan.empty()) {
            // Single-column path: width must be 1.
            if (f.width != 1) {
                // A wide stream under a single-column agg: aggregate every column
                // on its own. The plan lasts until reset(), so a reused Pipeline
                // can take a different width next call.
                learn_plan(make_per_column_plan(p_.agg, f.width));
                reset_accums();
                if (p_.mode == ResampleMode::ByIndex) push_by_index_multi(f.index, f.values);
                else push_by_count_multi(f.index, f.values);
                return;
            }
            if (p_.mode == ResampleMode::ByIndex)
                push_by_index(f.index, f.values[0]);
            else
                push_by_count(f.index, f.values[0]);
        } else {
            // Multi-column plan path: input width must match num_accums_.
            if (f.width != num_accums_)
                throw std::invalid_argument(
                    "a resample plan takes " + std::to_string(num_accums_) +
                    " column(s), but its input has " + std::to_string(f.width));
            if (p_.mode == ResampleMode::ByIndex)
                push_by_index_multi(f.index, f.values);
            else
                push_by_count_multi(f.index, f.values);
        }
    }

    void flush() override {
        if (p_.mode == ResampleMode::ByIndex) {
            if (has_any()) emit(cur_label());
        } else if (p_.mode == ResampleMode::ByClock) {
            // The end of input: no value event can still come, so every pending
            // tick emits, in order, each with the values up to its index. The
            // values after the last tick have no tick to close them and are
            // dropped (the clock drives emission; no clock tick = no bar).
            while (!ticks_.empty()) {
                const Index t = ticks_.front();
                take_buffered_upto(t);
                emit_clock_bar(t);
                ticks_.pop_front();
            }
            clear_buffer();
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
        if (learned_) {                // forget the plan learned from the last stream
            p_.plan = plan0_;
            learned_ = false;
            num_accums_ = p_.plan.empty() ? 1u : resample_plan_input_cols(p_.plan);
            set_out_width(resample_output_width(p_));
            accums_.resize(deferred_plan() ? 0u : num_accums_);
        }
        reset_accums();
        started_ = false;
        bucket_ = 0;
        cur_label_ = Index{};
        label_ok_ = true;
        has_bar_wm_ = false;
        count_in_bucket_ = 0;
        first_index_ = last_index_ = Index{};
        have_emitted_ = false;
        std::fill(last_emitted_.begin(), last_emitted_.end(),
                  std::numeric_limits<double>::quiet_NaN());
        have_last_finite_ = false;
        last_finite_index_ = Index{};
        cum_driver_ = 0.0;
        clock_port_flush_count_ = 0;
        ticks_.clear();
        clear_buffer();
        has_vfront_ = has_cfront_ = has_gfront_ = has_fwd_ = has_last_tick_ = false;
        vfront_ = cfront_ = gfront_ = fwd_ = last_tick_ = Index{};
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
    // any event has anchored a bucket (not started_), the first event can still open
    // the bucket that holds w, so the forward is at most that bucket's label (a
    // label-left bucket starts at or below w). The end-of-input watermark (the
    // int64 maximum, sent at a flush) emits the current bucket only if it has
    // data, as flush() does: trailing empty buckets are never filled.
    void on_watermark(Index w) override {
        advance(w);
        Index fwd = w;
        if (p_.mode == ResampleMode::ByIndex) {
            if (started_) { if (cur_label_ < fwd) fwd = cur_label_; }
            else if (w != std::numeric_limits<Index>::max()) {
                const Index first = static_cast<Index>(
                    resample_label_bound(resample_bucket_bound(w, p_), p_));
                if (first < fwd) fwd = first;
            }
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
        // The end-of-input watermark emits the current bucket if it has data
        // and fills no empty bucket, as flush() does: they are all trailing.
        const bool end = now == std::numeric_limits<Index>::max();
        std::int64_t target = end ? bucket_ + 1 : resample_bucket_bound(now, p_);
        if (target <= bucket_) return;
        if (p_.fill != ResampleFill::Skip && !end)
            check_fill_gap(bucket_, target, !has_any(), p_.max_fill);
        if (has_any()) emit(cur_label());
        else if (p_.fill != ResampleFill::Skip && !end) emit_fill(cur_label());
        if (p_.fill != ResampleFill::Skip && !end)
            for (std::int64_t b = bucket_ + 1; b < target; ++b) emit_fill(label_for(b));
        bucket_ = target; reset_accums(); set_index_label(target);
        if (end) started_ = false;     // as after a flush: the next event anchors
    }

    // ByClock 2-port interface: port(0) = value stream, port(1) = clock stream.
    // Used by the compiled graph to wire two separate input nodes directly into
    // the ResampleNode without combining them into a single frame first.
    //
    // A bar at clock tick t holds the value events with index in (previous
    // tick, t]. The ports may lag each other (a Filter or a threshold Resample
    // releases values late; a clock can come out of a Resample), so the node
    // proves each step from the ports' frontiers, never from arrival order:
    // - tick t emits once no value event at or below t can still come: a value
    //   event or a value-port watermark above t, or (value_in_order) any index
    //   above t seen on a port that is never ahead of the merge;
    // - a value event joins the first pending tick at or above its index, or,
    //   with no tick pending, the next tick once no tick below its index can
    //   still come (the clock frontier, a clock event or watermark, has reached
    //   it); until then it waits in a buffer.
    // Each port's watermark moves its frontier, and the node forwards the
    // lowest index it may still emit: the first pending tick, else the clock
    // frontier. So batch, lazy and live give the same bars whatever the drain
    // and advance() timing, and advance(now) releases other outputs' rows
    // while the clock is silent.
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
            if (port_idx == 0) node.push_by_clock_value(f.index, f.values, f.width);
            else node.push_by_clock_tick(f.index);         // the clock value is ignored
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
        void on_watermark(Index w) override { node.on_clock_watermark(port_idx, w); }
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

    // ByClock: how each port's producer relates to the merge of the graph's
    // inputs (CompiledGraph sets it at wiring). `in_order`: the port never
    // receives an index below one the merge has passed (an Input, or functors,
    // Select, Dropna, a lag or a forward Delay of one), so any index seen on a
    // never-ahead port bounds it. `never_ahead`: the port's indexes never pass
    // the merge (anything but a forward Delay). These relations hold only
    // while one merge feeds the graph in index order (set_merge_order: a batch
    // run, the lazy driver); a live session's pushes may come in any order
    // across inputs, so there each step is proven from the ports' own
    // frontiers and watermarks.
    void set_clock_ports(bool value_in_order, bool value_never_ahead,
                         bool clock_in_order, bool clock_never_ahead) {
        value_in_order_ = value_in_order;
        clock_in_order_ = clock_in_order;
        ahead_ok_[0] = value_never_ahead;
        ahead_ok_[1] = clock_never_ahead;
    }

    // Whether one merge feeds the graph's inputs in index order (see above).
    void set_merge_order(bool on) { merge_order_ = on; }

private:
    // ohlcv_bars without a plan: the plan comes from the first frame's width.
    bool deferred_plan() const {
        return p_.agg == ResampleAgg::OhlcvBars && p_.plan.empty();
    }

    void set_out_width(std::size_t w) {
        out_width_ = w;
        out_.assign(w, 0.0);
        last_emitted_.assign(w, std::numeric_limits<double>::quiet_NaN());
        nan_row_.assign(w, std::numeric_limits<double>::quiet_NaN());
    }

    // Installs a plan built from the stream's width. It lasts until reset(),
    // which restores the constructed plan, so a reused graph can take another
    // width.
    void learn_plan(std::vector<ResamplePlanEntry> plan) {
        p_.plan = std::move(plan);
        learned_ = true;
        num_accums_ = resample_plan_input_cols(p_.plan);
        set_out_width(p_.plan.size());
        accums_.resize(num_accums_);
    }

    bool has_any() const {
        for (const auto& a : accums_) if (a.has) return true;
        return false;
    }

    void reset_accums() {
        for (auto& a : accums_) a.reset();
    }

    void push_by_index(Index k, double v) {
        const std::int64_t nb = resample_bucket(static_cast<std::int64_t>(k), p_);
        check_label(nb);
        if (!started_) {
            started_ = true; bucket_ = nb; reset_accums(); set_index_label(nb);
        } else if (nb != bucket_) {
            move_to_bucket(nb);
        }
        accums_[0].add(v);
    }

    // Closes the current bucket and fills the empty ones before bucket nb
    // (fill nan or carry), after checking the gap against max_fill.
    void move_to_bucket(std::int64_t nb) {
        if (p_.fill != ResampleFill::Skip) {
            check_fill_gap(bucket_, nb, !has_any(), p_.max_fill);
            if (has_any()) emit(cur_label());
            else emit_fill(cur_label());
            for (std::int64_t b = bucket_ + 1; b < nb; ++b) emit_fill(label_for(b));
        } else if (has_any()) {
            emit(cur_label());
        }
        bucket_ = nb; reset_accums(); set_index_label(nb);
    }

    void push_by_count(Index k, double v) {
        if (count_in_bucket_ == 0) { first_index_ = k; open_bar_watermark(k); }
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
        const std::int64_t nb = resample_bucket(static_cast<std::int64_t>(k), p_);
        check_label(nb);
        if (!started_) {
            started_ = true; bucket_ = nb; reset_accums(); set_index_label(nb);
        } else if (nb != bucket_) {
            move_to_bucket(nb);
        }
        for (std::size_t c = 0; c < num_accums_; ++c)
            accums_[c].add(vals[c]);
    }

    void push_by_count_multi(Index k, const double* vals) {
        if (count_in_bucket_ == 0) { first_index_ = k; open_bar_watermark(k); }
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
        if (count_in_bucket_ == 0) { first_index_ = k; open_bar_watermark(k); }
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

    // ByClock frontiers (see ClockPort). vfront_: the highest value index or
    // value-port watermark; cfront_: the same for the clock port; gfront_:
    // the highest index seen on a port that is never ahead of the merge.
    void note_front(std::size_t port, Index k) {
        Index& f = port == 0 ? vfront_ : cfront_;
        bool& has = port == 0 ? has_vfront_ : has_cfront_;
        if (!has || k > f) { f = k; has = true; }
        if (ahead_ok_[port] && (!has_gfront_ || k > gfront_)) { gfront_ = k; has_gfront_ = true; }
    }

    // No value event at or below t can still come.
    bool value_past(Index t) const {
        return (has_vfront_ && vfront_ > t) ||
               (merge_order_ && value_in_order_ && has_gfront_ && gfront_ > t);
    }

    // Every tick still to come is at or above k.
    bool clock_reaches(Index k) const {
        return (has_cfront_ && cfront_ >= k) ||
               (merge_order_ && clock_in_order_ && has_gfront_ && gfront_ >= k);
    }

    // A value event that may join the current bar now: the bar of the first
    // pending tick (at or above k, since ticks below the value frontier have
    // emitted), or, with no tick pending, the bar of the next tick.
    bool joins_now(Index k) const {
        return ticks_.empty() ? clock_reaches(k) : k <= ticks_.front();
    }

    // Folds one value event into the current bar.
    void add_clock_value(Index k, const double* row, std::size_t n) {
        if (!p_.plan.empty()) {
            for (std::size_t c = 0; c < num_accums_ && c < n; ++c) accums_[c].add(row[c]);
        } else if (n) {
            accums_[0].add(row[0]);
        }
        // max_age is intentionally restricted to scalar last-value as-of
        // joins. A finite source observation updates its event-time stamp;
        // NaNs do not refresh freshness.
        if (p_.max_age >= 0 && n && !screamer::isnan2(row[0]) &&
                (!have_last_finite_ || k > last_finite_index_)) {
            have_last_finite_ = true;
            last_finite_index_ = k;
        }
    }

    // Moves the buffered values at or below t (a prefix: values arrive in
    // index order) into the current bar.
    void take_buffered_upto(Index t) {
        while (buf_head_ < buf_idx_.size() && buf_idx_[buf_head_] <= t) take_buffered();
    }
    void take_buffered() {
        add_clock_value(buf_idx_[buf_head_], buf_vals_.data() + buf_head_ * buf_width_, buf_width_);
        if (++buf_head_ == buf_idx_.size()) clear_buffer();   // drained: reuse the storage
    }

    // Empties the buffer and keeps its storage.
    void clear_buffer() {
        buf_idx_.clear();
        buf_vals_.clear();
        buf_head_ = 0;
    }

    // Before an append, drops the taken prefix once it is at least half the
    // buffer, so a buffer that never drains fully stays within twice its
    // waiting values.
    void compact_buffer() {
        if (buf_head_ == 0 || buf_head_ * 2 < buf_idx_.size()) return;
        buf_idx_.erase(buf_idx_.begin(), buf_idx_.begin() + static_cast<std::ptrdiff_t>(buf_head_));
        buf_vals_.erase(buf_vals_.begin(),
                        buf_vals_.begin() + static_cast<std::ptrdiff_t>(buf_head_ * buf_width_));
        buf_head_ = 0;
    }

    // Emits every pending tick that is proven complete and moves the buffered
    // values that the current bar may take.
    void settle_clock() {
        while (!ticks_.empty()) {
            const Index t = ticks_.front();
            take_buffered_upto(t);
            if (!value_past(t)) return;
            emit_clock_bar(t);
            ticks_.pop_front();
        }
        while (buf_head_ < buf_idx_.size() && clock_reaches(buf_idx_[buf_head_])) take_buffered();
    }

    // A watermark on port 0 (value) or 1 (clock): no later event below w.
    void on_clock_watermark(std::size_t port, Index w) {
        note_front(port, w);
        settle_clock();
        // The lowest index a bar may still have: the first pending tick, else
        // the clock frontier (every tick still to come is at or above it).
        Index f;
        const bool g = merge_order_ && clock_in_order_ && has_gfront_;
        if (!ticks_.empty()) f = ticks_.front();
        else if (has_cfront_ && g) f = gfront_ > cfront_ ? gfront_ : cfront_;
        else if (has_cfront_) f = cfront_;
        else if (g) f = gfront_;
        else return;
        if (has_fwd_ && f <= fwd_) return;
        fwd_ = f; has_fwd_ = true;
        downstream_.on_watermark(f);
    }

    // ByClock: a value event. vals holds num_val_cols values.
    void push_by_clock_value(Index k, const double* vals, std::size_t num_val_cols) {
        if (p_.max_age >= 0 && num_val_cols != 1) {
            throw std::invalid_argument(
                "a clock resample with max_age takes 1 value column, but its "
                "input has " + std::to_string(num_val_cols));
        }
        if (deferred_plan()) {
            // ohlcv_bars: [open, high, low, close, volumes...] from the value
            // width; make_ohlcv_bars_plan checks it.
            learn_plan(make_ohlcv_bars_plan(num_val_cols));
        } else if (p_.plan.empty() && num_val_cols > 1) {
            // Multi-column single-agg ByClock: every column is reduced on its own,
            // as in resample(multi_col, clock=..., agg='last'); 'ohlc' gives
            // (open, high, low, close) per column.
            learn_plan(make_per_column_plan(p_.agg, num_val_cols));
        } else if (!p_.plan.empty() && num_val_cols != num_accums_) {
            throw std::invalid_argument(
                "a resample plan takes " + std::to_string(num_accums_) +
                " column(s), but its input has " + std::to_string(num_val_cols));
        }
        const std::size_t nc = p_.plan.empty() ? (num_val_cols >= 1 ? 1u : 0u) : num_accums_;
        const std::size_t n = std::min(nc, num_val_cols);
        note_front(0, k);
        settle_clock();                     // ticks below k are complete
        if (buf_head_ == buf_idx_.size() && joins_now(k)) {
            add_clock_value(k, vals, n);
            return;
        }
        // A tick below k may still come: wait in the buffer. (settle_clock
        // above already emitted every tick this value could complete.)
        compact_buffer();
        buf_width_ = n;
        buf_idx_.push_back(k);
        buf_vals_.insert(buf_vals_.end(), vals, vals + n);
    }

    // ByClock: a clock event at k. Ticks at one index give one bar, and a
    // tick at or below one that has emitted adds no bar, whatever the arrival
    // order of the two ports.
    void push_by_clock_tick(Index k) {
        note_front(1, k);
        const bool fresh = ticks_.empty() ? !has_last_tick_ || k > last_tick_
                                          : k > ticks_.back();
        if (fresh) ticks_.push_back(k);
        settle_clock();
    }

    // Emits the bar of tick k from the current bar's values (the reducer
    // output, or the fill when it had no finite value), then starts the next.
    void emit_clock_bar(Index k) {
        bool has_real = false;
        for (const auto& a : accums_) if (a.count > 0) { has_real = true; break; }
        // An observation in (prev, k] can still be too old at a sparse clock
        // tick. Preserve the leading carry behavior before the first finite
        // observation; only an observed value can expire.
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
        last_tick_ = k;
        has_last_tick_ = true;
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
        if (resample_start(nb, p_, l)) open_bar_watermark(static_cast<Index>(l));
    }

    // A new bar opens at w (its start for every=, its first event for count=
    // and threshold=): the node emits no row below w from here, so w is a
    // watermark. Sent once per bar, never above the event that opened the bar
    // (the node stays never ahead of the input merge). A consumer (an output
    // alignment, a join) then holds at most the rows of one open bar.
    void open_bar_watermark(Index w) {
        if (has_bar_wm_ && w <= bar_wm_) return;
        bar_wm_ = w;
        has_bar_wm_ = true;
        downstream_.on_watermark(w);
    }
    Index cur_label() const {
        if (!label_ok_) throw_resample_label(p_);
        return cur_label_;
    }

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
        if (carries()) carry_cells();
        downstream_.push(Frame<Index>{label, out_.data(), out_.size()});
    }

    // fill="carry" under freq= and clock=: each NaN cell of a bar takes its
    // column's last non-NaN value (NaN before the column has one), and an
    // empty bar repeats those values. count= and threshold= bars are never
    // empty, so fill= leaves them as they are.
    bool carries() const {
        return p_.fill == ResampleFill::Carry &&
               (p_.mode == ResampleMode::ByIndex || p_.mode == ResampleMode::ByClock);
    }
    void carry_cells() {
        for (std::size_t i = 0; i < out_.size(); ++i) {
            if (std::isnan(out_[i])) out_[i] = last_emitted_[i];
            else last_emitted_[i] = out_[i];
        }
        have_emitted_ = true;
    }

    void emit_fill(Index label) {
        if (p_.fill == ResampleFill::Nan) {
            downstream_.push(Frame<Index>{label, nan_row_.data(), nan_row_.size()});
        } else if (have_emitted_) {
            downstream_.push(Frame<Index>{label, last_emitted_.data(), last_emitted_.size()});
        }
    }

    ResampleParams p_;
    std::vector<ResamplePlanEntry> plan0_;   // the constructed plan, restored by reset()
    Sink<Index>& downstream_;
    std::size_t out_width_;
    bool learned_ = false;            // plan was built from the stream's width; reverted by reset()
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
    bool label_ok_ = true;       // cur_label_ is the open bar's label (not clamped)
    Index bar_wm_{};             // the last open_bar_watermark
    bool has_bar_wm_ = false;
    std::int64_t count_in_bucket_ = 0;
    Index first_index_{}, last_index_{};
    double cum_driver_ = 0.0;         // ByCumulative: running driver accumulator
    // ByClock 2-port wiring (created lazily on first port() call).
    std::vector<ClockPort> clock_ports_;
    std::size_t clock_port_flush_count_ = 0;
    // ByClock state (see ClockPort). The current bar's values are in accums_.
    std::deque<Index> ticks_;          // pending ticks, increasing
    std::vector<Index> buf_idx_;       // values that wait for the clock frontier,
    std::vector<double> buf_vals_;     // their values, buf_width_ each, from
    std::size_t buf_head_ = 0;         // buf_head_ on (the prefix is taken)
    std::size_t buf_width_ = 0;
    Index vfront_{}, cfront_{}, gfront_{}, fwd_{}, last_tick_{};
    bool has_vfront_ = false, has_cfront_ = false, has_gfront_ = false, has_fwd_ = false;
    bool has_last_tick_ = false;      // a tick has emitted (last_tick_)
    bool value_in_order_ = false, clock_in_order_ = false;
    bool merge_order_ = false;        // the inputs come from one index-ordered merge
    bool ahead_ok_[2] = {false, false};
};

}} // namespace screamer::dag
#endif
