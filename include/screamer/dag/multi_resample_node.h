#ifndef SCREAMER_DAG_MULTI_RESAMPLE_NODE_H
#define SCREAMER_DAG_MULTI_RESAMPLE_NODE_H

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
#include "screamer/common/eval_op.h"
#include "screamer/common/float_info.h"
#include "screamer/dag/frame.h"
#include "screamer/dag/resample_params.h"
#include "screamer/dag/resettable.h"

namespace screamer { namespace dag {

// One bucketing clock shared across N input ports, each reduced by its own
// EvalOp. Emits one row per bar: the N reducers' outputs concatenated (width =
// sum of reducer n_out()), labelled by the bucket. Generalizes
// GenericResampleNode (1 port) to N ports; mirrors CombineLatestNode's per-port
// Sink fan-in and its all-ports-flushed coalescing. Supports ByIndex and ByCount
// (count = a bar every N DISTINCT ticks/indices, not per-port pushes) plus an
// optional trailing clock port: an extra input that only advances the bucket
// (feeds no reducer, adds no column) so a clock/timestamp stream can finalize
// empty time-bars straight from the data (ByIndex only).
// NaN-ignore per port: NaN samples are not fed to that port's reducer; a port
// with no finite sample in a bar emits NaN for its columns; a bar with any event
// (on any port) emits, empty bars follow the fill policy.
// Derives from Resettable so CompiledGraph can reset it via a single polymorphic
// list without knowing its concrete type (MultiResampleNode is not a Sink).
template <class Index>
class MultiResampleNode : public Resettable {
public:
    MultiResampleNode(ResampleParams clock, std::vector<EvalOp*> reducers,
                      bool has_clock, Sink<Index>& downstream)
        : p_(clock), reducers_(std::move(reducers)), downstream_(downstream) {
        if (reducers_.empty())
            throw std::runtime_error("dag::MultiResampleNode: no reducers");
        offsets_.assign(reducers_.size() + 1, 0);
        for (std::size_t i = 0; i < reducers_.size(); ++i) {
            if (!reducers_[i])
                throw std::runtime_error("dag::MultiResampleNode: null reducer");
            if (reducers_[i]->n_in() != 1)
                throw std::runtime_error(
                    "dag::MultiResampleNode: each reducer needs n_in()==1");
            offsets_[i + 1] = offsets_[i] + reducers_[i]->n_out();
        }
        width_ = offsets_.back();
        out_.assign(width_, 0.0);
        last_emitted_.assign(width_, 0.0);
        nan_row_.assign(width_, std::numeric_limits<double>::quiet_NaN());
        fed_.assign(reducers_.size(), false);
        col_out_.resize(reducers_.size());
        for (std::size_t i = 0; i < reducers_.size(); ++i)
            col_out_[i].assign(reducers_[i]->n_out(),
                               std::numeric_limits<double>::quiet_NaN());
        has_clock_ = has_clock;
        clock_idx_ = reducers_.size();      // the extra trailing port index
        std::size_t nports = reducers_.size() + (has_clock_ ? 1 : 0);
        flushed_.assign(nports, false);     // clock port's flush is counted too
        ports_.reserve(nports);             // reserve so port() addresses stay stable
        for (std::size_t i = 0; i < nports; ++i)
            ports_.emplace_back(*this, i);
        reset();
    }

    // Non-movable/copyable: the ports hold a reference back to this node.
    MultiResampleNode(const MultiResampleNode&) = delete;
    MultiResampleNode& operator=(const MultiResampleNode&) = delete;
    MultiResampleNode(MultiResampleNode&&) = delete;
    MultiResampleNode& operator=(MultiResampleNode&&) = delete;

    Sink<Index>& port(std::size_t i) { return ports_[i]; }
    std::size_t width() const { return width_; }

    void reset() override {
        clear_bucket();
        started_ = false;
        bucket_ = 0;
        cur_label_ = Index{};
        have_emitted_ = false;
        ticks_ = 0;
        std::fill(flushed_.begin(), flushed_.end(), false);
        flushed_count_ = 0;
    }

    // Time-driven finalization (mirror GenericResampleNode::advance).
    void advance(Index now) {
        if (p_.mode != ResampleMode::ByIndex) return;   // count mode: no time meaning
        if (!started_) return;                           // nothing anchored yet
        std::int64_t target =
            floordiv(static_cast<std::int64_t>(now) - p_.origin, p_.width);
        if (target <= bucket_) return;                   // still inside current bucket
        if (has_) emit(cur_label_);
        else if (p_.fill != ResampleFill::Skip) emit_fill(cur_label_);
        if (p_.fill != ResampleFill::Skip)
            for (std::int64_t b = bucket_ + 1; b < target; ++b) emit_fill(label_for(b));
        bucket_ = target; clear_bucket(); set_index_label(target);
    }

private:
    // A single input port: routes an event to its owning node with its port index.
    struct Port : Sink<Index> {
        MultiResampleNode& node;
        std::size_t idx;
        Port(MultiResampleNode& n, std::size_t i) : node(n), idx(i) {}
        void push(const Frame<Index>& f) override {
            if (node.has_clock_ && idx == node.clock_idx_) node.on_clock(f);
            else node.on_port(idx, f);
        }
        void flush() override { node.flush_port(idx); }  // flushed_ covers all ports
        // Each port accepts one value per event; output is owned by the parent node.
        std::size_t n_in()  const override { return 1; }
        std::size_t n_out() const override { return 0; }
    };
    friend struct Port;

    void on_port(std::size_t i, const Frame<Index>& f) {
        if (f.width != 1)
            throw std::runtime_error("dag::MultiResampleNode: ports are width-1");
        if (p_.mode == ResampleMode::ByIndex) advance_index(f.index);
        else                                  advance_count(f.index);
        add(i, f.values[0]);
    }

    // The clock port advances the bucket without contributing a column. ByIndex
    // only (count mode counts data ticks; a pure clock has no role there).
    void on_clock(const Frame<Index>& f) {
        if (f.width != 1)
            throw std::runtime_error("dag::MultiResampleNode: clock port is width-1");
        if (p_.mode == ResampleMode::ByIndex) advance_index(f.index);
    }

    // Bucket-by-index boundary logic. The added `else if (fill) emit_fill` branch
    // closes an EMPTY current bucket (a clock tick crossing a bar with no trades)
    // per policy. In column-only graphs the current bucket always has an event at
    // a crossing (has_ is true), so that branch is inert there - no Task-5 change.
    void advance_index(Index k) {
        std::int64_t nb = floordiv(static_cast<std::int64_t>(k) - p_.origin, p_.width);
        if (!started_) {
            started_ = true; bucket_ = nb; clear_bucket(); set_index_label(nb);
        } else if (nb != bucket_) {
            if (has_) emit(cur_label_);
            else if (p_.fill != ResampleFill::Skip) emit_fill(cur_label_);
            if (p_.fill != ResampleFill::Skip)
                for (std::int64_t b = bucket_ + 1; b < nb; ++b) emit_fill(label_for(b));
            bucket_ = nb; clear_bucket(); set_index_label(nb);
        }
    }

    // Count DISTINCT ticks. A bar holds `count` ticks; because a tick is complete
    // only when the next distinct index arrives, a full bar closes on the NEXT new
    // tick (deferred), never eagerly mid-tick. Label = first (Left) or last (Right)
    // tick index of the closed bar.
    void advance_count(Index k) {
        if (!started_) {
            started_ = true; ticks_ = 1; first_index_ = last_index_ = k; clear_bucket();
        } else if (k != last_index_) {              // a new distinct tick
            if (ticks_ >= p_.count) {               // current bar already full -> close
                if (has_) emit(count_label());
                clear_bucket(); ticks_ = 1; first_index_ = k;
            } else {
                ++ticks_;
            }
            last_index_ = k;
        }
        // k == last_index_: another port at the SAME tick -> no counter change
    }

    Index count_label() const {
        return static_cast<Index>(
            p_.label == ResampleLabel::Left ? first_index_ : last_index_);
    }

    void add(std::size_t i, double v) {
        has_ = true;                       // any event on any port
        if (screamer::isnan2(v)) return;   // ignore policy: do not feed NaN
        reducers_[i]->eval(&v, col_out_[i].data());
        fed_[i] = true;
    }

    void clear_bucket() {
        for (auto* r : reducers_) r->reset();
        // col_out_[i] is only read in emit() when fed_[i] is true, i.e. after
        // add() has written it via reducer eval; the !fed_[i] columns get NaN
        // directly in emit(). So no per-bar NaN-fill of col_out_ is needed here.
        std::fill(fed_.begin(), fed_.end(), false);
        has_ = false;
    }

    void emit(Index label) {
        for (std::size_t i = 0; i < reducers_.size(); ++i) {
            double* dst = out_.data() + offsets_[i];
            std::size_t w = offsets_[i + 1] - offsets_[i];
            if (fed_[i])
                std::copy(col_out_[i].begin(), col_out_[i].end(), dst);
            else
                std::fill(dst, dst + w, std::numeric_limits<double>::quiet_NaN());
        }
        if (p_.fill == ResampleFill::Carry) {
            last_emitted_.assign(out_.begin(), out_.end());
            have_emitted_ = true;
        }
        downstream_.push(Frame<Index>{label, out_.data(), width_});
    }

    // Emit a filler row for an empty bucket. Nan -> all-NaN row; Carry -> the
    // previous emitted row verbatim (skipped if nothing emitted yet).
    void emit_fill(Index label) {
        if (p_.fill == ResampleFill::Nan)
            downstream_.push(Frame<Index>{label, nan_row_.data(), width_});
        else if (have_emitted_)   // Carry
            downstream_.push(Frame<Index>{label, last_emitted_.data(), width_});
    }

    // Called once per input port at end-of-input. Each producer pushes its final
    // event just BEFORE flushing its own port, so we wait until EVERY port has
    // flushed before emitting the settled trailing partial bar once and
    // propagating the flush downstream. A per-port bitmask dedups repeat flushes.
    void flush_port(std::size_t i) {
        if (!flushed_[i]) { flushed_[i] = true; ++flushed_count_; }
        if (flushed_count_ < flushed_.size()) return;   // wait until ALL ports (incl clock) flush
        if (has_) emit(p_.mode == ResampleMode::ByIndex ? cur_label_ : count_label());
        clear_bucket();
        started_ = false;
        std::fill(flushed_.begin(), flushed_.end(), false);
        flushed_count_ = 0;
        downstream_.flush();
    }

    Index label_for(std::int64_t nb) const {
        std::int64_t start = p_.origin + nb * p_.width;
        return static_cast<Index>(
            p_.label == ResampleLabel::Left ? start : start + p_.width);
    }
    void set_index_label(std::int64_t nb) { cur_label_ = label_for(nb); }

    static std::int64_t floordiv(std::int64_t a, std::int64_t b) {
        std::int64_t q = a / b, r = a % b;
        if (r != 0 && ((r < 0) != (b < 0))) --q;
        return q;
    }

    ResampleParams p_;
    std::vector<EvalOp*> reducers_;
    Sink<Index>& downstream_;
    std::vector<std::size_t> offsets_;           // prefix widths per reducer
    std::size_t width_ = 0;
    std::vector<double> out_, last_emitted_, nan_row_;
    std::vector<std::vector<double>> col_out_;   // per-reducer latest output (reused)
    std::vector<bool> fed_;                      // reducer got >=1 finite sample
    bool has_ = false;                           // any event fell in current bucket
    bool started_ = false;
    bool have_emitted_ = false;                  // Carry: any real row emitted yet
    std::int64_t bucket_ = 0;
    Index cur_label_{};
    bool has_clock_ = false;                     // last port is a bucket-only clock
    std::size_t clock_idx_ = 0;                  // index of the clock port (if any)
    std::int64_t ticks_ = 0;                     // count mode: distinct ticks in current bar
    Index first_index_{}, last_index_{};         // count mode: bar's first/last tick index
    std::vector<Port> ports_;
    std::vector<bool> flushed_;                  // which ports flushed this cycle (incl clock)
    std::size_t flushed_count_ = 0;
};

}} // namespace screamer::dag
#endif
