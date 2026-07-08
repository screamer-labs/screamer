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

namespace screamer { namespace dag {

// One bucketing clock shared across N input ports, each reduced by its own
// EvalOp. Emits one row per bar: the N reducers' outputs concatenated (width =
// sum of reducer n_out()), labelled by the bucket. Generalizes
// GenericResampleNode (1 port) to N ports; mirrors CombineLatestNode's per-port
// Sink fan-in and its all-ports-flushed coalescing. ByIndex only in v1.
// NaN-ignore per port: NaN samples are not fed to that port's reducer; a port
// with no finite sample in a bar emits NaN for its columns; a bar with any event
// (on any port) emits, empty bars follow the fill policy.
template <class Index>
class MultiResampleNode {
public:
    MultiResampleNode(ResampleParams clock, std::vector<EvalOp*> reducers,
                      Sink<Index>& downstream)
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
        flushed_.assign(reducers_.size(), false);
        col_out_.resize(reducers_.size());
        for (std::size_t i = 0; i < reducers_.size(); ++i)
            col_out_[i].assign(reducers_[i]->n_out(),
                               std::numeric_limits<double>::quiet_NaN());
        ports_.reserve(reducers_.size());   // reserve so port() addresses stay stable
        for (std::size_t i = 0; i < reducers_.size(); ++i)
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

    void reset() {
        clear_bucket();
        started_ = false;
        bucket_ = 0;
        cur_label_ = Index{};
        have_emitted_ = false;
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
        void push(const Frame<Index>& f) override { node.on_port(idx, f); }
        void flush() override { node.flush_port(idx); }
    };
    friend struct Port;

    void on_port(std::size_t i, const Frame<Index>& f) {
        if (f.width != 1)
            throw std::runtime_error("dag::MultiResampleNode: ports are width-1");
        if (p_.mode != ResampleMode::ByIndex)
            throw std::runtime_error("dag::MultiResampleNode: ByIndex only");
        Index k = f.index;
        double v = f.values[0];
        std::int64_t nb = floordiv(static_cast<std::int64_t>(k) - p_.origin, p_.width);
        if (!started_) {
            started_ = true; bucket_ = nb; clear_bucket(); set_index_label(nb);
        } else if (nb != bucket_) {
            if (has_) emit(cur_label_);
            // Fill internal gaps; trailing empties are handled by advance()/flush().
            if (p_.fill != ResampleFill::Skip)
                for (std::int64_t b = bucket_ + 1; b < nb; ++b) emit_fill(label_for(b));
            bucket_ = nb; clear_bucket(); set_index_label(nb);
        }
        add(i, v);
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
        if (flushed_count_ < flushed_.size()) return;   // wait until ALL ports flush
        if (has_) emit(cur_label_);                     // trailing partial, once
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
    std::vector<Port> ports_;
    std::vector<bool> flushed_;                  // which ports flushed this cycle
    std::size_t flushed_count_ = 0;
};

}} // namespace screamer::dag
#endif
