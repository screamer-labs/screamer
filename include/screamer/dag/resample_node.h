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
        : p_(p), downstream_(downstream), out_(resample_width(p.agg)),
          last_emitted_(resample_width(p.agg)),
          nan_row_(resample_width(p.agg),
                   std::numeric_limits<double>::quiet_NaN()) { reset(); }

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

    void reset() override {
        acc_.reset();
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
        if (acc_.has) emit(cur_label_);
        else if (p_.fill != ResampleFill::Skip) emit_fill(cur_label_);
        // Trailing empty buckets strictly between the current one and `target`.
        if (p_.fill != ResampleFill::Skip)
            for (std::int64_t b = bucket_ + 1; b < target; ++b) emit_fill(label_for(b));
        // Move to the (still open, empty) bucket that contains `now`.
        bucket_ = target; acc_.reset(); set_index_label(target);
    }

private:
    void push_by_index(Index k, double v) {
        std::int64_t nb = floordiv(static_cast<std::int64_t>(k) - p_.origin, p_.width);
        if (!started_) {
            started_ = true; bucket_ = nb; acc_.reset(); set_index_label(nb);
        } else if (nb != bucket_) {
            if (acc_.has) emit(cur_label_);
            // A parked empty current bucket (e.g. advance() left it open with no
            // event) is filled too, so advance()+fill agrees with MultiResampleNode.
            else if (p_.fill != ResampleFill::Skip) emit_fill(cur_label_);
            // Fill internal gaps (empty buckets strictly between bucket_ and nb).
            // Trailing empties (after the last event) are handled by advance()/
            // flush(), not here.
            if (p_.fill != ResampleFill::Skip)
                for (std::int64_t b = bucket_ + 1; b < nb; ++b)
                    emit_fill(label_for(b));
            bucket_ = nb; acc_.reset(); set_index_label(nb);
        }
        acc_.add(v);
    }

    Index label_for(std::int64_t nb) const {
        std::int64_t start = p_.origin + nb * p_.width;
        return static_cast<Index>(p_.label == ResampleLabel::Left ? start : start + p_.width);
    }

    void set_index_label(std::int64_t nb) { cur_label_ = label_for(nb); }

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
        if (p_.fill == ResampleFill::Carry) {
            last_emitted_.assign(out_.begin(), out_.end());
            have_emitted_ = true;
        }
        downstream_.push(Frame<Index>{label, out_.data(), out_.size()});
    }

    // Emit a filler row for an internal empty bucket. Nan -> an all-NaN row;
    // Carry -> the previous emitted row's values verbatim (skipped if nothing
    // has been emitted yet, which cannot happen for a true internal gap).
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
    std::vector<double> out_;
    std::vector<double> last_emitted_;   // Carry: last emitted row (width out_)
    std::vector<double> nan_row_;        // Nan: reusable all-NaN row
    ResampleAccum acc_;
    bool started_ = false;
    bool have_emitted_ = false;          // Carry: any real row emitted yet
    std::int64_t bucket_ = 0;
    Index cur_label_{};
    std::int64_t count_in_bucket_ = 0;
    Index first_index_{}, last_index_{};
};

}} // namespace screamer::dag
#endif
