#ifndef SCREAMER_DETAIL_FIR_CORE_H
#define SCREAMER_DETAIL_FIR_CORE_H

// FirCore: the ring buffer and convolution shared by the library's FIR
// filters.
//
//     y[t] = sum_{k=0..L-1} taps[k] * x[t - k]
//
// taps[0] multiplies the current sample, taps[L-1] the oldest. The
// buffer holds the last L samples and is filled with zeros, so before L
// samples have arrived `convolve()` returns the zero-padded sum, which
// is the "zero" / "expanding" warmup output. Callers that want strict
// warmup check `warm()` first.
//
// `push` takes finite samples only. Under the library's `ignore` NaN
// policy the caller returns NaN at a NaN index without touching the
// filter, so a NaN never enters the buffer and never advances warmup.
//
// Used by MovingAverage (taps supplied by the caller) and FracDiff
// (taps from the fractional differencing recursion). O(L) per step.

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

namespace screamer {
namespace detail {

class FirCore {
public:
    explicit FirCore(std::vector<double> taps)
        : taps_(std::move(taps)),
          buffer_(taps_.size(), 0.0)
    {
        reset();
    }

    void reset() {
        std::fill(buffer_.begin(), buffer_.end(), 0.0);
        index_ = 0;
        n_seen_ = 0;
    }

    // Number of taps, which is also the number of samples strict warmup
    // waits for.
    std::size_t size() const { return taps_.size(); }

    const std::vector<double>& taps() const { return taps_; }

    // True once the buffer holds a full window of real samples.
    bool warm() const { return n_seen_ >= taps_.size(); }

    // The ring as it is (slot index_ is the next to write) and the count.
    template <class Ar>
    void state(Ar& ar) {
        const std::size_t L = taps_.size();
        ar.fixed("buffer", buffer_, L);
        ar("index", index_);
        ar("n_seen", n_seen_);
        ar.check(index_ < L, "index must be less than the number of taps");
        ar.check(n_seen_ <= L, "n_seen must be at most the number of taps");
        ar.check(n_seen_ == L || index_ == n_seen_, "index must equal n_seen while filling");
    }

    void push(double x) {
        buffer_[index_] = x;
        ++index_;
        if (index_ == buffer_.size()) {
            index_ = 0;
        }
        if (n_seen_ < taps_.size()) {
            ++n_seen_;
        }
    }

    double convolve() const {
        const std::size_t L = taps_.size();
        double acc = 0.0;
        // index_ is the next write slot, so the newest sample sits one
        // step behind it. Walk backwards in time alongside the taps.
        std::size_t idx = index_;
        for (std::size_t k = 0; k < L; ++k) {
            idx = (idx == 0) ? L - 1 : idx - 1;
            acc += taps_[k] * buffer_[idx];
        }
        return acc;
    }

private:
    const std::vector<double> taps_;
    std::vector<double> buffer_;
    std::size_t index_ = 0;
    std::size_t n_seen_ = 0;
};

}  // namespace detail
}  // namespace screamer

#endif
