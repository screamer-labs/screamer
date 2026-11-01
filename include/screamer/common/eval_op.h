#ifndef SCREAMER_EVAL_OP_H
#define SCREAMER_EVAL_OP_H

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include "screamer/serial/errors.h"
#include "screamer/serial/panel.h"

// An error path kept out of line and marked cold, so the check that calls it
// adds only a test and a branch to a hot path (functor_base.h and
// dag/checked_index.h define the same SCREAMER_NOINLINE).
#ifndef SCREAMER_NOINLINE
#if defined(__GNUC__) || defined(__clang__)
#define SCREAMER_NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#define SCREAMER_NOINLINE __declspec(noinline)
#else
#define SCREAMER_NOINLINE
#endif
#endif
#if defined(__GNUC__) || defined(__clang__)
#define SCREAMER_COLD __attribute__((cold))
#define SCREAMER_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define SCREAMER_COLD
#define SCREAMER_UNLIKELY(x) (x)
#endif

namespace screamer {

// The uniform op interface the DAG engine holds for ANY functor, regardless of
// arity. ScreamerBase (1->1) and FunctorBase<_,N,M> (N->M) both implement it in
// terms of their existing process_scalar/call. eval() processes exactly one
// event: it reads n_in() inputs from `in` and writes n_out() outputs to `out`.
//
// eval_column() is the batch entry point of both bindings: one call runs a
// whole column of a broadcast batch, so the per-event loop runs inside the
// operator, where the compiler sees the concrete type.
struct EvalOp {
    EvalOp() = default;
    // A copy never carries the per-column state (panel_): it starts with none,
    // and an assignment drops the target's. A move takes it along.
    EvalOp(const EvalOp&) {}
    EvalOp& operator=(const EvalOp&) {
        panel_.reset();
        return *this;
    }
    EvalOp(EvalOp&&) = default;
    EvalOp& operator=(EvalOp&&) = default;
    virtual ~EvalOp();

    virtual std::size_t n_in() const = 0;
    virtual std::size_t n_out() const = 0;
    virtual void eval(const double* in, double* out) = 0;
    virtual void reset() = 0;

    // Run one column of T steps. in[i] points at input i's first element of this
    // column and advances by in_stride[i] per step; out advances by out_stride
    // per step and holds n_out values per step. Resets the op first.
    virtual void eval_column(const double* const* in, const std::ptrdiff_t* in_stride,
                             std::size_t T, double* out, std::ptrdiff_t out_stride) = 0;

    // True when eval_column inlines the per-step call into its loop, so the
    // loop is tight enough that the column driver tiles long strided columns
    // (run_columns' `tight`). False for a per-step call kept out of line.
    virtual bool tight_column() const { return false; }

    // eval_column without the reset: the column continues from the current
    // state and leaves the end state, the batch call `resume`. It runs the
    // per-step loop, never a whole-column fast path (those may assume a
    // reset start). The default calls eval() per step.
    virtual void eval_column_resume(const double* const* in, const std::ptrdiff_t* in_stride,
                                    std::size_t T, double* out, std::ptrdiff_t out_stride);

    // A copy of the operator in its reset state, one column of a panel; null
    // when the operator cannot be copied.
    virtual std::unique_ptr<EvalOp> clone_fresh() const { return nullptr; }

    // The per-column state, made by the first `resume` on a (T, K) array
    // (K >= 1): panel_->cols[c] runs column c, and the operator's own state
    // stays reset. A default batch call and reset_all() drop it; a call with
    // one event raises while it is held (check_event_call). A resume that
    // throws part way leaves the columns it ran advanced.
    std::unique_ptr<serial::Panel> panel_;

    bool has_panel() const { return panel_ != nullptr; }
    std::size_t panel_columns() const { return panel_ ? panel_->cols.size() : 0; }
    void drop_panel() { panel_.reset(); }
    // The reset of the bindings: drops the panel, then resets the operator.
    void reset_all() {
        panel_.reset();
        reset();
    }
    // The error of a call with one event while a panel is held.
    void check_event_call() const {
        if (SCREAMER_UNLIKELY(panel_ != nullptr)) throw_panel_event_call();
    }

    // Save and load. serial::Saveable (serial/saveable.h) implements these for
    // an operator that describes its settings and state; any other operator
    // is not saveable, and its write and read calls raise NotSaveable.
    //
    // type_name() is the operator's registry name ("RollingMean"), or null.
    // write_settings writes the constructor arguments as an object, by their
    // documented names. write_state writes {"version": N, "values": {...}},
    // and read_state loads that tree, or raises a SaveError naming `path` and
    // leaves the operator unchanged.
    virtual bool saveable() const { return false; }
    virtual const char* type_name() const { return nullptr; }
    virtual int state_version() const { return 0; }
    virtual void write_settings(json::Value&) const { not_saveable(""); }
    virtual void write_state(json::Value&) const { not_saveable(""); }
    virtual void read_state(const json::Value&, const std::string& path) { not_saveable(path); }

private:
    [[noreturn]] SCREAMER_NOINLINE SCREAMER_COLD void throw_panel_event_call() const;
    [[noreturn]] static void not_saveable(const std::string& path) {
        throw serial::NotSaveable(path.empty() ? "operator" : path,
                                  "the operator has no save and load support yet");
    }
};

// Out of the class, where EvalOp is complete for Panel's destructor.
inline EvalOp::~EvalOp() = default;

inline void EvalOp::throw_panel_event_call() const {
    const std::string k = std::to_string(panel_->cols.size());
    throw std::invalid_argument("the object holds per-column state for " + k +
                                " columns; call resume with a (T, " + k +
                                ") array, or reset() first");
}

inline void EvalOp::eval_column_resume(const double* const* in, const std::ptrdiff_t* in_stride,
                                       std::size_t T, double* out, std::ptrdiff_t out_stride) {
    const std::size_t n = n_in();
    std::unique_ptr<double[]> row(new double[n ? n : 1]);
    std::unique_ptr<const double*[]> p(new const double*[n ? n : 1]);
    for (std::size_t i = 0; i < n; ++i) p[i] = in[i];
    for (std::size_t t = 0; t < T; ++t, out += out_stride) {
        for (std::size_t i = 0; i < n; ++i) {
            row[i] = *p[i];
            p[i] += in_stride[i];
        }
        eval(row.get(), out);
    }
}

} // namespace screamer
#endif
