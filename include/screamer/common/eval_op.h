#ifndef SCREAMER_EVAL_OP_H
#define SCREAMER_EVAL_OP_H

#include <cstddef>
#include <string>
#include "screamer/serial/errors.h"

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
    virtual ~EvalOp() = default;
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
    [[noreturn]] static void not_saveable(const std::string& path) {
        throw serial::NotSaveable(path.empty() ? "operator" : path,
                                  "the operator has no save and load support yet");
    }
};

} // namespace screamer
#endif
