"""Streaming benchmark: cost per event, against window size.

Run:

    python benchmarks/run_streaming_benchmarks.py            # all functions
    python benchmarks/run_streaming_benchmarks.py --func RollingMax

Writes one CSV per (func, lib) into benchmarks/experiments_streaming/ with the
same shape the batch runner uses, plus a summary table on stdout.

Read the result as cost per event in nanoseconds. The claim being tested is
that screamer's is flat in `window_size` while a per-event reduction over a
window grows linearly with it, so the ratio widens as the window grows. See
streaming_impls.py for why the alternatives are written the way they are.

Timing note: every variant loops in Python over the events, so all of them pay
the interpreter's per-iteration cost. That is deliberate. It is the cost a
Python user actually sees, and it is charged equally to every variant, so the
comparison is not flattered by hiding it. The screamer column therefore
includes its pybind11 call overhead per event and is *not* the engine's speed;
for that, see the batch suite or run through a Pipeline, where events stay in
C++.
"""
import argparse
import os
import timeit

import numpy as np
import pandas as pd

import streaming_impls

WINDOW_SIZES = [10, 50, 100, 500, 1000, 5000]
N_EVENTS = 20_000


def time_one(callable_name, values, window_size, repeat):
    fn = getattr(streaming_impls, callable_name)
    fn(values, window_size)          # warm up: numba compiles on first call
    best = min(
        timeit.timeit(lambda: fn(values, window_size), number=1) for _ in range(repeat)
    )
    return best / len(values) * 1e9        # nanoseconds per event


def main():
    parser = argparse.ArgumentParser(description="Run the streaming benchmarks.")
    parser.add_argument("--func", type=str, default=None, help="Function name filter")
    parser.add_argument("--lib", type=str, default=None, help="Library name filter")
    parser.add_argument("--repeat", type=int, default=5, help="Repeats per point")
    parser.add_argument("--events", type=int, default=N_EVENTS, help="Events per run")
    args = parser.parse_args()

    table = streaming_impls.all()
    if args.func:
        table = table[table["func"] == args.func]
    if args.lib:
        table = table[table["lib"] == args.lib]

    rng = np.random.default_rng(0)
    values = rng.normal(size=args.events)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "experiments_streaming")
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for _, row in table.iterrows():
        print(f"{row['callable']} ", end="", flush=True)
        rows = []
        for window_size in WINDOW_SIZES:
            print(".", end="", flush=True)
            ns = time_one(row["callable"], values, window_size, args.repeat)
            rows.append({"func": row["func"], "lib": row["lib"],
                         "window_size": window_size, "ns_per_event": ns})
        print()
        frame = pd.DataFrame(rows)
        frame.to_csv(os.path.join(out_dir, f"sbm__{row['func']}__{row['lib']}.csv"),
                     index=False)
        results.append(frame)

    if not results:
        return

    everything = pd.concat(results)
    print("\nns per event\n")
    pivot = everything.pivot_table(index=["func", "lib"], columns="window_size",
                                   values="ns_per_event")
    print(pivot.round(1).to_string())

    print("\nscaling: cost at the largest window divided by cost at the smallest\n")
    scaling = (pivot[WINDOW_SIZES[-1]] / pivot[WINDOW_SIZES[0]]).round(2)
    print(scaling.to_string())
    print("\nA flat cost is ~1.0. A per-event reduction over the window grows "
          "with it.")


if __name__ == "__main__":
    main()
