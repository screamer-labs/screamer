#!/usr/bin/env python3
"""Generate wasm/smoke/dag_oracle.json: a Python-oracle parity fixture for the
JS (WASM) Pipeline.

Run with the abi3 build loaded:

    poetry run python devtools/wasm/gen_dag_oracle.py

Bare `python3` uses pyenv 3.11 and cannot import the cp312 abi3 module.

Each entry describes ONE representative screamer Pipeline three ways:

* ``build`` - a small declarative, JS-buildable graph description. ``dag_parity.test.ts``
  interprets it: an ordered ``nodes`` list where each node names a ``kind``
  (input / functor / combineLatest / resample / select / dropna / delay / filter)
  plus its wiring (input node ids) and params. The JS interpreter maps each kind
  to the matching ``Input`` / op-factory / combinator call and assembles a
  ``Pipeline([inputs],[outputs])``.
* ``feeds`` - the bound input data, keyed by input name: ``{values, index}``.
* ``expect`` - the Python Pipeline's output(s), one ``{values, index, width}`` per
  declared output. ``values`` is flattened row-major; ``width`` > 1 marks a wide
  (2-D) output so the JS side can reshape and compare.

The graphs cover: (a) a functor chain, (b) combine_latest -> functor,
(c) resample by_index/mean, (d) resample by_count, (e) an ohlc resample (wide),
(f) select over a wide stream, (g) dropna, and (h) a resample with a FUNCTOR
reducer (the reducer path). A delay graph is included as a bonus. The
non-default resample surface is exercised by (i) a by_cumulative (threshold)
resample and (j) a fill="carry" + label="right" resample. The `filter`
combinator is covered by (k) a mask-gated stream, and (l) a two-output pipeline
proves the multi-output run-once path returns the right values for each output.
"""

import json
import math
import os

import numpy as np

from screamer import RollingMean, Diff, Sub, ExpandingMean, Filter
from screamer.dag import Input, Pipeline
from screamer.streams import combine_latest, resample, select, dropna, delay

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "wasm", "smoke", "dag_oracle.json")


def vi(values, index):
    """A (values, index) feed pair with an explicit int64 index.

    A flat feed longer than its index carries a wide (width-N) input; reshape it
    to (rows, width) row-major, exactly as the JS side reconstructs width from
    values.length / index.length.
    """
    v = np.asarray(values, dtype=np.float64)
    idx = np.asarray(index, dtype=np.int64)
    rows = idx.shape[0]
    if rows and v.size > rows and v.size % rows == 0:
        v = v.reshape(rows, v.size // rows)
    return (v, idx)


def feed_json(values, index):
    """Serialize a feed to {values, index} with NaN -> None (JSON null)."""
    return {
        "values": [jsonify(float(v)) for v in np.asarray(values, dtype=np.float64).ravel()],
        "index": [int(i) for i in np.asarray(index).ravel()],
    }


def jsonify(x):
    """NaN -> None (JSON null); leave everything else as-is."""
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


def expect_json(out):
    """Normalize one Pipeline output (values, index) to {values, index, width}.

    values is flattened row-major; width records the column count so a wide
    (2-D) result can be reshaped and compared on the JS side.
    """
    values, index = out
    values = np.asarray(values, dtype=np.float64)
    index = np.asarray(index)
    width = 1 if values.ndim == 1 else int(values.shape[1])
    flat = values.ravel()
    return {
        "values": [jsonify(float(v)) for v in flat],
        "index": [int(i) for i in index.ravel()],
        "width": width,
    }


def run(pipeline, feeds_ordered):
    """Call the pipeline positionally and normalize to a list of outputs."""
    out = pipeline(*[vi(f["values"], f["index"]) for f in feeds_ordered])
    # A single-output Pipeline returns one (values, index) tuple; multi-output
    # returns a tuple of them. Distinguish by inspecting the first element.
    if isinstance(out, tuple) and len(out) == 2 and not isinstance(out[0], tuple):
        outs = [out]
    else:
        outs = list(out)
    return [expect_json(o) for o in outs]


# ---------------------------------------------------------------------------
# Fixed inputs (kept small).
# ---------------------------------------------------------------------------
RAMP = list(range(1, 11))                 # 1..10
RAMP_IDX = list(range(10))                # 0..9
CHAIN_VALS = list(range(8))               # 0..7
CHAIN_IDX = list(range(8))


def build_entries():
    entries = []

    # (a) functor chain: Diff(1)(RollingMean(3)(x)) -----------------------
    x = Input("x")
    y = Diff(1)(RollingMean(3)(x))
    p = Pipeline(inputs=[x], outputs=[y])
    feeds = {"x": feed_json(CHAIN_VALS, CHAIN_IDX)}
    entries.append({
        "name": "functor_chain_diff_rollingmean",
        "build": {
            "inputs": ["x"],
            "nodes": [
                {"id": "x", "kind": "input", "name": "x"},
                {"id": "rm", "kind": "functor", "op": "RollingMean", "args": [3], "inputs": ["x"]},
                {"id": "d", "kind": "functor", "op": "Diff", "args": [1], "inputs": ["rm"]},
            ],
            "outputs": ["d"],
        },
        "feeds": feeds,
        "expect": run(p, [feeds["x"]]),
    })

    # (b) combine_latest([a,b]) -> Sub ------------------------------------
    a, b = Input("a"), Input("b")
    y = Sub()(combine_latest(a, b))
    p = Pipeline(inputs=[a, b], outputs=[y])
    fa = feed_json([1.0, 2.0, 3.0, 4.0], [0, 2, 4, 6])
    fb = feed_json([10.0, 20.0, 30.0], [1, 3, 5])
    feeds = {"a": fa, "b": fb}
    entries.append({
        "name": "combine_latest_sub",
        "build": {
            "inputs": ["a", "b"],
            "nodes": [
                {"id": "a", "kind": "input", "name": "a"},
                {"id": "b", "kind": "input", "name": "b"},
                {"id": "cl", "kind": "combineLatest", "inputs": ["a", "b"], "emit": "when_all"},
                {"id": "y", "kind": "functor", "op": "Sub", "args": [], "inputs": ["cl"]},
            ],
            "outputs": ["y"],
        },
        "feeds": feeds,
        "expect": run(p, [fa, fb]),
    })

    # (c) resample by_index, mean, every=5 --------------------------------
    x = Input("x")
    y = resample(x, every=5, agg="mean")
    p = Pipeline(inputs=[x], outputs=[y])
    feeds = {"x": feed_json(RAMP, RAMP_IDX)}
    entries.append({
        "name": "resample_by_index_mean",
        "build": {
            "inputs": ["x"],
            "nodes": [
                {"id": "x", "kind": "input", "name": "x"},
                {"id": "r", "kind": "resample", "input": "x",
                 "opts": {"every": 5, "agg": "mean"}},
            ],
            "outputs": ["r"],
        },
        "feeds": feeds,
        "expect": run(p, [feeds["x"]]),
    })

    # (d) resample by_count, sum, count=3 ---------------------------------
    x = Input("x")
    y = resample(x, count=3, agg="sum")
    p = Pipeline(inputs=[x], outputs=[y])
    feeds = {"x": feed_json(RAMP, RAMP_IDX)}
    entries.append({
        "name": "resample_by_count_sum",
        "build": {
            "inputs": ["x"],
            "nodes": [
                {"id": "x", "kind": "input", "name": "x"},
                {"id": "r", "kind": "resample", "input": "x",
                 "opts": {"count": 3, "agg": "sum"}},
            ],
            "outputs": ["r"],
        },
        "feeds": feeds,
        "expect": run(p, [feeds["x"]]),
    })

    # (e) ohlc resample (wide, 4 columns) ---------------------------------
    x = Input("x")
    y = resample(x, every=5, agg="ohlc")
    p = Pipeline(inputs=[x], outputs=[y])
    feeds = {"x": feed_json(RAMP, RAMP_IDX)}
    entries.append({
        "name": "resample_ohlc",
        "build": {
            "inputs": ["x"],
            "nodes": [
                {"id": "x", "kind": "input", "name": "x"},
                {"id": "r", "kind": "resample", "input": "x",
                 "opts": {"every": 5, "agg": "ohlc"}},
            ],
            "outputs": ["r"],
        },
        "feeds": feeds,
        "expect": run(p, [feeds["x"]]),
    })

    # (f) select over a wide (ohlc) stream: open + close ------------------
    x = Input("x")
    o = resample(x, every=5, agg="ohlc")
    y = select(o, [0, 3])
    p = Pipeline(inputs=[x], outputs=[y])
    feeds = {"x": feed_json(RAMP, RAMP_IDX)}
    entries.append({
        "name": "select_from_ohlc",
        "build": {
            "inputs": ["x"],
            "nodes": [
                {"id": "x", "kind": "input", "name": "x"},
                {"id": "o", "kind": "resample", "input": "x",
                 "opts": {"every": 5, "agg": "ohlc"}},
                {"id": "s", "kind": "select", "input": "o", "columns": [0, 3]},
            ],
            "outputs": ["s"],
        },
        "feeds": feeds,
        "expect": run(p, [feeds["x"]]),
    })

    # (g) dropna over combine_latest(on_any) ------------------------------
    a, b = Input("a"), Input("b")
    y = dropna(combine_latest(a, b, emit="on_any"))
    p = Pipeline(inputs=[a, b], outputs=[y])
    fa = feed_json([1.0, float("nan"), 3.0], [0, 1, 2])
    fb = feed_json([10.0, 20.0, float("nan")], [0, 1, 2])
    feeds = {"a": fa, "b": fb}
    entries.append({
        "name": "dropna_combine_latest",
        "build": {
            "inputs": ["a", "b"],
            "nodes": [
                {"id": "a", "kind": "input", "name": "a"},
                {"id": "b", "kind": "input", "name": "b"},
                {"id": "cl", "kind": "combineLatest", "inputs": ["a", "b"], "emit": "on_any"},
                {"id": "dn", "kind": "dropna", "inputs": ["cl"], "how": "any"},
            ],
            "outputs": ["dn"],
        },
        "feeds": feeds,
        "expect": run(p, [fa, fb]),
    })

    # (h) resample with a FUNCTOR reducer (ExpandingMean) -----------------
    x = Input("x")
    y = resample(x, every=5, agg=ExpandingMean())
    p = Pipeline(inputs=[x], outputs=[y])
    feeds = {"x": feed_json(RAMP, RAMP_IDX)}
    entries.append({
        "name": "resample_functor_reducer",
        "build": {
            "inputs": ["x"],
            "nodes": [
                {"id": "x", "kind": "input", "name": "x"},
                {"id": "r", "kind": "resample", "input": "x",
                 "opts": {"every": 5},
                 "reducer": {"op": "ExpandingMean", "args": []}},
            ],
            "outputs": ["r"],
        },
        "feeds": feeds,
        "expect": run(p, [feeds["x"]]),
    })

    # (bonus) delay a functor stream --------------------------------------
    x = Input("x")
    y = delay(RollingMean(2)(x), duration=2)
    p = Pipeline(inputs=[x], outputs=[y])
    feeds = {"x": feed_json(RAMP, RAMP_IDX)}
    entries.append({
        "name": "delay_rollingmean",
        "build": {
            "inputs": ["x"],
            "nodes": [
                {"id": "x", "kind": "input", "name": "x"},
                {"id": "rm", "kind": "functor", "op": "RollingMean", "args": [2], "inputs": ["x"]},
                {"id": "d", "kind": "delay", "input": "rm", "k": 2},
            ],
            "outputs": ["d"],
        },
        "feeds": feeds,
        "expect": run(p, [feeds["x"]]),
    })

    # (i) resample by_cumulative (threshold): single width-2 [value, driver] ---
    # The ResampleNode sees width-2 frames and closes a bar when the cumulative
    # driver reaches the threshold. Exercises the by_cumulative opts->code path.
    x = Input("x")
    y = resample(x, threshold=5.0, agg="sum")
    p = Pipeline(inputs=[x], outputs=[y])
    thr_vals = [1.0, 2.0, 2.0, 2.0, 3.0, 2.0, 1.0, 10.0]   # [value, driver] rows
    thr_drv = [2.0, 2.0, 2.0, 3.0, 1.0, 2.0, 2.0, 10.0]
    thr_2d = np.column_stack([thr_vals, thr_drv])
    thr_idx = list(range(8))
    feeds = {"x": feed_json(thr_2d, thr_idx)}
    entries.append({
        "name": "resample_by_cumulative_sum",
        "build": {
            "inputs": ["x"],
            "nodes": [
                {"id": "x", "kind": "input", "name": "x"},
                {"id": "r", "kind": "resample", "input": "x",
                 "opts": {"threshold": 5.0, "agg": "sum"}},
            ],
            "outputs": ["r"],
        },
        "feeds": feeds,
        "expect": run(p, [feeds["x"]]),
    })

    # (j) resample fill="carry" + label="right": empty buckets carry forward ---
    # Sparse index leaves interior buckets empty; carry repeats the prior bar and
    # label="right" stamps the bucket's right edge. Exercises the non-default
    # fill and label opts->code translation together.
    x = Input("x")
    y = resample(x, every=3, agg="last", fill="carry", label="right")
    p = Pipeline(inputs=[x], outputs=[y])
    carry_vals = [1.0, 2.0, 3.0, 10.0, 11.0]
    carry_idx = [0, 1, 2, 9, 10]          # buckets [3,6) and [6,9) are empty
    feeds = {"x": feed_json(carry_vals, carry_idx)}
    entries.append({
        "name": "resample_carry_right",
        "build": {
            "inputs": ["x"],
            "nodes": [
                {"id": "x", "kind": "input", "name": "x"},
                {"id": "r", "kind": "resample", "input": "x",
                 "opts": {"every": 3, "agg": "last", "fill": "carry", "label": "right"}},
            ],
            "outputs": ["r"],
        },
        "feeds": feeds,
        "expect": run(p, [feeds["x"]]),
    })

    # (k) filter: gate a data stream by a mask stream --------------------------
    # Mask nonzero keeps the aligned value; zero or NaN drops it. Wrong arg order
    # (data/mask swapped) or wrong gate semantics would diverge here.
    d, m = Input("d"), Input("m")
    y = Filter()(d, m)
    p = Pipeline(inputs=[d, m], outputs=[y])
    fd = feed_json([10.0, 20.0, 30.0, 40.0, 50.0], [0, 1, 2, 3, 4])
    fm = feed_json([1.0, 0.0, float("nan"), 2.0, 1.0], [0, 1, 2, 3, 4])
    feeds = {"d": fd, "m": fm}
    entries.append({
        "name": "filter_mask_gate",
        "build": {
            "inputs": ["d", "m"],
            "nodes": [
                {"id": "d", "kind": "input", "name": "d"},
                {"id": "m", "kind": "input", "name": "m"},
                {"id": "f", "kind": "filter", "data": "d", "mask": "m"},
            ],
            "outputs": ["f"],
        },
        "feeds": feeds,
        "expect": run(p, [fd, fm]),
    })

    # (l) two-output pipeline: multi-output run-once correctness ---------------
    # One input feeds two independent branches. The batch path runs the graph
    # ONCE and reads each cached output; both must match Python exactly.
    x = Input("x")
    out_mean = RollingMean(3)(x)
    out_diff = Diff(1)(x)
    p = Pipeline(inputs=[x], outputs=[out_mean, out_diff])
    feeds = {"x": feed_json(RAMP, RAMP_IDX)}
    entries.append({
        "name": "multi_output_mean_diff",
        "build": {
            "inputs": ["x"],
            "nodes": [
                {"id": "x", "kind": "input", "name": "x"},
                {"id": "rm", "kind": "functor", "op": "RollingMean", "args": [3], "inputs": ["x"]},
                {"id": "d", "kind": "functor", "op": "Diff", "args": [1], "inputs": ["x"]},
            ],
            "outputs": ["rm", "d"],
        },
        "feeds": feeds,
        "expect": run(p, [feeds["x"]]),
    })

    return entries


def main():
    entries = build_entries()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(entries, f, indent=2, allow_nan=False)
        f.write("\n")
    for e in entries:
        outs = e["expect"]
        shape = ", ".join(f"{len(o['values'])}x{o['width']}" for o in outs)
        print(f"  {e['name']:32s} outputs=[{shape}]")
    print(f"\nWrote {len(entries)} DAG oracle entries -> {OUT}")


if __name__ == "__main__":
    main()
