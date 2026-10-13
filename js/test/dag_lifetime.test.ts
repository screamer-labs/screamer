import { test } from "node:test";
import assert from "node:assert/strict";
import { ready, Input, Pipeline, Diff, RollingMean, RollingSum, ExpandingMean, Add, resample } from "../src/index.ts";
import type { Output } from "../src/index.ts";

// Op-lifetime safety: the Embind DAG layer holds only a non-owning `EvalOp*`
// per functor node (see js/src/pipeline.ts `compile()` and bindings_dag.cpp).
// The JS Pipeline must (1) reject sharing one stateful op instance across two
// nodes, (2) pin each node's op wrapper for the graph's lifetime so it
// survives GC once the caller drops its own reference, (3) free the C++
// graph exactly once on dispose() without disturbing the (separately owned)
// op wrappers, and (4) fail clearly - not silently - when a declared input
// is never fed at call time.

function assertNanEq(got: ArrayLike<number>, expected: ArrayLike<number>, label: string, tol = 1e-9) {
  assert.equal(got.length, expected.length, `${label}: length ${got.length} != ${expected.length}`);
  for (let i = 0; i < expected.length; i++) {
    const e = expected[i], g = got[i];
    if (Number.isNaN(e)) assert.ok(Number.isNaN(g), `${label}[${i}]: expected NaN, got ${g}`);
    else assert.ok(Math.abs(g - e) <= tol, `${label}[${i}]: ${g} != ${e}`);
  }
}

// --- 1. One-instance-per-node throws -------------------------------------
//
// A single functor instance (`m`) is applied to two different Input nodes,
// so it would have to back two graph nodes at once and share its C++ state
// between them. `wrapOp`'s node branch lets `m(x)` and `m(other)` each
// return a fresh Node, but both Nodes carry the SAME underlying `raw` op
// (`{functor: raw, wrapper: op}`), so the reuse is structurally reachable
// and must be rejected at graph-build time by `checkStatefulSafety`
// (mirrors Python's `_check_stateful_safety`).
test("lifetime: reusing one op instance across two nodes throws", async () => {
  await ready();
  const x = Input("x");
  const other = Input("other");
  const m = RollingMean(3);
  const y = Diff(1)(m(x));
  const z = m(other); // same `m` backing a second node

  assert.throws(
    () => new Pipeline([x, other], [y, z]),
    /same functor instance backs two nodes/,
  );
});

// A control case: two DISTINCT instances of the same op kind on different
// nodes must be perfectly fine (the rule is about instance identity, not op
// kind).
test("lifetime: two distinct instances of the same op kind are fine", async () => {
  await ready();
  const x = Input("x");
  const other = Input("other");
  const y = Diff(1)(RollingMean(3)(x));
  const z = RollingMean(3)(other);
  const p = new Pipeline([x, other], [y, z]);
  const out = p({ x: new Float64Array([1, 2, 3, 4]), other: new Float64Array([5, 6, 7]) }) as Output[];
  assert.equal(out.length, 2);
  p.dispose();
});

// --- 2. Retained op survives ----------------------------------------------
//
// `compile()` pushes each functor node's op wrapper into `this.ops` to keep
// the backing C++ op alive for as long as the Pipeline lives, because the
// Embind graph itself only stores a non-owning `EvalOp*`. Build the graph
// inside a function so every local reference to the Input/op/intermediate
// Nodes goes out of scope on return, force a GC pass if the runtime exposes
// one, then run the pipeline - it must still produce correct output, which
// only holds if the Pipeline's retention (not the caller's variables) is
// what kept the op alive.
test("lifetime: a Pipeline keeps its ops alive after the caller drops all references", async () => {
  await ready();

  function build() {
    const x = Input("x");
    const m = RollingMean(3);
    const y = Diff(1)(m(x));
    return new Pipeline([x], [y]);
  }

  const p = build(); // only the Pipeline itself remains reachable

  if (typeof globalThis.gc === "function") {
    globalThis.gc();
  }

  const data = new Float64Array([1, 2, 3, 4, 5, 6]);
  const { values } = p(data) as Output;

  const rm = RollingMean(3)(data) as Float64Array;
  const expected = Diff(1)(rm) as Float64Array;
  assertNanEq(values as Float64Array, expected, "retained-op output");

  p.dispose();
});

// --- 3. Dispose safety -----------------------------------------------------

test("lifetime: dispose() frees the graph, is idempotent, and use-after-dispose throws cleanly", async () => {
  await ready();
  const x = Input("x");
  const y = RollingMean(3)(x);
  const p = new Pipeline([x], [y]);

  const before = p(new Float64Array([1, 2, 3, 4])) as Output;
  assert.equal(before.values.length, 4);

  p.dispose();
  // Idempotent: a second dispose() must not throw (no double-free of the
  // CompiledGraph, and dropping `this.ops` again is a no-op).
  assert.doesNotThrow(() => p.dispose());

  // Use after dispose must throw a clear JS error, not crash / return
  // garbage. Both the batch call path and live() must be guarded.
  assert.throws(() => p(new Float64Array([1, 2, 3])), /used after dispose/);
  assert.throws(() => p.live(), /used after dispose/);
});

// --- 4. Unfed input --------------------------------------------------------

test("lifetime: a declared-and-used input left unfed at call time throws", async () => {
  await ready();
  const x = Input("x");
  const y = Input("y");
  const out = Add()(x, y);
  const p = new Pipeline([x, y], [out]);

  assert.throws(
    () => p({ x: new Float64Array([1, 2, 3]) }), // 'y' never fed
    /missing feed for input 'y'/,
  );

  p.dispose();
});

// --- Each Pipeline owns its operators' state (H4) --------------------------
// The twin of tests/test_pipeline_owns_state.py.

const vals = (o: Output) => Array.from(o.values as Float64Array);

test("lifetime: two Pipelines from the same nodes have their own state", async () => {
  await ready();
  const x = Input("x");
  const y = RollingSum(2)(x);
  const p1 = new Pipeline([x], [y]), p2 = new Pipeline([x], [y]);
  try {
    const s1 = p1.live(), s2 = p2.live();
    for (let t = 0; t < 4; t++) { s1.push("x", t, 1); s2.push("x", t, 100); }
    s1.flush(); s2.flush();
    assertNanEq(vals(s1.result() as Output), [NaN, 2, 2, 2], "p1");
    assertNanEq(vals(s2.result() as Output), [NaN, 200, 200, 200], "p2");
    // A batch call of p1 does not reset a session of p2.
    const t2 = p2.live();
    t2.push("x", 0, 1); t2.push("x", 1, 1);
    p1(new Float64Array([5, 5, 5]));
    t2.push("x", 2, 1); t2.flush();
    assertNanEq(vals(t2.result() as Output), [NaN, 2, 2], "p2 after p1 batch");
  } finally { p1.dispose(); p2.dispose(); }
});

test("lifetime: disposing or calling the original op does not affect the Pipeline", async () => {
  await ready();
  const op = RollingSum(2);
  const x = Input("x");
  const p = new Pipeline([x], [op(x)]);
  try {
    const s = p.live();
    s.push("x", 0, 1);
    assert.ok(Number.isNaN(op(50) as number));       // the op's own state
    s.push("x", 1, 1); s.flush();
    assertNanEq(vals(s.result() as Output), [NaN, 2], "session");
    assert.equal(op(50), 100);                       // unaffected by the session
    op.dispose();                                    // the Pipeline keeps its copy
    assertNanEq(vals(p(new Float64Array([1, 2, 3])) as Output), [NaN, 3, 5], "after dispose");
  } finally { p.dispose(); }
});

test("lifetime: a functor reducer is owned by each Pipeline", async () => {
  await ready();
  const x = Input("x");
  const agg = ExpandingMean();
  const node = resample(x, { every: 2, reducer: agg(Input("_")) as any });
  const p1 = new Pipeline([x], [node]), p2 = new Pipeline([x], [node]);
  try {
    const s1 = p1.live(), s2 = p2.live();
    for (let t = 0; t < 4; t++) { s1.push("x", t, 1); s2.push("x", t, 10); }
    s1.flush(); s2.flush();
    assertNanEq(vals(s1.result() as Output), [1, 1], "p1");
    assertNanEq(vals(s2.result() as Output), [10, 10], "p2");
    agg.dispose();
    assertNanEq(vals(p1(new Float64Array([1, 1])) as Output), [1], "after dispose");
  } finally { p1.dispose(); p2.dispose(); }
});

// Fix round 1, N3: every live session runs on an engine of its own.
test("lifetime: two sessions of one Pipeline do not mix; a batch call does not end a session", async () => {
  await ready();
  const x = Input("x");
  const p = new Pipeline([x], [RollingSum(2)(x)]);
  try {
    const s3 = p.live(), s4 = p.live();
    for (let t = 0; t < 3; t++) { s3.push("x", t, 1); s4.push("x", t, 100); }
    p(new Float64Array([7, 7, 7]));                  // a batch call between pushes
    s3.push("x", 3, 1);
    s3.flush(); s4.flush();
    assertNanEq(vals(s3.result() as Output), [NaN, 2, 2, 2], "s3");
    assertNanEq(vals(s4.result() as Output), [NaN, 200, 200], "s4");
    s3.dispose();
    assert.throws(() => s3.push("x", 4, 1), /live session used after dispose/);
    s3.dispose();                                    // idempotent
    const s5 = p.live();
    p.dispose();                                     // frees the open sessions too
    assert.throws(() => s5.push("x", 0, 1), /pipeline used after dispose/);
  } finally { p.dispose(); }
});
