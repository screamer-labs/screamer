import { test } from "node:test";
import assert from "node:assert/strict";
import {
  ready, Input, Pipeline, select, resample, combineLatest, filter, delay,
  RollingMean, IsNan, type Output,
} from "../src/index.ts";

// Pipeline alignOutputs (B7): the outputs of a multi-output Pipeline on one
// index, as Python's align_outputs. Mirrors the A11 tests of
// tests/test_pipeline_node_widths.py and tests/test_pipeline_align_outputs.py.

const nd = (rows: number[][]) => ({ data: Float64Array.from(rows.flat()), shape: [rows.length, rows[0].length] });
const arr = (x: any): number[] => Array.from((x?.data ?? x) as Float64Array);
const shape = (x: any): number[] => x?.shape ?? [x.length];
const same = (a: number[], b: number[]) =>
  a.length === b.length && a.every((v, i) => Object.is(v, b[i]));

// Assert two results are equal: values (NaN equal), shapes and indexes.
function assertSame(got: Output | Output[], want: Output | Output[], what: string): void {
  const g = Array.isArray(got) ? got : [got];
  const w = Array.isArray(want) ? want : [want];
  assert.equal(g.length, w.length, `${what}: output count`);
  g.forEach((o, j) => {
    assert.ok(same(arr(o.values), arr(w[j].values)), `${what}: values of output ${j}: ${arr(o.values)} != ${arr(w[j].values)}`);
    assert.deepEqual(shape(o.values), shape(w[j].values), `${what}: shape of output ${j}`);
    assert.deepEqual(Array.from(o.index), Array.from(w[j].index), `${what}: index of output ${j}`);
  });
}

type Ev = [string, number, number | number[]];

// Push the events into a new session, flush, and return result().
function live(p: Pipeline, events: Ev[]): Output | Output[] {
  const s = p.live();
  for (const [name, i, v] of events) s.push(name, i, v);
  s.flush();
  return s.result();
}

const X3 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];

test("alignOutputs: the default is true, as in Python", async () => {
  await ready();
  const a = Input("a");
  const p = new Pipeline([a], [a, select(a, [0])]);
  const q = new Pipeline([a], [a, select(a, [0])], { alignOutputs: false });
  try {
    assert.equal(p.alignOutputs, true);
    assert.equal(q.alignOutputs, false);
  } finally {
    p.dispose();
    q.dispose();
  }
});

test("alignOutputs: mixed widths keep their own columns, batch and live", async () => {
  await ready();
  const a = Input("a");
  const p = new Pipeline([a], [a, select(a, [1]), select(a, [2, 0]), RollingMean(2)(select(a, [0]) as any) as any]);
  try {
    const out = p(nd(X3) as any) as Output[];
    assert.deepEqual(arr(out[0].values), X3.flat());
    assert.deepEqual(shape(out[0].values), [3, 3]);
    assert.deepEqual(arr(out[1].values), [2, 5, 8]);
    assert.deepEqual(arr(out[2].values), [3, 1, 6, 4, 9, 7]);
    assert.deepEqual(shape(out[2].values), [3, 2]);
    assert.ok(same(arr(out[3].values), [NaN, 2.5, 5.5]));
    for (const o of out) assert.deepEqual(Array.from(o.index), [0, 1, 2]);
    // The aligned outputs share one index array.
    for (const o of out) assert.equal(o.index, out[0].index);
    assertSame(live(p, X3.map((r, k) => ["a", k, r] as Ev)), out, "live");
  } finally {
    p.dispose();
  }
});

test("alignOutputs: the reported case", async () => {
  await ready();
  const a = Input("a");
  const p = new Pipeline([a], [a, select(a, [0])]);
  try {
    const [o0, o1] = p(nd([[1, 2], [3, 4]]) as any) as Output[];
    assert.deepEqual(arr(o0.values), [1, 2, 3, 4]);
    assert.deepEqual(arr(o1.values), [1, 3]);
    assert.deepEqual(Array.from(o0.index), [0, 1]);
    assert.deepEqual(Array.from(o1.index), [0, 1]);
  } finally {
    p.dispose();
  }
});

test("alignOutputs: outputs on different indexes align as of, batch and live", async () => {
  await ready();
  // a is 2 columns at 0, 2, 4; b is 1 column at 1, 2, 5; c is 3 columns at 3.
  const a = Input("a"), b = Input("b"), c = Input("c");
  const p = new Pipeline([a, b, c], [a, b, c]);
  try {
    const feeds = {
      a: { values: nd([[1, 2], [3, 4], [5, 6]]), index: [0, 2, 4] },
      b: { values: [10, 20, 30], index: [1, 2, 5] },
      c: { values: nd([[7, 8, 9]]), index: [3] },
    };
    // Every output has a value from index 3 on; one row per index 3, 4, 5.
    const k = Float64Array.from([3, 4, 5]);
    const want: Output[] = [
      { values: nd([[3, 4], [5, 6], [5, 6]]) as any, index: k },
      { values: Float64Array.from([20, 20, 30]), index: k },
      { values: nd([[7, 8, 9], [7, 8, 9], [7, 8, 9]]) as any, index: k },
    ];
    const got = p(feeds as any) as Output[];
    assertSame(got, want, "batch");
    const events: Ev[] = [
      ["a", 0, [1, 2]], ["b", 1, 10], ["a", 2, [3, 4]], ["b", 2, 20],
      ["c", 3, [7, 8, 9]], ["a", 4, [5, 6]], ["b", 5, 30],
    ];
    assertSame(live(p, events), want, "live");
  } finally {
    p.dispose();
  }
});

test("alignOutputs: false keeps independent outputs, batch and live", async () => {
  await ready();
  const a = Input("a"), b = Input("b");
  const p = new Pipeline([a, b], [a, b], { alignOutputs: false });
  try {
    const got = p({ a: { values: [1, 2, 3], index: [0, 2, 4] }, b: { values: [10, 20], index: [1, 3] } } as any) as Output[];
    assert.deepEqual(arr(got[0].values), [1, 2, 3]);
    assert.deepEqual(Array.from(got[0].index), [0, 2, 4]);
    assert.deepEqual(arr(got[1].values), [10, 20]);
    assert.deepEqual(Array.from(got[1].index), [1, 3]);
    const l = live(p, [["a", 0, 1], ["b", 1, 10], ["a", 2, 2], ["b", 3, 20], ["a", 4, 3]]);
    assertSame(l, got, "live");
  } finally {
    p.dispose();
  }
});

test("alignOutputs: one output is one Output, whatever the option", async () => {
  await ready();
  const a = Input("a");
  for (const alignOutputs of [true, false]) {
    const p = new Pipeline([a], [RollingMean(2)(a) as any], { alignOutputs });
    try {
      const out = p([1, 2, 3]) as Output;
      assert.ok(!Array.isArray(out));
      assert.ok(same(arr(out.values), [NaN, 1.5, 2.5]));
      assertSame(live(p, [["a", 0, 1], ["a", 1, 2], ["a", 2, 3]]), out, "live");
    } finally {
      p.dispose();
    }
  }
});

test("alignOutputs: batch equals live for resample, filter and duplicate indexes", async () => {
  await ready();
  const x = Input("x"), m = Input("m");
  const n = 60;
  const xs = Array.from({ length: n }, (_, i) => Math.sin(i * 0.7) * 3 + (i % 5 === 0 ? NaN : 0));
  const xi = Array.from({ length: n }, (_, i) => Math.floor(i / 2));   // each index twice
  const ms = Array.from({ length: n }, (_, i) => (i % 3 === 0 ? 1 : 0));
  for (const alignOutputs of [true, false]) {
    const p = new Pipeline([x, m], [
      x, resample(x, { every: 5, agg: "ohlc" }), filter(x, m), RollingMean(3)(x) as any,
    ], { alignOutputs });
    try {
      const batch = p({ x: { values: xs, index: xi }, m: { values: ms, index: xi } } as any) as Output[];
      const events: Ev[] = [];
      for (let i = 0; i < n; i++) { events.push(["x", xi[i], xs[i]]); events.push(["m", xi[i], ms[i]]); }
      assertSame(live(p, events), batch, `alignOutputs ${alignOutputs}`);
      if (alignOutputs) for (const o of batch) assert.equal(o.index, batch[0].index);
    } finally {
      p.dispose();
    }
  }
});

test("alignOutputs: aligned rows equal a combineLatest of the outputs", async () => {
  await ready();
  // The output alignment is the as-of join of combineLatest (emit when_all),
  // one row per distinct index, split back into the outputs.
  const x = Input("x"), y = Input("y");
  const outs = () => [RollingMean(2)(x) as any, resample(y, { every: 3, agg: "sum" })];
  const p = new Pipeline([x, y], outs());
  const q = new Pipeline([x, y], [combineLatest(outs())]);
  try {
    const feeds = {
      x: { values: [1, 2, 3, 4, 5, 6, 7], index: [0, 1, 1, 3, 6, 7, 9] },
      y: { values: [10, 20, 30, 40, 50], index: [0, 2, 4, 5, 8] },
    };
    const [a, b] = p(feeds as any) as Output[];
    const j = q(feeds as any) as Output;
    const rows = arr(j.values);
    assert.deepEqual(arr(a.values), rows.filter((_, i) => i % 2 === 0));
    assert.deepEqual(arr(b.values), rows.filter((_, i) => i % 2 === 1));
    assert.deepEqual(Array.from(a.index), Array.from(j.index));
  } finally {
    p.dispose();
    q.dispose();
  }
});

test("alignOutputs: an empty output gives no rows, with each output's width", async () => {
  await ready();
  // As in Python: a is empty, so no row is complete; select(b, [0, 2]) keeps
  // its 2 columns, a one-column output is a Float64Array.
  const a = Input("a"), b = Input("b");
  const p = new Pipeline([a, b], [a, select(b, [0, 2])]);
  try {
    const out = p({ a: { values: [], index: [] }, b: nd([[1, 2, 3], [4, 5, 6]]) } as any) as Output[];
    assert.equal(out[0].index.length, 0);
    assert.ok(out[0].values instanceof Float64Array);
    assert.equal((out[0].values as Float64Array).length, 0);
    assert.deepEqual(shape(out[1].values), [0, 2]);
    assert.equal(arr(out[1].values).length, 0);
    assert.equal(out[1].index, out[0].index);
    assertSame(live(p, [["b", 0, [1, 2, 3]], ["b", 1, [4, 5, 6]]]), out, "live");
  } finally {
    p.dispose();
  }
});

test("Pipeline options: a wrong kind or an unknown key is a TypeError", async () => {
  await ready();
  const a = Input("a");
  const isType = (re: RegExp) => (e: unknown) => e instanceof TypeError && re.test((e as Error).message);
  assert.throws(() => new Pipeline([a], [a], { alignOutputs: 1 } as any), isType(/alignOutputs must be a boolean, got a number/));
  assert.throws(() => new Pipeline([a], [a], { align_outputs: false } as any), isType(/unknown option 'align_outputs'/));
  assert.throws(() => new Pipeline([a], [a], true as any), isType(/the options are/));
  assert.throws(() => new Pipeline([a], [a], new Map() as any), isType(/the options are/));
});

// Concatenate the results of repeated live drains into one result per output.
function concat(parts: Output[][]): Output[] {
  const n = parts[0].length;
  return Array.from({ length: n }, (_, j) => {
    const vals: number[] = [], idx: number[] = [];
    let width = 1;
    for (const p of parts) {
      const v = p[j].values as any;
      if (v.shape) width = v.shape[1];
      vals.push(...arr(v));
      idx.push(...Array.from(p[j].index));
    }
    const rows = idx.length;
    const values = width === 1 && !(parts[0][j].values as any).shape
      ? Float64Array.from(vals) : ({ data: Float64Array.from(vals), shape: [rows, width] } as any);
    return { values, index: Float64Array.from(idx) };
  });
}

test("alignOutputs: repeated live drains, concatenated, equal batch", async () => {
  await ready();
  const x = Input("x"), y = Input("y"), m = Input("m");
  const n = 90;
  const xs = Array.from({ length: n }, (_, i) => Math.sin(i * 0.7) * 3 + (i % 11 === 0 ? NaN : 0));
  const xi = Array.from({ length: n }, (_, i) => Math.floor(i / 2));
  const ys = Array.from({ length: n / 3 }, (_, i) => [i, i * 0.5]);
  const yi = Array.from({ length: n / 3 }, (_, i) => 1 + i * 2);
  const ms = Array.from({ length: n }, (_, i) => (i % 3 === 0 ? 1 : 0));
  const graphs = [
    () => [RollingMean(3)(x) as any, y, select(y, [1]), filter(x, m), resample(x, { every: 7, agg: "ohlc" })],
    () => [delay(x, 3), resample(select(y, [0]), { count: 4, agg: "mean" }), filter(x, m)],
  ];
  for (const outs of graphs) {
    const p = new Pipeline([x, y, m], outs());
    try {
      const batch = p({ x: { values: xs, index: xi }, y: { values: nd(ys), index: yi }, m: { values: ms, index: xi } } as any) as Output[];
      const events: Ev[] = [];
      for (let i = 0; i < n; i++) { events.push(["x", xi[i], xs[i]]); events.push(["m", xi[i], ms[i]]); }
      ys.forEach((r, i) => events.push(["y", yi[i], r]));
      events.sort((a, b) => a[1] - b[1]);
      for (const period of [1, 2, 3, 7, 1000]) {
        const s = p.live();
        const parts: Output[][] = [];
        events.forEach(([name, i, v], k) => {
          s.push(name, i, v);
          if ((k + 1) % period === 0) parts.push(s.result() as Output[]);
        });
        s.flush();
        parts.push(s.result() as Output[]);
        assertSame(concat(parts), batch, `drain every ${period} events`);
      }
    } finally {
      p.dispose();
    }
  }
});

test("maxPending: an integer from 1 to 2**53, checked like Python", async () => {
  await ready();
  const a = Input("a");
  const isType = (re: RegExp) => (e: unknown) => e instanceof TypeError && re.test((e as Error).message);
  const isRange = (re: RegExp) => (e: unknown) => e instanceof RangeError && re.test((e as Error).message);
  for (const ok of [1, 5, 2 ** 53]) {
    const p = new Pipeline([a], [a], { maxPending: ok });
    assert.equal(p.maxPending, ok);
    p.dispose();
  }
  assert.equal(new Pipeline([a], [a]).maxPending, 1_000_000);
  for (const bad of [1.5, NaN, Infinity])
    assert.throws(() => new Pipeline([a], [a], { maxPending: bad }), isType(/maxPending must be an integer, got/));
  for (const bad of ["5", true, null, {}])
    assert.throws(() => new Pipeline([a], [a], { maxPending: bad } as any), isType(/maxPending must be an integer, got/));
  for (const bad of [0, -1, 2 ** 53 + 2])
    assert.throws(() => new Pipeline([a], [a], { maxPending: bad }), isRange(/maxPending must be at least 1 and at most 2\*\*53/));
  assert.throws(() => new Pipeline([a], [combineLatest([a, a], { maxPending: 0 })]), isRange(/maxPending must be at least 1/));
});

test("maxPending: the Pipeline's cap applies to combineLatest, filter and a cumulative driver", async () => {
  await ready();
  // b arrives only after 10 events of a, so 10 events wait in each join.
  const a = Input("a"), b = Input("b");
  const feeds = {
    a: { values: Array.from({ length: 10 }, (_, i) => i + 1), index: Array.from({ length: 10 }, (_, i) => i) },
    b: { values: [1], index: [20] },
  };
  const graphs = [
    () => combineLatest([a, b]),
    () => filter(a, b),
    () => resample(a, { threshold: 3, driver: b, agg: "sum" }),
  ];
  for (const g of graphs) {
    const small = new Pipeline([a, b], [g()], { maxPending: 5 });
    const big = new Pipeline([a, b], [g()], { maxPending: 20 });
    try {
      assert.throws(() => small(feeds as any), (e: unknown) => e instanceof RangeError && /overflow/.test((e as Error).message));
      big(feeds as any);
    } finally {
      small.dispose();
      big.dispose();
    }
  }
  // A node's own maxPending wins over the Pipeline's.
  const own = new Pipeline([a, b], [combineLatest([a, b], { maxPending: 20 })], { maxPending: 5 });
  try { own(feeds as any); } finally { own.dispose(); }
});

// Live drains at any timing, concatenated, equal batch: an output a filter
// empties, a rarely emitting resample, and a filter of an input that is all 0.
test("alignOutputs: live drains at any timing equal batch, with a late or empty output", async () => {
  await ready();
  const n = 1000;
  const xs = Array.from({ length: n }, (_, i) => Math.sin(i * 0.37) * 2);
  const ix = Array.from({ length: n }, (_, i) => i);
  const cases: [string, () => { p: Pipeline; feeds: any; events: Ev[] }][] = [
    ["filter drops all", () => {
      const a = Input("a");
      return { p: new Pipeline([a], [a, filter(a, IsNan()(a) as any)], { maxPending: 100 }),
               feeds: { values: xs, index: ix }, events: xs.map((v, i) => ["a", i, v] as Ev) };
    }],
    ["rare resample", () => {
      const a = Input("a");
      return { p: new Pipeline([a], [RollingMean(5)(a) as any, resample(a, { every: 500, agg: "mean" })], { maxPending: 100 }),
               feeds: { values: xs, index: ix }, events: xs.map((v, i) => ["a", i, v] as Ev) };
    }],
    ["filter of an input", () => {
      const a = Input("a"), m = Input("m");
      const events: Ev[] = [];
      xs.forEach((v, i) => { events.push(["a", i, v]); events.push(["m", i, 0]); });
      return { p: new Pipeline([a, m], [a, filter(a, m)], { maxPending: 100 }),
               feeds: { a: { values: xs, index: ix }, m: { values: xs.map(() => 0), index: ix } }, events };
    }],
  ];
  const timings: [string, (k: number) => boolean][] = [
    ["every 1", () => true], ["every 7", (k) => k % 7 === 0], ["every 50", (k) => k % 50 === 0],
    ["odd", (k) => [5, 6, 333, 999].includes(k)], ["end", () => false],
  ];
  for (const [name, make] of cases) {
    const { p, feeds, events } = make();
    try {
      const batch = p(feeds) as Output[];
      for (const [when, at] of timings) {
        const s = p.live();
        const parts: Output[][] = [];
        events.forEach(([nm, i, v], k) => { s.push(nm, i, v); if (at(k + 1)) parts.push(s.result() as Output[]); });
        s.flush();
        parts.push(s.result() as Output[]);
        // A drain with no rows shows an output of unknown width as 1-D; skip it.
        const full = parts.filter((q) => q[0].index.length);
        assertSame(concat(full.length ? full : parts.slice(-1)), batch, `${name}, drain ${when}`);
      }
    } finally {
      p.dispose();
    }
  }
});
