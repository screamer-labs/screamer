import { test } from "node:test";
import assert from "node:assert/strict";
import {
  ready, current, Input, Pipeline, select, resample, combineLatest, filter, delay, dropna,
  split, forecastPairs, RollingMean, Last, type Output,
} from "../src/index.ts";

// Stream robustness (Task 30), the twin of tests/test_stream_robustness.py: a
// decreasing index raises at every stream entry point (G26), a fill resample
// bounds the empty bars of one gap (G22), and a clock resample honors
// watermarks (G19, G20).

const DEC = /the index of input 0 decreases at event 2, from 20 to 10; a stream's index must be non-decreasing/;
const V = [1, 2, 3, 4];
const K = [0, 20, 10, 30];          // decreases at event 2
const EQ = [0, 10, 10, 30];         // equal indexes are allowed
const arr = (x: any): number[] => Array.from((x?.data ?? x) as Float64Array);
const one = (r: Output | Output[]) => (Array.isArray(r) ? r : [r]);

function pipe(op: (x: any) => any = (x) => RollingMean(1)(x)) {
  const x = Input("x");
  return new Pipeline([x], [op(x)]);
}

function run(p: Pipeline, feeds: any) {
  try { return p(feeds); } finally { p.dispose(); }
}

function live(k: number[], wide = false) {
  const p = pipe(wide ? (x) => select(x, 0) : undefined);
  try {
    const s = p.live();
    V.forEach((v, i) => s.push("x", k[i], wide ? [v, v] : v));
    s.flush();
    return s.result();
  } finally { p.dispose(); }
}

function* events(v: number[], k: number[]) {
  for (let i = 0; i < v.length; i++) yield { value: v[i], index: k[i] };
}

const ENTRY_POINTS: Record<string, (k: number[]) => unknown> = {
  "combineLatest": (k) => combineLatest([{ values: V, index: k }, { values: [5, 6], index: [1, 2] }]),
  "combineLatest iterators": (k) => [...combineLatest([events(V, k), events([5, 6], [1, 2])])],
  "Pipeline": (k) => run(pipe(), { x: { values: V, index: k } }),
  "Pipeline 2 inputs": (k) => {
    const a = Input("a"), b = Input("b");
    return run(new Pipeline([a, b], [RollingMean(1)(a), RollingMean(1)(b)]),
      { a: { values: V, index: k }, b: { values: [5, 6], index: [1, 2] } });
  },
  "Pipeline live": (k) => live(k),
  "Pipeline live wide": (k) => live(k, true),
  "resample every": (k) => run(pipe((x) => resample(x, { every: 10, agg: "last" })), { x: { values: V, index: k } }),
  "resample count": (k) => run(pipe((x) => resample(x, { count: 2, agg: "last" })), { x: { values: V, index: k } }),
  "delay": (k) => run(pipe((x) => delay(x, 5)), { x: { values: V, index: k } }),
  "dropna": (k) => run(pipe((x) => dropna(x)), { x: { values: V, index: k } }),
  "forecastPairs": (k) => forecastPairs({ values: V, index: k }, { values: V, index: k }, { duration: 5 }),
};

for (const [name, fn] of Object.entries(ENTRY_POINTS)) {
  test(`G26: a decreasing index raises a RangeError naming the input and event: ${name}`, async () => {
    await ready();
    assert.throws(() => fn(K), (e: any) => e instanceof RangeError && DEC.test(e.message));
    fn(EQ);                                  // equal indexes stay allowed
  });
}

test("G26: split names its stream; positional split has no index", async () => {
  await ready();
  assert.throws(() => split(V, [0, 1, 0, 1], { index: K }),
    (e: any) => e instanceof RangeError && /the index of the split stream decreases at event 2, from 20 to 10/.test(e.message));
  assert.throws(() => split(V, [0, 1, 0, 1], { index: [0.5, 2, 1.5, 3] }), /from 2 to 1\.5/);
  split(V, [0, 1, 0, 1], { index: EQ });
  split(V, [0, 1, 0, 1]);
});

test("G26: a rejected live event leaves the session usable; an event below advance() raises", async () => {
  await ready();
  const p = pipe();
  const a = Input("a"), b = Input("b");
  const p2 = new Pipeline([a, b], [RollingMean(1)(a), RollingMean(1)(b)]);
  try {
    const s = p.live();
    s.push("x", 0, 1); s.push("x", 20, 2);
    assert.throws(() => s.push("x", 10, 3), /decreases at event 2, from 20 to 10/);
    s.push("x", 30, 4); s.flush();
    const got = s.result() as Output;
    const want = p({ x: { values: [1, 2, 4], index: [0, 20, 30] } }) as Output;
    assert.deepEqual(arr(got.index), arr(want.index));
    assert.deepEqual(arr(got.values), arr(want.values));
    const s2 = p2.live();
    s2.push("a", 0, 1); s2.advance(50); s2.advance(20);
    assert.throws(() => s2.push("b", 45, 2),
      (e: any) => e instanceof RangeError && /the index of input 1 is 45 at event 0, below an earlier advance\(50\)/.test(e.message));
    s2.push("b", 50, 2);
  } finally { p.dispose(); p2.dispose(); }
});

// ---- G22 -------------------------------------------------------------------

const FILL = (bars: number, max: number) =>
  new RegExp(`a fill Resample would emit ${bars} empty bars for one gap, more than max_fill=${max} \\(maxFill in JavaScript\\)`);

test("G22: a huge gap raises before it fills; maxFill raises the limit; a normal gap fills", async () => {
  await ready();
  for (const fill of ["nan", "carry"] as const) {
    const gap = { x: { values: [1, 2], index: [0, 2_000_001] } };
    assert.throws(() => run(pipe((x) => resample(x, { every: 1, fill })), gap),
      (e: any) => e instanceof RangeError && FILL(2_000_000, 1_000_000).test(e.message));
    const out = run(pipe((x) => resample(x, { every: 1, fill, maxFill: 2_000_000 })), gap) as Output;
    assert.equal(out.index.length, 2_000_002);
    const small = run(pipe((x) => resample(x, { every: 10, fill })), { x: { values: [1, 2], index: [0, 95] } }) as Output;
    assert.deepEqual(arr(small.index), [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]);
  }
});

test("G22: the limit counts the empty bars of one gap, batch and live, builtin and functor", async () => {
  await ready();
  const feed = { x: { values: [1, 2], index: [0, 60] } };
  for (const reducer of [undefined, Last()(Input("r"))]) {
    const opts = (maxFill: number) => ({ every: 10, fill: "nan" as const, maxFill, ...(reducer ? { reducer } : {}) });
    const ok = run(pipe((x) => resample(x, opts(5))), feed) as Output;
    assert.deepEqual(arr(ok.index), [0, 10, 20, 30, 40, 50, 60]);
    assert.throws(() => run(pipe((x) => resample(x, opts(4))), feed), FILL(5, 4));
    const p = pipe((x) => resample(x, opts(4)));
    try {
      const s = p.live();
      s.push("x", 0, 1);
      assert.throws(() => s.push("x", 60, 2), FILL(5, 4));
    } finally { p.dispose(); }
  }
  const skip = run(pipe((x) => resample(x, { every: 10, maxFill: 1 })), feed) as Output;
  assert.deepEqual(arr(skip.index), [0, 60]);
});

test("G22: advance(2**53) with every=10 and a fill raises cleanly", async () => {
  await ready();
  const p = pipe((x) => resample(x, { every: 10, fill: "nan" }));
  try {
    const s = p.live();
    s.push("x", 0, 1);
    assert.throws(() => s.advance(2 ** 53),
      (e: any) => e instanceof RangeError && /a fill Resample would emit 900719925474098 empty bars/.test(e.message));
    assert.throws(() => s.push("x", 1, 1), /rejected an event/);
    const s2 = p.live();
    s2.push("x", 0, 1); s2.advance(1000); s2.flush();
    assert.equal((s2.result() as Output).index.length, 100);
  } finally { p.dispose(); }
});

test("G22: maxFill follows the count-setting rule", async () => {
  await ready();
  for (const bad of [0, -1, 2 ** 53 + 2])
    assert.throws(() => pipe((x) => resample(x, { every: 10, fill: "nan", maxFill: bad })),
      (e: any) => e instanceof RangeError && /maxFill must be at least 1 and at most 2\*\*53/.test(e.message));
  for (const bad of [1.5, NaN, true, null as any, {}])
    if (bad !== null)
      assert.throws(() => pipe((x) => resample(x, { every: 10, fill: "nan", maxFill: bad as any })),
        (e: any) => e instanceof TypeError && /maxFill must be an integer/.test(e.message));
  assert.throws(() => resample(Input("x"), { every: 10, maxFill: "3" as any }), TypeError);
  for (const ok of [1, 2 ** 53, 7]) pipe((x) => resample(x, { every: 10, fill: "nan", maxFill: ok })).dispose();
  assert.equal((current() as any).defaultMaxFill(), 1_000_000);
});

// ---- G19, G20 ---------------------------------------------------------------

test("G20: a gated value joins the bar of its tick, whatever the advance timing", async () => {
  await ready();
  const x = Input("x"), m = Input("m"), c = Input("c");
  const p = new Pipeline([x, m, c], [resample(filter(x, m), { clock: true, driver: c, agg: "last" })]);
  const feeds = { x: { values: [1], index: [5] }, m: { values: [1, 1, 1], index: [0, 5, 20] }, c: { values: [0, 0, 0], index: [6, 9, 12] } };
  try {
    const want = p(feeds) as Output;
    assert.deepEqual(arr(want.index), [6]);
    const evs: [number, string, number][] = [];
    for (const [n, f] of Object.entries(feeds)) f.index.forEach((t, i) => evs.push([t, n, f.values[i]]));
    evs.sort((a, b) => a[0] - b[0] || a[1].localeCompare(b[1]));
    for (const adv of [null, 6, 8, 11, 19]) {
      const s = p.live();
      const idx: number[] = [];
      let advanced = false;
      for (const [t, n, v] of evs) {
        if (adv !== null && t > adv && !advanced) { s.advance(adv); advanced = true; idx.push(...arr((s.result() as Output).index)); }
        s.push(n, t, v);
        idx.push(...arr((s.result() as Output).index));
      }
      s.flush();
      idx.push(...arr((s.result() as Output).index));
      assert.deepEqual(idx, [6], `advance(${adv})`);
    }
  } finally { p.dispose(); }
});

test("G19: advance(now) releases aligned rows while the clock is silent", async () => {
  await ready();
  const a = Input("a"), c = Input("c");
  const p = new Pipeline([a, c], [RollingMean(1)(a), resample(a, { clock: true, driver: c, agg: "last", fill: "carry" })]);
  try {
    const ka = Array.from({ length: 49 }, (_, i) => i + 1);
    const want = one(p({ a: { values: ka, index: ka }, c: { values: [1, 1], index: [2, 60] } }));
    const wk = arr(want[0].index);
    const s = p.live();
    s.push("a", 1, 1); s.push("c", 2, 1);
    for (let t = 2; t < 50; t++) s.push("a", t, t);
    const k0 = arr(one(s.result())[0].index);
    assert.ok(k0.length <= 1, "rows wait for the silent clock");
    s.advance(50);
    const k1 = arr(one(s.result())[0].index);
    assert.deepEqual([...k0, ...k1], wk.filter((t) => t < 50));
    s.push("c", 60, 1); s.flush();
    const k2 = arr(one(s.result())[0].index);
    assert.deepEqual([...k0, ...k1, ...k2], wk);
  } finally { p.dispose(); }
});

test("G19: a stalled clock with periodic advance keeps the WASM heap flat", async () => {
  await ready();
  const M: any = current();
  const a = Input("a"), c = Input("c");
  const p = new Pipeline([a, c], [RollingMean(5)(a), resample(a, { clock: true, driver: c, agg: "last" })]);
  try {
    const runFor = (n: number) => {
      const s = p.live();
      s.push("c", 0, 1);
      for (let t = 0; t < n; t++) {
        s.push("a", t, t);
        if (t % 100 === 99) { s.advance(t); s.result(); }
      }
    };
    runFor(100_000);                        // warm-up
    const h0 = M.heapInUse();
    runFor(1_000_000);
    const grew = M.heapInUse() - h0;
    assert.ok(grew < 65536, `WASM heap grew by ${grew} bytes`);
  } finally { p.dispose(); }
});
