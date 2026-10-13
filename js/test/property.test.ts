import { test } from "node:test";
import assert from "node:assert/strict";
import {
  ready, Input, Pipeline, select, resample, combineLatest, filter, delay, dropna,
  RollingMean,
} from "../src/index.ts";

// Randomized property: batch == live for multi-output Pipelines (B7), the
// JavaScript twin of tests/test_pipeline_property.py. Each case builds 2 to 4
// outputs from random trees over the inputs a, b (with NaN), m (a mask), c (a
// clock) and y (2 columns): filter, dropna, select, combineLatest, delay,
// RollingMean and resample (every with label left or right and fill skip, nan
// or carry; count; threshold with a dropna driver; clock). 15% of cases are
// unaligned, and in 40% the inputs end with rows that dropna or filter drop (a
// NaN tail, a 0 mask tail). Three live sessions per case drain after each
// event with probability 1, 0.3 and 0.05, with advance(now) between events
// and, at the end, past the last event; the concatenated drains must equal
// batch. advance(now) may pass an input's last event: a resample fills an
// empty bucket only between two buckets with data, so advance changes no bar.
// Cases with a clock resample run without advance. The seeds are fixed.

function rngOf(seed: number) {
  let s = seed >>> 0 || 1;
  const r = () => { s = (s + 0x6d2b79f5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  return { r, int: (lo: number, hi: number) => lo + Math.floor(r() * (hi - lo)) };
}

const arr = (x: any): number[] => Array.from((x?.data ?? x) as Float64Array);

function makeCase(seed: number) {
  const clockOn = true;
  let hasClock = false;
  const g = rngOf(seed);
  const idx = (k: number, start: number) => { const o: number[] = []; let c = start; for (let i = 0; i < k; i++) { c += g.int(0, 4); o.push(c); } return o; };
  const n = g.int(5, 120);
  const nanP = g.r() * 0.5;
  const feeds: Record<string, { values: number[] | number[][]; index: number[] }> = {
    a: { values: Array.from({ length: n }, () => (g.r() < nanP ? NaN : g.r() * 4 - 2)), index: idx(n, g.int(0, 30)) },
  };
  const nb = g.int(1, 60);
  feeds.b = { values: Array.from({ length: nb }, () => (g.r() < 0.2 ? NaN : g.r() * 4 - 2)), index: idx(nb, g.int(0, 60)) };
  const nm = g.int(1, n + 1); const th = g.r();
  feeds.m = { values: Array.from({ length: nm }, () => (g.r() < 0.1 ? NaN : g.r() > th ? 1 : 0)), index: idx(nm, g.int(0, 30)) };
  const nc = g.int(1, 15);
  feeds.c = { values: Array.from({ length: nc }, () => 1), index: idx(nc, g.int(0, 80)).map((v) => v * 3) };
  const ny = g.int(1, 40);
  feeds.y = { values: Array.from({ length: ny }, () => [g.r() < 0.1 ? NaN : g.r(), g.r() < 0.1 ? NaN : g.r()]), index: idx(ny, g.int(0, 40)) };
  if (g.r() < 0.4) {                    // inputs that end with dropped rows
    const tail = (len: number) => len - g.int(1, Math.floor(len / 2) + 2);
    const av = feeds.a.values as number[], bv = feeds.b.values as number[], mv = feeds.m.values as number[];
    av.fill(NaN, Math.max(0, tail(av.length)));
    bv.fill(NaN, Math.max(0, tail(bv.length)));
    mv.fill(0, Math.max(0, tail(mv.length)));
  }
  const ins: Record<string, any> = { a: Input("a"), b: Input("b"), m: Input("m"), c: Input("c"), y: Input("y") };
  const used = new Set<string>();
  const inp = (k: string) => { used.add(k); return ins[k]; };
  const aggs = ["mean", "last", "sum", "first", "count"];
  const scalar = (d: number): any => {
    const r = g.int(0, d < 2 ? 9 : 3);
    if (r === 0) return inp("a");
    if (r === 1) return inp("b");
    if (r === 2) return select(inp("y"), g.int(0, 2));
    if (r === 3) return dropna(scalar(d + 1));
    if (r === 4) return filter(scalar(d + 1), inp("m"));
    if (r === 5) return delay(scalar(d + 1), g.int(0, 6));
    if (r === 6) return RollingMean(g.int(1, 4))(scalar(d + 1));
    if (r === 7) return select(combineLatest([scalar(d + 1), scalar(d + 1)]), g.int(0, 2));
    return rs(d + 1);
  };
  const rs = (d: number): any => {
    const r = g.int(0, clockOn ? 4 : 3);
    const agg = aggs[g.int(0, 5)];
    if (r === 0) {
      const o: any = { every: g.int(1, 30), agg };
      if (g.r() < 0.3) o.label = "right";
      if (g.r() < 0.3) o.fill = g.r() < 0.5 ? "carry" : "nan";
      return resample(scalar(d), o);
    }
    if (r === 1) return resample(scalar(d), { count: g.int(1, 6), agg });
    if (r === 2) return resample(scalar(d), { threshold: 0.5 + g.r() * 3.5, driver: dropna(inp("b")), agg: "sum" });
    hasClock = true;
    return resample(scalar(d), { clock: true, driver: inp("c"), agg: "last" });
  };
  const output = (): any => {
    const r = g.int(0, 4);
    if (r === 0) return combineLatest([scalar(1), scalar(1)]);
    if (r === 1) return g.r() < 0.5 ? inp("y") : dropna(inp("y"));
    return scalar(0);
  };
  const m = g.int(2, 5);
  const outs = Array.from({ length: m }, output);
  const names = ["a", "b", "m", "c", "y"].filter((k) => used.has(k));
  return { g, inputs: names.map((k) => ins[k]), outs, feeds, names, hasClock };
}

function sameOut(got: any[], want: any[]): string | null {
  if (got.length !== want.length) return `count ${got.length} vs ${want.length}`;
  for (let j = 0; j < got.length; j++) {
    const gk = arr(got[j].index), wk = arr(want[j].index);
    if (gk.join() !== wk.join()) return `out ${j} index ${gk.slice(0, 30)} vs ${wk.slice(0, 30)}`;
    const gv = arr(got[j].values), wv = arr(want[j].values);
    if (gv.length !== wv.length || gv.some((v, i) => !Object.is(v, wv[i]))) return `out ${j} values ${gv.slice(0, 20)} vs ${wv.slice(0, 20)}`;
  }
  return null;
}

function concatParts(parts: any[][], m: number, align: boolean) {
  const out: any[] = [];
  for (let j = 0; j < m; j++) {
    let full = parts.filter((p) => (align ? arr(p[0].index).length : arr(p[j].index).length));
    if (!full.length) full = parts.slice(-1);
    out.push({ index: full.flatMap((p) => arr(p[j].index)), values: full.flatMap((p) => arr(p[j].values)) });
  }
  return out;
}


function runCase(seed: number): string | null {
  const { g, inputs, outs, feeds, names, hasClock } = makeCase(seed);
  const align = g.r() < 0.85;
  const p = new Pipeline(inputs, outs, { alignOutputs: align });
  try {
    const bf = (k: string) => k === "y" ? { values: { data: Float64Array.from((feeds.y.values as number[][]).flat()), shape: [feeds.y.index.length, 2] }, index: feeds.y.index } : feeds[k];
    const want0 = p(Object.fromEntries(names.map((k) => [k, bf(k)])) as any);
    const want = (Array.isArray(want0) ? want0 : [want0]) as any[];
    const evs: [number, number, number, string, any][] = [];
    names.forEach((k, i) => feeds[k].index.forEach((t, j) => evs.push([t, i, j, k, (feeds[k].values as any)[j]])));
    evs.sort((x, y) => x[0] - y[0] || x[1] - y[1] || x[2] - y[2]);
    for (const prob of [1, 0.3, 0.05]) {
      const s = p.live();
      const parts: any[][] = [];
      let last: number | null = null;
      for (const [t, , , k, v] of evs) {
        if (!hasClock && last !== null && g.r() < 0.15)
          s.advance(last + g.int(0, t - last + 1));
        s.push(k, t, v);
        last = t;
        if (g.r() < prob) { const r = s.result(); parts.push(Array.isArray(r) ? r : [r]); }
      }
      if (!hasClock && last !== null && g.r() < 0.5) {
        s.advance(g.r() < 0.3 ? 2 ** 62 : last + g.int(0, 200));
        if (g.r() < 0.5) { const r = s.result(); parts.push(Array.isArray(r) ? r : [r]); }
      }
      s.flush();
      const r = s.result(); parts.push(Array.isArray(r) ? r : [r]);
      const msg = sameOut(concatParts(parts, outs.length, align), want);
      if (msg) return `seed ${seed}, drain probability ${prob}, align ${align}: ${msg}`;
    }
    return null;
  } finally {
    p.dispose();
  }
}

const N_CASES = 250, CHUNK = 25;
for (let c = 0; c < N_CASES / CHUNK; c++) {
  test(`property: batch == live on random graphs, seeds ${c * CHUNK} to ${(c + 1) * CHUNK - 1}`, async () => {
    await ready();
    const bad: string[] = [];
    for (let seed = c * CHUNK; seed < (c + 1) * CHUNK; seed++) {
      const m = runCase(seed);
      if (m) bad.push(m);
    }
    assert.deepEqual(bad, []);
  });
}
