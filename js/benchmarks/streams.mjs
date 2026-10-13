// Per-event speed of the stream paths: the iterator forms of combineLatest and
// a live Pipeline, where every event crosses from JavaScript into WASM, and a
// batch Pipeline for scale. Prints one JSON object, {case: ms per call}, the
// best of 7 runs.
//
//   npm run build:wasm && npm run build && node benchmarks/streams.mjs
//
// Pass another build's index.js to time it, for example a copy of dist/ made
// before a change:
//
//   node benchmarks/streams.mjs /tmp/old-dist/index.js

const target = process.argv[2] ?? "../dist/index.js";
const s = await import(new URL(target, import.meta.url).href);
await s.ready();

const N = 100_000;       // events per source of an iterator or live case
const T = 1_000_000;     // rows of a batch case
const a = new Float64Array(T), b = new Float64Array(T), ia = new Float64Array(T), ib = new Float64Array(T);
let x = 1, y = 1;
for (let i = 0; i < T; i++) {
  a[i] = 100 + Math.sin(i / 1000) + (i % 13) * 0.01;
  b[i] = 50 + Math.cos(i / 700) + (i % 7) * 0.02;
  x += 1 + (i % 3); y += 1 + ((i * 7) % 3);
  ia[i] = x; ib[i] = y;
}
const ev = (v, t, n) => { const o = new Array(n); for (let i = 0; i < n; i++) o[i] = { value: v[i], index: t[i] }; return o; };
const ea = ev(a, ia, N), eb = ev(b, ib, N);
const na = Array.from(a.subarray(0, N)), nb = Array.from(b.subarray(0, N));
const wa = na.map((v) => [v, v]), wb = nb.map((v) => [v]);
// Indexed events whose values are rows of two (a wide merge).
const ewa = ea.map((e) => ({ value: [e.value, e.value], index: e.index }));
const ewb = eb.map((e) => ({ value: [e.value, -e.value], index: e.index }));

const drain = (it) => { let n = 0; for (const _ of it) n++; return n; };

const pipe = (inputs, out) => {
  const ins = inputs.map((n) => s.Input(n));
  return s.Pipeline(ins, [out(...ins)]);
};
const live1 = pipe(["x"], (x) => s.EwMean({ span: 20 })(x));
const multi = pipe(["a", "b"], (a, b) => s.Sub()(s.combineLatest([a, b])));
const lag = pipe(["x"], (x) => s.Lag(1)(x));

const cases = {
  combineLatest_indexed_iter: () => drain(s.combineLatest([ea.values(), eb.values()])),
  merge_indexed_iter: () => drain(s.merge([ea.values(), eb.values()])),
  merge_indexed_iter_rows: () => drain(s.merge([ewa.values(), ewb.values()])),
  merge_positional_iter: () => drain(s.merge([na.values(), nb.values()])),
  combineLatest_positional_iter: () => drain(s.combineLatest([na.values(), nb.values()])),
  combineLatest_positional_iter_arrays: () => drain(s.combineLatest([wa.values(), wb.values()])),
  pipeline_live_push: () => {
    const L = live1.live();
    for (let i = 0; i < N; i++) L.push("x", i, na[i]);
    L.flush();
    return L.result();
  },
  pipeline_live_push_wide: () => {
    const p = pipe(["x"], (x) => s.dropna(x));
    const L = p.live();
    for (let i = 0; i < N; i++) L.push("x", i, wa[i]);
    L.flush();
    const r = L.result();
    p.dispose();
    return r;
  },
  // The per-event operator call (a plain WASM export) and an operator stream.
  op_scalar_calls: () => { const op = s.RollingMean(20); let v = 0; for (let i = 0; i < N; i++) v = op(na[i]); return v; },
  op_scalar_calls_3out: () => { const op = s.BollingerBands(20); let v; for (let i = 0; i < N; i++) v = op(na[i]); return v; },
  op_stream: () => drain(s.RollingMean(20)(na.values())),
  pipeline_lazy: () => drain(live1(ea.values())),
  pipeline_lag: () => lag({ x: { values: a, index: ia } }),
  pipeline_multi_input: () => multi({ a: { values: a, index: ia }, b: { values: b, index: ib } }),
};

const best = (fn, runs = 7) => {
  fn(); fn();                                     // warm up the JIT
  let min = Infinity;
  for (let r = 0; r < runs; r++) {
    const t0 = process.hrtime.bigint();
    fn();
    min = Math.min(min, Number(process.hrtime.bigint() - t0));
  }
  return +(min / 1e6).toFixed(3);
};

const out = {};
// An older build without a case's API records null for it.
for (const [name, fn] of Object.entries(cases)) {
  try { out[name] = best(fn); } catch { out[name] = null; }
}
console.log(JSON.stringify(out, null, 1));
