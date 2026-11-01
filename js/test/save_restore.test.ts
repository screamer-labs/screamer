// Save, load and resume in JavaScript, with the files Python writes.
//
// wasm/smoke/save_oracle.json (devtools/wasm/gen_save_oracle.py writes it;
// tests/test_save_restore.py::test_js_oracle_fresh keeps it current) holds,
// for every operator that can be saved, Python's file after a prefix of
// events and Python's outputs after loading it. The one C++ codec writes and
// reads the files, so a file JavaScript loads it writes back byte for byte,
// in JSON and in CBOR. The numbers in a state come from arithmetic, and a
// native build may fuse a multiply-add that WebAssembly does not (decision
// D4 of the 3.0.0 issues: clang keeps -ffp-contract=on), so JavaScript's own
// file after the same events equals Python's byte for byte when every number
// agrees, and to 1e-9 otherwise, as the other oracles compare. Within
// JavaScript a continuation is bit for bit. The error cases mirror
// test_edited_setting_refuses_or_fresh and test_bad_files_named_errors.
import { test } from "node:test";
import assert from "node:assert/strict";
import { chmodSync, lstatSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, statSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import * as S from "../src/index.ts";
import * as OPS from "../src/generated/ops.ts";
import type { NdArray } from "../src/ndarray.ts";

await S.ready();

type Num = number | null | string;
interface Case {
  name: string;
  settings: Record<string, unknown>;
  prefix: Num[][];
  rest: Num[][];
  json_text: string;
  cbor_hex: string;
  expected_rest_output: Num[][];
}
const ORACLE = JSON.parse(readFileSync(new URL("../../wasm/smoke/save_oracle.json", import.meta.url), "utf8")) as {
  cases: Case[];
  registry: { name: string; settings: Record<string, unknown> }[];
  int_errors: { name: string; settings: Record<string, unknown>; message: string | null }[];
};

// null is NaN, and "Infinity" / "-Infinity" the infinities.
const num = (v: Num): number => (v === null ? NaN : typeof v === "string" ? Number(v) : v);
// Settings keep null ("not given"); only the infinities are decoded, at any depth.
const decodeSettings = (x: any): any =>
  Array.isArray(x) ? x.map(decodeSettings)
    : x !== null && typeof x === "object" ? Object.fromEntries(Object.entries(x).map(([k, v]) => [k, decodeSettings(v)]))
    : typeof x === "string" && /^-?Infinity$/.test(x) ? num(x) : x;
const same = (a: number, b: number) => Object.is(a, b) || (Number.isNaN(a) && Number.isNaN(b));
const flat = (x: any): number[] => (typeof x === "number" ? [x] : Array.from(x.data ?? x, (v: any) => v as number));
const hex = (b: Uint8Array) => Array.from(b, (v) => v.toString(16).padStart(2, "0")).join("");
const bytes = (h: string) => Uint8Array.from(h.match(/../g)!.map((v) => parseInt(v, 16)));

// The library version, which the oracle holds as it was when written.
const noVersion = (text: string) => text.replace(/"library_version": "[^"]*"/, '"library_version": "*"');
const KEY = hex(new TextEncoder().encode("library_version"));
const noVersionHex = (h: string) => {
  const at = h.indexOf("6f" + KEY) + 2 + KEY.length;     // a text string of 15 bytes, then the value
  const n = parseInt(h.slice(at, at + 2), 16) - 0x60;      // a text string shorter than 24 bytes
  return h.slice(0, at) + h.slice(at + 2 + 2 * n);
};

function assertBitwise(got: number[], want: number[], what: string) {
  assert.equal(got.length, want.length, `${what}: length`);
  for (let i = 0; i < got.length; i++) assert.ok(same(got[i], want[i]), `${what}[${i}]: ${got[i]} vs ${want[i]}`);
}

// Equal to 1e-9 (relative, or absolute near zero), NaN equal to NaN.
const close = (a: number, b: number, tol = 1e-9) =>
  same(a, b) || Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
// Operators whose outputs JavaScript and Python round apart by more: at the
// cases' price scale of 1e3, BayesianRegression's 2x2 solve is ill
// conditioned, so the last-bit differences of the native and WASM builds
// grow to a few 1e-9.
const TOL: Record<string, number> = { BayesianRegression: 1e-8 };
function assertClose(got: number[], want: number[], what: string, tol = 1e-9) {
  assert.equal(got.length, want.length, `${what}: length`);
  for (let i = 0; i < got.length; i++) assert.ok(close(got[i], want[i], tol), `${what}[${i}]: ${got[i]} vs ${want[i]}`);
}

// Two trees of the same shape whose numbers agree by `eq`.
function treesAgree(a: any, b: any, eq: (x: number, y: number) => boolean): boolean {
  if (typeof a === "number" || typeof b === "number") return typeof a === "number" && typeof b === "number" && eq(a, b);
  if (Array.isArray(a) || Array.isArray(b))
    return Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every((x, i) => treesAgree(x, b[i], eq));
  if (a !== null && typeof a === "object") {
    const ka = Object.keys(a), kb = Object.keys(b ?? {});
    return ka.length === kb.length && ka.every((k, i) => k === kb[i] && treesAgree(a[k], b[k], eq));
  }
  return a === b;
}

// Feeds events [0, n) of `inputs` one at a time; the outputs, flat.
function feed(op: any, inputs: Num[][]): number[] {
  const out: number[] = [];
  const n = inputs[0].length;
  for (let t = 0; t < n; t++) out.push(...flat(op(...inputs.map((x) => num(x[t])))));
  return out;
}

const arrays = (inputs: Num[][]) => inputs.map((x) => Float64Array.from(x, num));
// Nested rows (a row op's (T, K) input, PortfolioReport's (T, K, 4) block) as
// an NdArray.
const nd = (x: any): NdArray => {
  const shape: number[] = [];
  for (let y = x; Array.isArray(y); y = y[0]) shape.push(y.length);
  return { data: Float64Array.from(x.flat(Infinity) as Num[], num), shape };
};
// The inputs of a resume: series, or for a row op and PortfolioReport blocks.
const resumeInputs = (op: any, inputs: any[]) =>
  op.rowOp || op.opName === "PortfolioReport" ? inputs.map(nd) : arrays(inputs);

for (const c of ORACLE.cases) {
  const want = c.expected_rest_output.flat().map(num);
  const fed = () => {
    const op = S.fromSettings({ type: c.name, settings: decodeSettings(c.settings) });
    feed(op, c.prefix);
    return op;
  };

  test(`save: ${c.name} writes Python's file after the same events`, () => {
    const op = fed();
    const text = op.dumps();
    const blob = op.dumps("cbor");
    assert.equal(typeof text, "string");
    assert.ok(blob instanceof Uint8Array);
    const py = S.loads(c.json_text);
    assert.deepEqual(op.settings(), py.settings());
    if (treesAgree(op.state(), py.state(), same)) {
      assert.equal(noVersion(text), noVersion(c.json_text));
      assert.equal(noVersionHex(hex(blob)), noVersionHex(c.cbor_hex));
    } else {
      // A fused multiply-add in the native build (D4): the last bits differ.
      assert.ok(treesAgree(op.state(), py.state(), close), `${c.name}: the state differs from Python's`);
      const layout = (t: string) => noVersion(t).replace(/-?(\d+\.?\d*([eE][-+]?\d+)?|NaN|Infinity)/g, "#");
      assert.equal(layout(text), layout(c.json_text));
    }
    op.dispose(); py.dispose();
  });

  test(`save: ${c.name} reads and writes Python's JSON and CBOR files byte for byte`, () => {
    for (const data of [c.json_text, bytes(c.cbor_hex)]) {
      const op = S.loads(data);
      assert.equal(op.opName, c.name);
      // Byte for byte but for the library version, which is this build's.
      assert.equal(noVersion(op.dumps()), noVersion(c.json_text));
      assert.equal(noVersionHex(hex(op.dumps("cbor"))), noVersionHex(c.cbor_hex));
      assertClose(feed(op, c.rest), want, `${c.name} rest`, TOL[c.name]);
      op.dispose();
    }
  });

  test(`save: ${c.name} continues bit for bit after a save`, () => {
    const a = fed();
    const b = S.loads(a.dumps());
    const d = S.loads(a.dumps("cbor"));
    const wantA = feed(a, c.rest);
    assertBitwise(feed(b, c.rest), wantA, `${c.name} JSON`);
    assertBitwise(feed(d, c.rest), wantA, `${c.name} CBOR`);
    a.dispose(); b.dispose(); d.dispose();
  });

  test(`save: ${c.name} fromSettings round-trips and starts fresh`, () => {
    const op = S.fromSettings({ type: c.name, settings: decodeSettings(c.settings) });
    const again = S.fromSettings({ type: c.name, settings: op.settings() });
    assert.deepEqual(again.settings(), op.settings());
    assert.deepEqual(op.state(), again.state());
    feed(op, c.prefix);
    op.reset();
    assert.deepEqual(op.state(), again.state());
    op.dispose(); again.dispose();
  });

  test(`save: ${c.name} resume equals one event at a time, and Python`, () => {
    const op = S.fromSettings({ type: c.name, settings: decodeSettings(c.settings) });
    const ref = fed();
    op.resume(...resumeInputs(op, c.prefix));
    assert.equal(op.dumps(), ref.dumps());
    const got = flat(op.resume(...resumeInputs(op, c.rest)));
    assertBitwise(got, feed(ref, c.rest), `${c.name} resume`);
    assertClose(got, want, `${c.name} resume vs Python`, TOL[c.name]);
    // The end state stays: the object continues from it.
    assert.equal(op.dumps(), ref.dumps());
    op.dispose(); ref.dispose();
  });
}

test("save: fromSettings builds every operator of the registry", () => {
  for (const r of ORACLE.registry) {
    const op = S.fromSettings({ type: r.name, settings: decodeSettings(r.settings) });
    assert.equal(op.opName, r.name);
    const factory = (S as any)[r.name];
    if (!Object.values(r.settings).some((v) => v !== null && typeof v === "object" && !Array.isArray(v))) {
      const camel = Object.fromEntries(Object.entries(decodeSettings(r.settings)).map(
        ([k, v]) => [k.replace(/_([a-z0-9])/g, (_m, ch) => ch.toUpperCase()), v]));
      const ref = factory(camel);
      assert.equal(op.numInputs, ref.numInputs, r.name);
      assert.equal(op.numOutputs, ref.numOutputs, r.name);
      ref.dispose();
    }
    op.dispose();
  }
});

// Runs `fn` with `keep`, which records an op to dispose when `fn` ends.
async function owning(fn: (keep: <T extends { dispose(): void }>(op: T) => T) => void | Promise<void>) {
  const ops: { dispose(): void }[] = [];
  try {
    await fn((op) => { ops.push(op); return op; });
  } finally {
    ops.forEach((op) => op.dispose());
  }
}

const rollingMean20 = () => {
  const a = S.RollingMean(20);
  for (let v = 0; v < 30; v++) a(v);
  return a;
};
const text20 = () => { const a = rollingMean20(); try { return a.dumps() as string; } finally { a.dispose(); } };

test("save: an edited setting refuses, or starts fresh with a warning", () => owning((keep) => {
  const text = text20();
  const edited = text.replace('"window_size": 20', '"window_size": 50');
  assert.notEqual(edited, text);
  assert.throws(() => S.loads(edited), (e: any) => e instanceof S.StateMismatch && /window_size|buffer/.test(e.message));
  const warned: string[] = [];
  const warn = console.warn;
  console.warn = (m: string) => { warned.push(String(m)); };
  let b: any;
  try { b = keep(S.loads(edited, { onMismatch: "fresh" })); } finally { console.warn = warn; }
  assert.equal(warned.length, 1);
  assert.match(warned[0], /RollingMean/);
  const fresh = keep(S.RollingMean(50));
  const x = Array.from({ length: 100 }, (_, i) => i);
  assertBitwise(x.map((v) => b(v)), x.map((v) => fresh(v) as number), "fresh");
}));

type Edit = (d: any) => void;
const STATE_EDITS: [Edit, string][] = [
  [(d) => { d.object.state.values.extra = 1; }, "RollingMean.extra"],
  [(d) => { delete d.object.state.values.sum; }, "RollingMean.sum"],
  [(d) => { d.object.state.values.sum = "a lot"; }, "RollingMean.sum"],
  [(d) => { d.object.state.values.buffer = [1.0, 2.0]; }, "RollingMean.buffer"],
];
const BAD_FILES: [Edit | string | Uint8Array, string][] = [
  [(d) => { d.object.type = "NoSuchOperator"; }, "object.type"],
  ...STATE_EDITS,
  ["{not json", "save JSON"],
  [Uint8Array.from([0xa1, 0x61]), "CBOR"],
  ['{"screamer": "save/2", "object": {}}', "save/2"],
];
const edit = (text: string, fn: Edit) => { const d = JSON.parse(text); fn(d); return JSON.stringify(d); };

test("save: a bad file raises a SaveError naming the place", () => {
  for (const [bad, where] of BAD_FILES) {
    const data = typeof bad === "function" ? edit(text20(), bad) : bad;
    assert.throws(() => S.loads(data), (e: any) => e instanceof S.SaveError && e.message.includes(where),
      `expected a SaveError naming ${where}`);
  }
});

test("save: a failed setState leaves the operator unchanged", () => owning((keep) => {
  for (const [bad, where] of STATE_EDITS) {
    const a = keep(rollingMean20());
    const before = a.state();
    const state = JSON.parse(edit(a.dumps() as string, bad)).object.state;
    assert.throws(() => a.setState(state), (e: any) => e instanceof S.StateMismatch && e.message.includes(where));
    assert.deepEqual(a.state(), before);
    const b = keep(rollingMean20());
    for (let v = 30; v < 60; v++) assert.ok(same(a(v) as number, b(v) as number));
  }
}));

test("save: the error classes", () => {
  for (const E of [S.StateVersionError, S.NotSaveable, S.StateMismatch]) {
    assert.ok(new E("x") instanceof S.SaveError);
  }
  assert.ok(new S.SaveError("x") instanceof RangeError);
  assert.equal(new S.StateMismatch("x").name, "StateMismatch");
  const text = text20().replace('"version": 1', '"version": 99');
  assert.throws(() => S.loads(text), (e: any) => e instanceof S.StateVersionError && /RollingMean.*99.*\b1\b/.test(e.message));
  assert.throws(() => S.loads(text, { onMismatch: "ignore" as any }), /onMismatch/);
  assert.throws(() => S.loads(42 as any), TypeError);
  assert.throws(() => S.fromSettings({ type: "RollingMean", settings: { window: 3 } }),
    (e: any) => e instanceof S.SaveError && e.message.includes("RollingMean.settings.window: unknown parameter"));
  assert.throws(() => S.fromSettings({ type: "RollingMean", settings: { window_size: 0 } }),
    (e: any) => e instanceof S.SaveError && e.message.includes("RollingMean.settings"));
  // A nested operator's errors name the setting that holds it.
  assert.throws(() => S.fromSettings({ type: "Segment", settings: { op: { type: "RollingMean", settings: { window: 3 } } } }),
    (e: any) => e instanceof S.SaveError && e.message.includes("Segment.settings.op.settings.window: unknown parameter"));
});

test("save: every operator of the JavaScript API can be saved", () => {
  const registry = new Map(ORACLE.registry.map((r) => [r.name, r.settings]));
  const names = Object.keys(OPS).filter((k) => /^[A-Z]/.test(k) && typeof (OPS as any)[k] === "function");
  assert.ok(names.length > 0);
  const failed: string[] = [];
  for (const name of names) {
    const settings = registry.get(name);
    if (!settings) {
      failed.push(`${name} (not in the registry)`);
      continue;
    }
    const op = S.fromSettings({ type: name, settings: decodeSettings(settings) });
    try {
      op.dumps();
    } catch (e) {
      if (!(e instanceof S.NotSaveable)) throw e;
      failed.push(name);
    } finally {
      op.dispose();
    }
  }
  assert.deepEqual(failed, []);
});

test("save: a nested operator saves and loads", () => owning((keep) => {
  const a = keep(S.Segment(S.RollingMean(5)));
  const labels = Array.from({ length: 72 }, (_, i) => Math.floor(i / 9));
  const x = labels.map((_, i) => Math.sin(i));
  for (let t = 0; t < 40; t++) a(labels[t], x[t]);
  const text = a.dumps() as string;
  assert.deepEqual(JSON.parse(text).object.settings,
    { op: { type: "RollingMean", settings: { window_size: 5, start_policy: "strict" } } });
  const b = keep(S.loads(text));
  for (let t = 40; t < 72; t++) assert.ok(same(b(labels[t], x[t]) as number, a(labels[t], x[t]) as number));
  // A loaded operator saves in a Pipeline as one built by its factory.
  const c = keep(S.loads(text));
  const l = S.Input("l"), xs = S.Input("x");
  const p = new S.Pipeline([l, xs], [c(l, xs)] as any);
  assert.match(p.toJSONText(), /"cls": ?"Segment"/);
}));

test("save: save and load files, CBOR for a .cbor path", () => owning(async (keep) => {
  const dir = mkdtempSync(join(tmpdir(), "screamer-save-"));
  try {
    const a = keep(rollingMean20());
    for (const name of ["m.json", "m.cbor"]) {
      const path = join(dir, name);
      await a.save(path);
      const raw = readFileSync(path);
      assert.equal(raw[0] === 0x7b, name.endsWith(".json"));
      const b = keep(await S.load(path));
      assert.equal(b.dumps(), a.dumps());
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}));

test("save: resume on a (T, K) array keeps one state per column", () => owning((keep) => {
  const T = 6, K = 3;
  const data = Float64Array.from({ length: 2 * T * K }, (_, i) => Math.cos(i));
  const first = { data: data.subarray(0, T * K), shape: [T, K] };
  const second = { data: data.subarray(T * K), shape: [T, K] };
  const a = keep(S.RollingMean(4));
  a.resume(first);
  assert.throws(() => a(1.0), /per-column state for 3 columns/);
  assert.throws(() => a.resume({ data: new Float64Array(2 * T), shape: [T, 2] }), /holds 3 columns, got 2/);
  assert.throws(() => a.resume(new Float64Array(T)), /holds 3 columns, got a single series/);
  const text = a.dumps() as string;
  assert.ok(Array.isArray(JSON.parse(text).object.state.columns));
  const b = keep(S.loads(text));
  const wantB: any = a.resume(second);
  assertBitwise(flat(b.resume(second)), flat(wantB), "panel");
  // The columns continue as one batch call over both halves gives.
  const whole: any = keep(S.RollingMean(4))({ data, shape: [2 * T, K] });
  assertBitwise(flat(wantB), flat(whole).slice(T * K), "panel vs batch");
  // A default batch call or reset() drops the columns.
  b.reset();
  assert.ok(Number.isNaN(b(1.0) as number));
  a(new Float64Array(5));
  assert.ok(Number.isNaN(a(1.0) as number));
  assert.throws(() => a.resume(2.0 as any), TypeError);
}));

// Inputs of a (T, K) resume: K = 2 columns per input.
const panelOf = (n: number, T: number, shift: number) =>
  Array.from({ length: n }, (_, i) => ({ data: Float64Array.from({ length: 2 * T }, (_, j) => 100 + Math.sin(j + i + shift)), shape: [T, 2] }));

async function* agen(v: number) { yield v; }

test("save: one-event calls and streams raise on an object holding per-column state", () => {
  const checked = new Set<string>();
  for (const r of ORACLE.registry) {
    if (r.name === "PortfolioReport") continue;
    const op = S.fromSettings({ type: r.name, settings: decodeSettings(r.settings) });
    try {
      if ((op as any).rowOp) continue;
      const n = op.numInputs;
      try { op.resume(...panelOf(n, 30, 0)); } catch { continue; }   // keeps no per-column state
      checked.add(r.name);
      const ev = Array.from({ length: n }, (_, i) => 100 + i);
      const panelError = /per-column state for 2 columns/;
      assert.throws(() => op(...ev), panelError, `${r.name}: one event`);
      assert.throws(() => op(...ev.map((v) => [v][Symbol.iterator]())), panelError, `${r.name}: a stream`);
      assert.throws(() => op(...ev.map((v) => agen(v))), panelError, `${r.name}: an async stream`);
      if (n > 1) assert.throws(() => op([ev][Symbol.iterator]()), panelError, `${r.name}: one stream of events`);
      // The columns are kept.
      assert.throws(() => op.resume(...panelOf(n, 30, 1).map((p) => ({ data: p.data.subarray(0, 30), shape: [30, 1] }))),
        /holds 2 columns, got 1/, r.name);
      op.resume(...panelOf(n, 30, 1));
    } finally {
      op.dispose();
    }
  }
  // Every saveable operator but a row op (one object for K assets) and
  // PortfolioReport (tested below) keeps per-column state.
  for (const c of ORACLE.cases) {
    if (c.name === "PortfolioReport") continue;
    const op: any = S.fromSettings({ type: c.name, settings: decodeSettings(c.settings) });
    const row = !!op.rowOp;
    op.dispose();
    if (!row) assert.ok(checked.has(c.name), `${c.name} kept no per-column state`);
  }
});

// A (T, K) block of a row op's input, or a (T, K, 4) one of PortfolioReport's.
const block = (T: number, K: number, F: number, shift: number): NdArray => ({
  data: Float64Array.from({ length: T * K * F }, (_, i) => 1 + ((i * 7 + shift) % 11) + 0.01 * i),
  shape: F === 1 ? [T, K] : [T, K, F],
});
const rows = (x: NdArray, a: number, b: number): NdArray => {
  const w = x.data.length / x.shape[0];
  return { data: x.data.slice(a * w, b * w), shape: [b - a, ...x.shape.slice(1)] };
};

test("save: PortfolioReport resume continues the report for its assets", () => owning((keep) => {
  const T = 12, K = 3, x = block(T, K, 4, 0), prices = block(T, K, 1, 3);
  for (const priced of [false, true]) {
    const args = (a: number, b: number) => (priced ? [rows(x, a, b), rows(prices, a, b)] : [rows(x, a, b)]);
    const want = flat(keep(S.PortfolioReport())(...args(0, T)));
    const pr: any = keep(S.PortfolioReport());
    const first = flat(pr.resume(...args(0, 5)));
    const loaded: any = keep(S.loads(pr.dumps()));
    assertBitwise([...first, ...flat(loaded.resume(...args(5, T)))], want, "loaded");
    assertBitwise([...first, ...flat(pr.resume(...args(5, T)))], want, "resumed");
    assert.throws(() => pr.resume(...(priced ? [block(2, 2, 4, 0), block(2, 2, 1, 0)] : [block(2, 2, 4, 0)])),
      /resume: object holds 3 assets, got 2/);
    pr(x);   // a default batch call is unchanged
    assertBitwise(flat(keep(S.PortfolioReport()).resume(...args(0, T))), want, "fresh");
  }
}));

test("save: a row op's resume continues its assets", () => owning((keep) => {
  const T = 12, K = 3;
  for (const name of ["BacktestPortfolioTarget", "BacktestPortfolioOHLCTarget", "CrossRank", "CrossStd"]) {
    const make = (): any => keep((S as any)[name]());
    const fi = make().rowOp.fieldsIn;
    const ins = Array.from({ length: fi }, (_, f) => block(T, K, 1, f));
    const want = flat(make()(...ins));
    const op = make();
    const first = flat(op.resume(...ins.map((x) => rows(x, 0, 5))));
    const loaded: any = keep(S.loads(op.dumps()));
    const rest = ins.map((x) => rows(x, 5, T));
    assertBitwise([...first, ...flat(loaded.resume(...rest))], want, `${name} loaded`);
    assertBitwise([...first, ...flat(op.resume(...rest))], want, `${name} resumed`);
    assert.throws(() => op.resume(...ins.map(() => block(2, 2, 1, 0))), /resume: object holds 3 assets, got 2/, name);
    assert.throws(() => op.resume(...ins.map(() => new Float64Array(K))), TypeError, name);
    assertBitwise(flat(op(...ins)), want, `${name} default batch`);
    assertBitwise(flat(make().resume(...ins)), want, `${name} fresh`);
  }
}));

test("save: the rename table applies to fromSettings and loads", () => owning((keep) => {
  const M: any = S.current();
  M._serialTestRename("RollingMean", "window", "window_size");
  try {
    const a = keep(S.fromSettings({ type: "RollingMean", settings: { window: 7 } }));
    assert.deepEqual(a.settings(), { window_size: 7, start_policy: "strict" });
    const ref = keep(S.RollingMean(7));
    const old = (ref.dumps() as string).replace('"window_size"', '"window"');
    assert.equal(keep(S.loads(old)).dumps(), ref.dumps());
    // A nested operator's settings are renamed too, in the record Pipeline.toJSON reads.
    const seg = keep(S.fromSettings({ type: "Segment", settings: { op: { type: "RollingMean", settings: { window: 7 } } } }));
    const l = S.Input("l"), xs = S.Input("x");
    assert.match(new S.Pipeline([l, xs], [seg(l, xs)] as any).toJSONText(), /"window_size": ?7/);
    assert.throws(() => S.fromSettings({ type: "RollingMean", settings: { window: 7, window_size: 7 } }),
      (e: any) => e instanceof S.SaveError && /RollingMean.settings.window: renamed to window_size/.test(e.message));
  } finally {
    M._serialTestRenameClear();
  }
  assert.throws(() => S.fromSettings({ type: "RollingMean", settings: { window: 7 } }), S.SaveError);
}));

test("save: wasm/smoke/save_js.json is what scripts/gen_save_js.ts writes today", async () => {
  const { fixtureText, FIXTURE_PATH } = await import("../scripts/gen_save_js.ts");
  const norm = (text: string) => (JSON.parse(text) as { name: string; json_text: string; cbor_hex: string }[])
    .map((c) => ({ ...c, json_text: noVersion(c.json_text), cbor_hex: noVersionHex(c.cbor_hex) }));
  assert.deepEqual(norm(await fixtureText()), norm(readFileSync(FIXTURE_PATH, "utf8")),
    "wasm/smoke/save_js.json is stale: run node --import tsx scripts/gen_save_js.ts");
});

// ---- Limits, nesting and file handling (tests/test_save_restore.py) ----------

test("save: a file nested too deeply raises, and the module keeps working", () => owning((keep) => {
  // A value nested n deep inside the file's object: JSON text and CBOR (a
  // map of one key holding n arrays). 128 levels pass the reader; deeper
  // ones used to overflow the WebAssembly stack and break every later call.
  const json = (n: number) => '{"x": ' + "[".repeat(n - 1) + "0" + "]".repeat(n - 1) + "}";
  const cbor = (n: number) => Uint8Array.from([0xa1, 0x61, 0x78, ...new Array(n - 1).fill(0x81), 0x00]);
  for (const make of [json, cbor]) {
    assert.throws(() => S.loads(make(128)), (e: any) => e instanceof S.SaveError && /"screamer" field/.test(e.message));
    for (const n of [129, 600, 100000])
      assert.throws(() => S.loads(make(n)), (e: any) => e instanceof S.SaveError && /nested too deeply/.test(e.message));
  }
  const a = keep(rollingMean20());
  assert.equal(keep(S.loads(a.dumps("cbor"))).dumps(), a.dumps());
}));

test("save: nested operators load once per level, and their nesting is capped", () => owning((keep) => {
  let op: any = S.RollingMean(3);
  for (let d = 0; d < 16; d++) op = S.Segment(op);
  keep(op);
  for (let t = 0; t < 5; t++) op(...new Array(16).fill(0), t);
  const text = op.dumps() as string;
  const start = performance.now();
  assert.equal(keep(S.loads(text)).dumps(), text);
  assert.ok(performance.now() - start < 1000);
  let node: any = { type: "RollingMean", settings: { window_size: 3 } };
  for (let d = 0; d < 33; d++) node = { type: "Segment", settings: { op: node } };
  const where = "Segment" + ".settings.op".repeat(32) + ": operators nested more than 32 deep";
  assert.throws(() => S.fromSettings(node), (e: any) => e instanceof S.SaveError && e.message === where);
}));

test("save: a nested operator refuses per-column state", () => owning((keep) => {
  const a = keep(S.Segment(S.RollingMean(2)));
  a(0, 4);
  a(0, 5);
  const state: any = a.state();
  state.values.op = { version: 1, columns: [state.values.op.values] };
  assert.throws(() => a.setState(state), (e: any) => e instanceof S.StateMismatch &&
    e.message === "Segment.op.columns: a nested operator holds no per-column state");
  assert.equal(a(0, 6), 5.5);
}));

test("save: an integer setting out of range words the error as Python does", () => {
  assert.ok(ORACLE.int_errors.length > 300);
  for (const c of ORACLE.int_errors) {
    let got: string | null = null;
    try {
      S.fromSettings({ type: c.name, settings: decodeSettings(c.settings) }).dispose();
    } catch (e: any) {
      assert.ok(e instanceof S.SaveError, `${c.name}: ${e}`);
      got = e.message;
    }
    assert.equal(got, c.message, `${c.name} ${JSON.stringify(c.settings)}`);
  }
});

test("save: setState of a value that is not a state raises StateMismatch", () => owning((keep) => {
  const a = keep(rollingMean20());
  for (const [bad, kind] of [[[1], "an array"], ["x", "a string"], [null, "null"], [3, "a number"]] as const)
    assert.throws(() => a.setState(bad), (e: any) => e instanceof S.StateMismatch &&
      e.message === `RollingMean: expected a state {"version": N, "values": {...}}, got ${kind}`);
  assert.throws(() => a.setState(undefined), TypeError);
}));

test("save: a byte order mark, other files and unknown envelope fields", () => owning((keep) => {
  const text = text20();
  assert.equal(keep(S.loads(Uint8Array.from([0xef, 0xbb, 0xbf, ...new TextEncoder().encode(text)]))).dumps(), text);
  for (const bad of ["[1, 2]", new Uint8Array(0), Uint8Array.from([0xef, 0xbb, 0xbf, 0x5b, 0x31, 0x5d]), Uint8Array.from([0])])
    assert.throws(() => S.loads(bad), (e: any) => e instanceof S.SaveError && /neither a JSON object nor CBOR/.test(e.message));
  const d = JSON.parse(text);
  d.note = 1;
  assert.throws(() => S.loads(JSON.stringify(d)), (e: any) => e instanceof S.SaveError && e.message.startsWith('not a screamer save file: unknown top-level field "note"'));
}));

test("save: a save follows a symbolic link and keeps the file's permissions", () => owning(async (keep) => {
  if (process.platform === "win32") return;
  const dir = mkdtempSync(join(tmpdir(), "screamer-save-"));
  try {
    const a = keep(rollingMean20());
    const target = join(dir, "real.json");
    await a.save(target);
    chmodSync(target, 0o600);
    symlinkSync("real.json", join(dir, "link.json"));
    await keep(OPS.RollingMean(7)).save(join(dir, "link.json"));
    assert.ok(lstatSync(join(dir, "link.json")).isSymbolicLink());
    assert.equal(statSync(target).mode & 0o777, 0o600);
    assert.equal(keep(await S.load(target)).settings().window_size, 7);
    symlinkSync("new.json", join(dir, "dangling.json"));
    await a.save(join(dir, "dangling.json"));
    assert.ok(lstatSync(join(dir, "dangling.json")).isSymbolicLink());
    assert.equal(keep(await S.load(join(dir, "new.json"))).dumps(), a.dumps());
    assert.deepEqual(readdirSync(dir).sort(), ["dangling.json", "link.json", "new.json", "real.json"]);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}));

test("save: a resume keeps the panel's asset shape", () => owning((keep) => {
  const T = 4;
  const data = Float64Array.from({ length: T * 6 }, (_, i) => i);
  const a = keep(S.RollingMean(2));
  a.resume({ data, shape: [T, 2, 3] });
  for (const shape of [[T, 3, 2], [T, 6]])
    assert.throws(() => a.resume({ data, shape }), /resume: object holds columns of shape \(2, 3\), got/);
  assert.deepEqual((a.state() as any).shape, [2, 3]);
  const b = keep(S.loads(a.dumps("cbor")));
  assert.throws(() => b.resume({ data, shape: [T, 6] }), /shape \(2, 3\), got \(6,\)/);
  assertBitwise(flat(b.resume({ data, shape: [T, 2, 3] })), flat(a.resume({ data, shape: [T, 2, 3] })), "shape");
}));

test("save: a failed save leaves the previous file and no temporary file", () => owning(async (keep) => {
  const dir = mkdtempSync(join(tmpdir(), "screamer-save-"));
  try {
    const a = keep(rollingMean20());
    const path = join(dir, "m.json");
    await a.save(path);
    await a.save(path);
    assert.deepEqual(readdirSync(dir), ["m.json"]);
    // A target the rename cannot replace (a directory that holds a file).
    const blocked = join(dir, "d.json");
    mkdirSync(blocked);
    writeFileSync(join(blocked, "keep"), "x");
    await assert.rejects(a.save(blocked));
    const x = S.Input("x");
    const pipe = new S.Pipeline([x], [OPS.RollingMean(3)(x)]);
    await assert.rejects(pipe.save(blocked));
    await assert.rejects(pipe.live().save(blocked));
    pipe.dispose();
    assert.deepEqual(readdirSync(dir).sort(), ["d.json", "m.json"]);
    assert.deepEqual(readdirSync(blocked), ["keep"]);
    assert.equal(keep(await S.load(path)).dumps(), a.dumps());
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}));

