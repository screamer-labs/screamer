// Save and load of Pipelines and live sessions in JavaScript, with the
// sessions Python saves.
//
// wasm/smoke/session_oracle.json (devtools/wasm/gen_session_oracle.py writes
// it; tests/test_save_pipeline.py keeps it current) holds, for each graph,
// the events before and after a save, Python's session file and the outputs
// after the second part and a flush. JavaScript continues Python's file, and
// its own session after the same events writes Python's file: byte for byte
// when every number of the state agrees, to 1e-9 otherwise (a native build
// may fuse a multiply-add that WebAssembly does not, see save_restore.test.ts).
// The other tests mirror tests/test_save_pipeline.py.
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import * as S from "../src/index.ts";
import * as OPS from "../src/generated/ops.ts";
import { SETTINGS } from "../src/runtime.ts";
import { combineLatest } from "../src/combinators.ts";
import type { LiveDriver, Output, Pipeline } from "../src/pipeline.ts";

await S.ready();

type Num = number | null;
type Event = [number, number, Num | Num[]];
interface Case {
  name: string;
  inputs: string[];
  first: Event[];
  rest: Event[];
  file: string;
  outputs: { values: any[]; index: number[] }[];
}
const ORACLE = JSON.parse(readFileSync(new URL("../../wasm/smoke/session_oracle.json", import.meta.url), "utf8")) as Case[];

const VERSION = /"library_version": "[^"]*"/;
const noVersion = (text: string) => text.replace(VERSION, '"library_version": "*"');

// A session file's text as a tree, its non-finite numbers as strings, and
// back.
const tree = (text: string): any =>
  JSON.parse(text.replace(/-?Infinity\b|\bNaN\b/g, (m) => `"${m}"`));
const text = (t: any): string => JSON.stringify(t).replace(/"(NaN|-?Infinity)"/g, "$1");

// Whether two trees are equal, numbers to 1e-9 (relative, or absolute near 0).
function close(a: any, b: any, path = "$"): string | null {
  if (typeof a === "number" && typeof b === "number") {
    if (a === b || Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(a), Math.abs(b))) return null;
    return `${path}: ${a} != ${b}`;
  }
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return `${path}: length ${a.length} != ${b.length}`;
    for (let i = 0; i < a.length; i++) { const d = close(a[i], b[i], `${path}[${i}]`); if (d) return d; }
    return null;
  }
  if (a && b && typeof a === "object" && typeof b === "object") {
    const ka = Object.keys(a), kb = Object.keys(b);
    if (ka.join() !== kb.join()) return `${path}: keys ${ka} != ${kb}`;
    for (const k of ka) { const d = close(a[k], b[k], `${path}.${k}`); if (d) return d; }
    return null;
  }
  return a === b ? null : `${path}: ${JSON.stringify(a)} != ${JSON.stringify(b)}`;
}

// The Pipeline of a session file: the same file as a Pipeline's, state null.
// The state is the last field of the object, which is the file's last field.
function pipelineOf(file: string): Pipeline {
  const at = file.indexOf('\n    "state": {');
  const text = file.slice(0, at).replace('"type": "LiveSession"', '"type": "Pipeline"') +
    '\n    "state": null\n  }\n}\n';
  return S.loads<Pipeline>(text);
}

function push(s: LiveDriver, inputs: string[], events: Event[]): void {
  for (const [k, t, v] of events) s.push(inputs[k], t, v === null ? NaN : (v as any));
}

// The outputs of result() as flat numbers.
function flatOutputs(r: Output | Output[]): { values: number[]; index: number[] }[] {
  return (Array.isArray(r) ? r : [r]).map((o) => ({
    values: Array.from((o.values as any).data ?? o.values, (v: any) => v as number),
    index: Array.from(o.index),
  }));
}
const flatExpected = (c: Case) => c.outputs.map((o) => ({
  values: (o.values.flat(2) as Num[]).map((v) => (v === null ? NaN : v)),
  index: o.index,
}));

const sameNum = (a: number, b: number, tol: number) =>
  (Number.isNaN(a) && Number.isNaN(b)) || a === b ||
  (tol > 0 && Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b)));

function assertOutputs(got: { values: number[]; index: number[] }[],
                       want: { values: number[]; index: number[] }[], tol: number, name: string) {
  assert.equal(got.length, want.length, name);
  got.forEach((g, j) => {
    assert.deepEqual(g.index, want[j].index, `${name}: index of output ${j}`);
    assert.equal(g.values.length, want[j].values.length, `${name}: values of output ${j}`);
    g.values.forEach((v, i) => assert.ok(sameNum(v, want[j].values[i], tol),
      `${name}: output ${j} value ${i}: ${v} != ${want[j].values[i]}`));
  });
}

test("JavaScript continues the sessions Python saved", () => {
  for (const c of ORACLE) {
    const s = S.loads<LiveDriver>(c.file);
    push(s, c.inputs, c.rest);
    s.flush();
    assertOutputs(flatOutputs(s.result()), flatExpected(c), 1e-9, c.name);
    s.dispose();
  }
});

test("a JavaScript session writes Python's file, and continues bit for bit", () => {
  let exact = 0;
  for (const c of ORACLE) {
    const pipe = pipelineOf(c.file);
    const s = pipe.live();
    push(s, c.inputs, c.first);
    const saved = s.dumps() as string;
    if (noVersion(saved) === noVersion(c.file)) exact++;
    else assert.equal(close(tree(noVersion(saved)), tree(noVersion(c.file))), null, c.name);
    // Loaded from its own JSON or CBOR, the session continues as one that ran
    // without a stop.
    const whole = pipe.live();
    push(whole, c.inputs, [...c.first, ...c.rest]);
    whole.flush();
    const want = flatOutputs(whole.result());
    for (const format of ["json", "cbor"] as const) {
      const b = S.loads<LiveDriver>(s.dumps(format));
      assert.equal(b.dumps(), saved, `${c.name}: a loaded session saves the same file`);
      push(b, c.inputs, c.rest);
      b.flush();
      assertOutputs(flatOutputs(b.result()), want, 0, `${c.name} (${format})`);
      b.dispose();
    }
    pipe.dispose();
  }
  // Most states hold no fused arithmetic, so the files agree byte for byte.
  assert.ok(exact > ORACLE.length / 2, `only ${exact} of ${ORACLE.length} files are byte for byte`);
});

// x, y -> RollingMean(2)(x) joined with y.
function makePipe(): Pipeline {
  const x = S.Input("x"), y = S.Input("y");
  return new S.Pipeline([x, y], [combineLatest([OPS.RollingMean(2)(x), y])]);
}

test("a Pipeline saves its description only", () => {
  const pipe = makePipe();
  const text = pipe.dumps() as string;
  const f = JSON.parse(text);
  assert.equal(f.object.type, "Pipeline");
  assert.equal(f.object.state, null);
  assert.deepEqual(f.object.settings, pipe.toJSON());
  const loaded = S.loads<Pipeline>(text);
  assert.equal(loaded.toJSONText(), pipe.toJSONText());
  assert.equal(S.loads<Pipeline>(pipe.dumps("cbor")).toJSONText(), pipe.toJSONText());
  loaded.dispose();
  pipe.dispose();
});

test("session and Pipeline files", async () => {
  const dir = mkdtempSync(join(tmpdir(), "screamer-session-"));
  try {
    const pipe = makePipe();
    const s = pipe.live();
    s.push("x", 1, 1.0); s.push("y", 1, 5.0); s.push("x", 2, 2.0);
    for (const name of ["s.json", "s.cbor"]) {
      await s.save(join(dir, name));
      const b = await S.load<LiveDriver>(join(dir, name));
      assert.equal(b.dumps(), s.dumps());
      b.dispose();
      await pipe.save(join(dir, `p-${name}`));
      const p = await S.load<Pipeline>(join(dir, `p-${name}`));
      assert.equal(p.toJSONText(), pipe.toJSONText());
      p.dispose();
    }
    pipe.dispose();
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

function sessionFile(): { pipe: Pipeline; f: any } {
  const x = S.Input("x");
  const pipe = new S.Pipeline([x], [OPS.Clip({ lower: 0 })(OPS.RollingMean(3)(x))]);
  const s = pipe.live();
  for (let t = 0; t < 4; t++) s.push("x", t, t);
  return { pipe, f: tree(s.dumps() as string) };
}

function warnings<T>(f: () => T): { value: T; messages: string[] } {
  const messages: string[] = [];
  const warn = console.warn;
  console.warn = (m: string) => { messages.push(m); };
  try { return { value: f(), messages }; } finally { console.warn = warn; }
}

test("an edited graph refuses the state, or starts fresh", () => {
  const { f } = sessionFile();
  f.object.settings.nodes = f.object.settings.nodes.slice(0, 2);
  f.object.settings.outputs = [1];
  const edited = text(f);
  assert.throws(() => S.loads(edited), (e: any) => e instanceof S.StateMismatch &&
    /LiveSession\.nodes\.2: unknown field/.test(e.message));
  const { value: s, messages } = warnings(() => S.loads<LiveDriver>(edited, { onMismatch: "fresh" }));
  assert.match(messages[0], /LiveSession starts fresh/);
  s.push("x", 10, 1.0);
  assert.ok(Number.isNaN((s.result() as Output).values[0] as number));
  s.dispose();
});

test("an edited setting refuses the state", () => {
  const { f } = sessionFile();
  f.object.settings.nodes[1].params.window_size = 5;
  assert.throws(() => S.loads(text(f)), (e: any) => e instanceof S.StateMismatch &&
    /LiveSession\.nodes\.1\.values\.op\.buffer/.test(e.message));
});

test("a poisoned session cannot be saved", () => {
  const x = S.Input("x"), y = S.Input("y");
  const s = new S.Pipeline([x, y], [combineLatest([x, y], { maxPending: 2 })]).live();
  assert.throws(() => { for (let t = 0; t < 5; t++) s.push("x", t, 1.0); });
  assert.throws(() => s.dumps(), (e: any) => e instanceof S.NotSaveable &&
    /^LiveSession: this session rejected an event/.test(e.message));
});

test("a functor without recorded settings cannot be saved; the error names its node", () => {
  const x = S.Input("x");
  const op = OPS.RollingMean(3);
  delete (op as any)[SETTINGS];
  const pipe = new S.Pipeline([x], [op(x)]);
  assert.throws(() => pipe.dumps(), (e: any) => e instanceof S.NotSaveable && /^Pipeline\.nodes\.1: /.test(e.message));
  assert.throws(() => pipe.live().dumps(), (e: any) => e instanceof S.NotSaveable &&
    /^LiveSession\.nodes\.1: a functor without recorded settings/.test(e.message));
});

test("a lazy call cannot be saved", () => {
  const x = S.Input("x");
  const pipe = new S.Pipeline([x], [OPS.RollingMean(2)(x)]);
  const g: any = pipe([{ value: 1, index: 0 }, { value: 2, index: 1 }][Symbol.iterator]());
  g.next();
  for (const call of [() => g.save("lazy.json"), () => g.dumps()])
    assert.throws(call, (e: any) => e instanceof S.NotSaveable && /Pipeline\.live\(\)/.test(e.message));
  g.return();
});

test("Python's time index unit stays in a JavaScript session's file", () => {
  const { f } = sessionFile();
  f.object.state.engine.index_unit = "datetime64[ns]";
  const s = S.loads<LiveDriver>(text(f));
  assert.equal(tree(s.dumps() as string).object.state.engine.index_unit, "datetime64[ns]");
  s.dispose();
});
