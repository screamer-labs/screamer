// B8: Pipeline.toJSON / Pipeline.fromJSON, the same JSON as Python's to_json /
// from_json (one C++ schema, reader and writer: graph_json.h).
//
// wasm/smoke/pipeline_io_oracle.json holds graphs Python saved (every node kind
// and every Resample agg, functor reducers included), their feeds and Python's
// results: each loads here and gives those results. wasm/smoke/pipeline_io_js.json
// holds the graphs as JavaScript saves them, with JavaScript's results, which
// tests/test_pipeline_io_js.py loads in Python. Regenerate it with
//   WRITE_PIPELINE_IO=1 node --import tsx --test test/pipeline_io.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, writeFileSync } from "node:fs";
import {
  ready, current, Pipeline, Input, RollingMean, EwMean, Sub, Clip, BollingerBands,
  combineLatest, dropna, select, delay, resample, filter, forecastPairs,
  type NdArray, type Output,
} from "../src/index.ts";

await ready();

type Plain = { values: any[]; index: any[] };
interface Case { name: string; json: string; inputs: string[]; feeds: Plain[]; outputs: Plain[] }

const ORACLE = new URL("../../wasm/smoke/pipeline_io_oracle.json", import.meta.url);
const JS_FIXTURE = new URL("../../wasm/smoke/pipeline_io_js.json", import.meta.url);
const cases: Case[] = JSON.parse(readFileSync(ORACLE, "utf8"));

// A feed: rows as an NdArray, numbers (null is NaN) as an array.
function feed(f: Plain) {
  const wide = f.values.length > 0 && Array.isArray(f.values[0]);
  const values = wide
    ? ({ data: Float64Array.from(f.values.flat(), (v: number | null) => v ?? NaN), shape: [f.values.length, f.values[0].length] } as NdArray)
    : f.values;
  return { values, index: f.index };
}

function feedsOf(c: { inputs: string[]; feeds: Plain[] }) {
  return Object.fromEntries(c.inputs.map((n, i) => [n, feed(c.feeds[i])]));
}

// Outputs as plain {values, index}, NaN as null, as the fixtures hold them.
function plain(r: Output | Output[]): Plain[] {
  const list = Array.isArray(r) ? r : [r];
  return list.map((o) => {
    const nd = o.values as NdArray;
    const flat = Array.from((nd && (nd as any).shape ? nd.data : o.values) as Float64Array, (v) => (Number.isNaN(v) ? null : v));
    const shape: number[] = (nd as any).shape ?? [flat.length];
    const values = shape.length === 1 ? flat
      : Array.from({ length: shape[0] }, (_, t) => flat.slice(t * shape[1], (t + 1) * shape[1]));
    return { values, index: Array.from(o.index) };
  });
}

function sameOutputs(got: Plain[], want: Plain[], what: string) {
  assert.equal(got.length, want.length, `${what}: outputs`);
  got.forEach((g, o) => {
    assert.deepEqual(g.index, want[o].index, `${what}: index of output ${o}`);
    const a = g.values.flat(), b = want[o].values.flat();
    assert.equal(a.length, b.length, `${what}: values of output ${o}`);
    a.forEach((v: number | null, i: number) => {
      const w = b[i];
      if (v === null || w === null) assert.equal(v, w, `${what}: output ${o} item ${i}`);
      else assert.ok(Math.abs(v - w) <= 1e-9 * Math.max(1, Math.abs(w)), `${what}: output ${o} item ${i}: ${v} vs ${w}`);
    });
  });
}

// The checked description of a text, to compare two texts as descriptions
// (JavaScript writes 3 where Python writes 3.0).
const described = (text: string) => (current() as any).graphFromJson(text).value;

function run(p: Pipeline, c: { inputs: string[]; feeds: Plain[] }) {
  return plain(p(feedsOf(c) as any));
}

test("pipeline io: every graph Python saved loads and runs to Python's results", () => {
  assert.ok(cases.length > 70);
  for (const c of cases) {
    const p = Pipeline.fromJSON(c.json);
    try {
      sameOutputs(run(p, c), c.outputs, c.name);
      // JavaScript writes the same description, and the text loads again.
      const text = p.toJSONText();
      assert.deepEqual(described(text), described(c.json), `${c.name}: toJSONText`);
      // toJSON() is the description object; JSON.stringify(p) its text.
      assert.deepEqual(p.toJSON(), p.toDict());
      if (!/NaN|Infinity/.test(c.json))
        assert.deepEqual(described(JSON.stringify(p)), described(c.json), `${c.name}: JSON.stringify`);
      const q = Pipeline.fromJSON(text);
      sameOutputs(run(q, c), c.outputs, `${c.name} (reloaded)`);
      q.dispose();
      // A second run of the same pipeline: no state carried over.
      sameOutputs(run(p, c), c.outputs, `${c.name} (run again)`);
    } finally {
      p.dispose();
    }
  }
});

test("pipeline io: toJSONText writes Python's text; toJSON is the object JSON.stringify writes", () => {
  // Without floats that JavaScript writes as integers, the text is Python's.
  const c = cases.find((x) => x.name === "Resample count agg=EwMean(alpha=0.5)")!;
  const p = Pipeline.fromJSON(c.json);
  assert.equal(p.toJSONText(), c.json);
  assert.equal(JSON.stringify(p), JSON.stringify(p.toDict()));
  assert.equal(JSON.stringify({ model: p }), JSON.stringify({ model: p.toDict() }));   // embeds, not double-encoded
  // fromJSON takes the text or the object.
  assert.equal(Pipeline.fromJSON(JSON.stringify(p)).toJSONText(), c.json);
  assert.equal(Pipeline.fromJSON(p.toJSON()).toJSONText(), c.json);
  // indent null: one line, as Python's to_json(indent=None).
  assert.ok(!p.toJSONText(null).includes("\n"));
  assert.deepEqual(described(p.toJSONText(null)), described(c.json));
  assert.throws(() => p.toJSONText(-1), { name: "TypeError" });
  assert.deepEqual(Pipeline.fromDict(p.toDict()).toDict(), p.toDict());
  p.dispose();
  // Settings that are not finite are written as Python writes them.
  const clip = Pipeline.fromJSON(cases.find((x) => x.name === "Clip -inf")!.json);
  assert.match(clip.toJSONText(), /"lower": -Infinity/);
  assert.equal((clip.toDict() as any).nodes[1].params.lower, -Infinity);
  clip.dispose();
});

// Graphs built in JavaScript with its own factories, for Python to load.
function nativeGraphs(): { name: string; pipe: Pipeline; from: string }[] {
  const two = () => { const x = Input("x"), y = Input("y"); return { x, y }; };
  const out: { name: string; pipe: Pipeline; from: string }[] = [];
  {
    const { x, y } = two();
    const d = dropna(Sub()(combineLatest([x, y], { emit: "on_any", maxPending: 100 })));
    out.push({ name: "js: chain", from: "CombineLatest", pipe: new Pipeline([x, y], [
      RollingMean({ windowSize: 3 })(d),
      resample(d, { every: 5, reducer: EwMean({ alpha: 0.25 })(Input("r")), fill: "carry" }),
      select(combineLatest([x, y]), [1]),
      delay(BollingerBands(4, 1.5)(x), 2),
      forecastPairs(x, y, { count: 2 }),
    ], { alignOutputs: false, maxPending: 5000 }) });
  }
  {
    const { x, y } = two();
    out.push({ name: "js: filter and clip", from: "CombineLatest", pipe: new Pipeline([x, y], [
      Clip({ lower: -Infinity, upper: 101 })(filter(x, y)),
    ]) });
  }
  {
    const { x, y } = two();
    out.push({ name: "js: resample threshold and clock", from: "CombineLatest", pipe: new Pipeline([x, y], [
      resample(x, { driver: y, threshold: 3, agg: "ohlc" }),
      resample(x, { driver: y, clock: true, agg: "last", fill: "carry", maxAge: 4 }),
    ], { alignOutputs: false }) });
  }
  return out;
}

test("pipeline io: graphs JavaScript saves load in Python (wasm/smoke/pipeline_io_js.json)", () => {
  const entries: Case[] = [];
  for (const c of cases) {
    const p = Pipeline.fromJSON(c.json);
    entries.push({ name: c.name, json: p.toJSONText(), inputs: c.inputs, feeds: c.feeds, outputs: run(p, c) });
    p.dispose();
  }
  for (const g of nativeGraphs()) {
    const base = cases.find((x) => x.name === g.from)!;
    entries.push({ name: g.name, json: g.pipe.toJSONText(), inputs: base.inputs, feeds: base.feeds, outputs: run(g.pipe, base) });
    g.pipe.dispose();
  }
  if (process.env.WRITE_PIPELINE_IO) {
    writeFileSync(JS_FIXTURE, JSON.stringify(entries) + "\n");
    return;
  }
  const committed: Case[] = JSON.parse(readFileSync(JS_FIXTURE, "utf8"));
  assert.equal(committed.length, entries.length, "pipeline_io_js.json is stale: regenerate it");
  committed.forEach((c, i) => {
    assert.equal(c.json, entries[i].json, `${c.name}: pipeline_io_js.json is stale: regenerate it`);
    sameOutputs(entries[i].outputs, c.outputs, c.name);
  });
});

test("pipeline io: a malformed text or description throws naming the problem", () => {
  const ok = cases.find((x) => x.name === "functor")!.json;
  const d = JSON.parse(ok);
  const bad: Array<[unknown, string | RegExp]> = [
    ["{", /^invalid Pipeline JSON: expected a property name in double quotes at line 1, column 2$/],
    ['{"a": 1} x', /^invalid Pipeline JSON: extra data after the JSON value at line 1, column 10$/],
    ["[]", "Pipeline description: a Pipeline description is a JSON object, got an array"],
    [JSON.stringify({ ...d, screamer_dag: 2 }), "unsupported screamer_dag version: 2 (expected 1)"],
    [JSON.stringify({ ...d, nodes: [d.nodes[0], { ...d.nodes[1], kind: "x" }] }), "unknown node kind: 'x'"],
    [JSON.stringify({ ...d, nodes: [d.nodes[0], { ...d.nodes[1], cls: "NoSuchOp" }] }), "unknown functor class: 'NoSuchOp'"],
    [JSON.stringify({ ...d, nodes: [d.nodes[0], { id: 1, kind: "operator", op: "Nope", params: {}, in: [0] }] }),
     "unknown DAG operator: 'Nope'"],
    [JSON.stringify({ ...d, nodes: [d.nodes[0], { ...d.nodes[1], in: [7] }] }),
     "Pipeline description: node 1: input 7 is not an earlier node"],
    [JSON.stringify({ ...d, nodes: [d.nodes[0], { id: 1, kind: "operator", op: "Delay", params: { k: 1 }, in: [0] }] }),
     "Pipeline description: node 1 (Delay): unknown parameter 'k'"],
    [JSON.stringify({ ...d, inputs: ["y"] }), "Pipeline description: \"inputs\" names 'y', which is not an input node"],
    [JSON.stringify({ ...d, outputs: [9] }), "Pipeline description: output 9 is not a node"],
  ];
  for (const [text, message] of bad)
    assert.throws(() => Pipeline.fromJSON(text as string), (e: unknown) => e instanceof RangeError &&
      (typeof message === "string" ? (e as Error).message === message : message.test((e as Error).message)), String(text));
  // A setting its factory refuses throws as the factory does.
  assert.throws(() => Pipeline.fromJSON(JSON.stringify({ ...d, nodes: [d.nodes[0], { ...d.nodes[1], params: { window_size: "3" } }] })),
    { name: "TypeError", message: "RollingMean: windowSize must be an integer, got a string" });
  assert.throws(() => Pipeline.fromJSON(5 as any), { name: "TypeError" });
  assert.throws(() => Pipeline.fromDict("x" as any), { name: "TypeError" });
  // The object form takes the same checks.
  assert.throws(() => Pipeline.fromJSON({ ...d, screamer_dag: 2 }),
    { name: "RangeError", message: "unsupported screamer_dag version: 2 (expected 1)" });
  // A lone surrogate is refused (G6 of the review).
  assert.throws(() => Pipeline.fromJSON(ok.replace('"x"', '"\\ud800"')),
    (e: unknown) => e instanceof RangeError && /^invalid Pipeline JSON: a lone surrogate in a string at line \d+, column \d+$/.test((e as Error).message));
  // A functor fed another number of inputs.
  const twoIn = JSON.parse(cases.find((x) => x.name === "CombineLatest")!.json);
  twoIn.nodes.push({ id: 9, kind: "functor", cls: "RollingMean", params: {}, in: [0, 1] });
  twoIn.outputs = [9];
  assert.throws(() => Pipeline.fromJSON(twoIn), { name: "RangeError",
    message: "Pipeline description: node 3 (RollingMean) takes 1 input, got 2" });
});
