import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import * as screamer from "../src/index.ts";
import { ready, Input, Pipeline, combineLatest, resample, select, dropna, filter, delay } from "../src/index.ts";
import type { Node, Output } from "../src/index.ts";

// Load the Python-oracle fixture (built by devtools/wasm/gen_dag_oracle.py).
const HERE = dirname(fileURLToPath(import.meta.url));
const ORACLE = resolve(HERE, "../../wasm/smoke/dag_oracle.json");

interface Feed { values: (number | null)[]; index: number[] }
interface Expect { values: (number | null)[]; index: number[]; width: number }
interface NodeSpec {
  id: string;
  kind: string;
  name?: string;
  op?: string;
  args?: any[];
  inputs?: string[];
  input?: string;
  emit?: "when_all" | "on_any";
  opts?: Record<string, any>;
  reducer?: { op: string; args: any[] };
  columns?: number[];
  how?: "any" | "all";
  k?: number;
  data?: string;
  mask?: string;
}
interface Build { inputs: string[]; nodes: NodeSpec[]; outputs: string[] }
interface Entry { name: string; build: Build; feeds: Record<string, Feed>; expect: Expect[] }

const entries: Entry[] = JSON.parse(readFileSync(ORACLE, "utf8"));

// JSON null marks NaN (allow_nan=False on the Python side).
function toF64(xs: (number | null)[]): Float64Array {
  const out = new Float64Array(xs.length);
  for (let i = 0; i < xs.length; i++) out[i] = xs[i] === null ? NaN : (xs[i] as number);
  return out;
}

// Resolve an op factory by name from the package's public exports.
function opFactory(name: string): (...args: any[]) => any {
  const f = (screamer as any)[name];
  if (typeof f !== "function") throw new Error(`unknown op factory '${name}'`);
  return f;
}

// Interpret a build description into a JS Pipeline. Nodes are listed in
// topological order, so each node's referenced inputs are already built.
function buildPipeline(build: Build): ReturnType<typeof Pipeline> {
  const byId = new Map<string, Node>();
  const get = (id: string): Node => {
    const n = byId.get(id);
    if (!n) throw new Error(`node '${id}' referenced before definition`);
    return n;
  };

  for (const spec of build.nodes) {
    let node: Node;
    switch (spec.kind) {
      case "input":
        node = Input(spec.name!);
        break;
      case "functor": {
        const factory = opFactory(spec.op!);
        const ins = (spec.inputs ?? []).map(get);
        node = factory(...(spec.args ?? []))(...ins) as Node;
        break;
      }
      case "combineLatest":
        node = combineLatest((spec.inputs ?? []).map(get), { emit: spec.emit ?? "when_all" });
        break;
      case "resample": {
        const opts: Record<string, any> = { ...(spec.opts ?? {}) };
        if (spec.reducer) {
          // A functor reducer is an op applied to a placeholder node; the
          // Pipeline only reads its functor slot (placeholder is never wired in).
          opts.reducer = opFactory(spec.reducer.op)(...spec.reducer.args)(Input("__reducer__"));
        }
        node = resample(get(spec.input!), opts);
        break;
      }
      case "select":
        node = select(get(spec.input!), spec.columns!);
        break;
      case "dropna":
        node = dropna((spec.inputs ?? []).map(get), { how: spec.how ?? "any" });
        break;
      case "delay":
        node = delay(get(spec.input!), spec.k!);
        break;
      case "filter":
        node = filter(get(spec.data!), get(spec.mask!));
        break;
      default:
        throw new Error(`unknown node kind '${spec.kind}'`);
    }
    byId.set(spec.id, node);
  }

  const inputs = build.inputs.map(get);
  const outputs = build.outputs.map(get);
  return new Pipeline(inputs, outputs);
}

function assertNanEq(got: ArrayLike<number>, expected: ArrayLike<number>, label: string, tol = 1e-9) {
  assert.equal(got.length, expected.length, `${label}: length ${got.length} != ${expected.length}`);
  for (let i = 0; i < expected.length; i++) {
    const e = expected[i], g = got[i];
    if (Number.isNaN(e)) assert.ok(Number.isNaN(g), `${label}[${i}]: expected NaN, got ${g}`);
    else assert.ok(Math.abs(g - e) <= tol, `${label}[${i}]: ${g} != ${e}`);
  }
}

// Flatten an Output's values (scalar Float64Array or wide NdArray) row-major.
function flatValues(out: Output): Float64Array {
  const v = out.values as any;
  return v instanceof Float64Array ? v : (v.data as Float64Array);
}

for (const entry of entries) {
  test(`dag parity: ${entry.name}`, async () => {
    await ready();

    const feeds: Record<string, { values: Float64Array; index: Float64Array }> = {};
    for (const [name, feed] of Object.entries(entry.feeds)) {
      feeds[name] = { values: toF64(feed.values), index: Float64Array.from(feed.index) };
    }

    const p = buildPipeline(entry.build);
    try {
      const raw = p(feeds);
      const outs = Array.isArray(raw) ? raw : [raw];
      assert.equal(outs.length, entry.expect.length, `${entry.name}: output count`);
      for (let o = 0; o < outs.length; o++) {
        const exp = entry.expect[o];
        assertNanEq(flatValues(outs[o]), toF64(exp.values), `${entry.name}#${o}.values`);
        assertNanEq(outs[o].index, Float64Array.from(exp.index), `${entry.name}#${o}.index`);
      }
    } finally {
      p.dispose();
    }
  });
}
