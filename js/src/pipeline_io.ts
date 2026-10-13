// Pipeline.toJSON / Pipeline.fromJSON: the saved form of a Pipeline, Python's
// to_json / from_json. The schema, the JSON reader and the writer are C++
// (include/screamer/dag/graph_json.h, bound in wasm/graph_json_bindings.cpp),
// shared with Python; this module walks the nodes into a description and
// builds nodes back from a checked one with the JavaScript factories.
import { Input, isNode, type Node } from "./node.js";
import { combineLatest, delay, dropna, filter, resample, select } from "./combinators.js";
import { forecastPairs } from "./supervised.js";
import * as ops from "./generated/ops.js";
import { current } from "./index.js";
import { aKind } from "./kinds.js";
import { SETTINGS } from "./runtime.js";

type Dict = Record<string, any>;

// JSON text of a description, with a non-finite number spelled as Python's
// json module writes it (NaN, Infinity, -Infinity) and -0 as -0.0;
// JSON.stringify alone would write null and 0. Such a number goes through a
// marker string that no setting holds.
const MARK = "\u0000";
const MARKED = /"\\u0000(NaN|Infinity|-Infinity|-0\.0)"/g;
function jsonText(x: unknown): string {
  return JSON.stringify(x, (_k, v) => (typeof v !== "number" ? v
    : !Number.isFinite(v) ? MARK + String(v) : Object.is(v, -0) ? MARK + "-0.0" : v))
    .replace(MARKED, "$1");
}

// A functor node's op as Python writes it: its class and the settings its
// factory recorded.
function functorOf(node: Node, who: string): { cls: string; params: Dict } {
  const rec = (node.op as any)?.wrapper?.[SETTINGS];
  if (!rec) throw new TypeError(`${who}: a functor without recorded settings cannot be saved; build it with its factory`);
  // The settings given (undefined, and null for an optional setting, are not
  // given), by their Python names; an array setting is copied.
  const params: Dict = {};
  (rec.names as string[]).forEach((n, i) => {
    const v = rec.given[i];
    if (v === undefined || v === null) return;
    params[n] = Array.isArray(v) || ArrayBuffer.isView(v) ? Array.from(v as ArrayLike<unknown>) : v;
  });
  return { cls: rec.cls, params };
}

/** The description of a pipeline, as Python's `Pipeline.to_dict()` gives it. */
export function describe(inputs: Node[], outputs: Node[], alignOutputs: boolean, maxPending: number): Dict {
  const ids = new Map<Node, number>();
  const nodes: Dict[] = [];
  // Each node once, numbered in dependency order (its inputs first), as
  // Python's graph walk numbers them.
  const visit = (node: Node): number => {
    const seen = ids.get(node);
    if (seen !== undefined) return seen;
    const inIds = node.inputs.map(visit);
    const op = node.op as any;
    const id = nodes.length;
    ids.set(node, id);
    if (op?.input !== undefined) {
      nodes.push({ id, kind: "input", name: op.input });
    } else if (op?.functor !== undefined) {
      const f = functorOf(node, "Pipeline.toJSON");
      nodes.push({ id, kind: "functor", cls: f.cls, params: { ...f.params }, in: inIds });
    } else if (op?.spec) {
      const params: Dict = {};
      for (const [k, v] of Object.entries(op.spec.params as Dict)) {
        if (isNode(v)) {   // a Resample reducer
          const f = functorOf(v, "Pipeline.toJSON");
          params[k] = { functor: f.cls, params: { ...f.params } };
        } else {
          params[k] = v;
        }
      }
      nodes.push({ id, kind: "operator", op: op.spec.op, params, in: inIds });
    } else {
      throw new TypeError("Pipeline.toJSON: a node of an unknown kind");
    }
    return id;
  };
  const outIds = outputs.map(visit);
  return {
    screamer_dag: 1,
    inputs: inputs.map((n) => (n.op as any).input),
    align_outputs: alignOutputs,
    max_pending: maxPending,
    nodes,
    outputs: outIds,
  };
}

// The result of a C++ graph call: { value } or { error }, the error thrown
// here (graph_json_bindings.cpp never throws, so a bad description frees
// everything it was given).
function unwrap(r: { value?: any; error?: string }): any {
  if (r.error !== undefined) throw new RangeError(r.error);
  return r.value;
}

/** The JSON text of a description, as Python's `to_json(indent)` writes it. */
export function describeJson(d: Dict, indent: unknown): string {
  if (indent !== null && (typeof indent !== "number" || !Number.isInteger(indent) || indent < 0))
    throw new TypeError(`Pipeline.toJSONText: indent must be a non-negative integer or null, got ${aKind(indent)}`);
  const M: any = current();
  return unwrap(M.graphToJson(jsonText(d), indent === null ? -1 : (indent as number)));
}

/** A description, JSON text or an object, checked against the shared schema, in its normal form. */
export function checkDescription(x: unknown, who: string): Dict {
  const M: any = current();
  if (typeof x === "string") return unwrap(M.graphFromJson(x));
  if (x === null || typeof x !== "object" || Array.isArray(x))
    throw new TypeError(`${who}: takes the JSON text or a description object, got ${aKind(x)}`);
  return unwrap(M.graphFromJson(jsonText(x)));
}

const camel = (k: string) => k.replace(/_([a-z0-9])/g, (_m, c) => c.toUpperCase());

// The op of a functor node: its factory, given its settings by name (or
// positionally under "args", Python's fallback for a functor without a schema).
function makeFunctor(cls: string, params: Dict): any {
  const factory = /^[A-Z]/.test(cls) ? (ops as any)[cls] : undefined;
  if (typeof factory !== "function") throw new RangeError((current() as any).unknownFunctorMessage(cls));
  const { args, ...named } = params;
  if (args !== undefined) return factory(...(args as unknown[]));
  const options: Dict = {};
  for (const [k, v] of Object.entries(named)) options[camel(k)] = v;
  return factory(options);
}

// The node of an operator, built by its factory from Python's parameters.
function makeOperator(op: string, p: Dict, feeds: Node[]): Node {
  switch (op) {
    case "CombineLatest":
      return combineLatest(feeds, { emit: p.emit, maxPending: p.max_pending });
    case "Dropna":
      return dropna(feeds, { how: p.how });
    case "Select":
      return select(feeds[0], p.columns);
    case "Filter":
      return filter(feeds[0], feeds[1]);
    case "Delay":
      return delay(feeds[0], p.duration);
    case "forecast_pairs":
      return forecastPairs(feeds[0], feeds[1], { count: p.count, duration: p.duration, dropna: p.dropna });
    case "Resample": {
      const agg = p.agg;
      const functor = agg !== null && typeof agg === "object" && "functor" in agg;
      return resample(feeds[0], {
        every: p.every ?? p.freq, count: p.count, threshold: p.threshold, clock: p.clock,
        origin: p.origin, label: p.label, fill: p.fill, maxAge: p.max_age, maxFill: p.max_fill,
        agg: functor ? undefined : agg,
        reducer: functor ? makeFunctor(agg.functor, agg.params ?? {})(Input("_reducer")) : undefined,
        driver: feeds[1],
      });
    }
    default:
      throw new TypeError(`Pipeline.fromJSON: unknown operator '${op}'`);   // the schema refuses it first
  }
}

/** The inputs, outputs and options of the pipeline a checked description gives. */
export function build(d: Dict): { inputs: Node[]; outputs: Node[]; opts: Dict } {
  const built = new Map<number, Node>();
  const byName = new Map<string, Node>();
  for (const [k, n] of (d.nodes as Dict[]).entries()) {
    if (n.kind === "input") {
      const node = Input(n.name);
      built.set(n.id, node);
      byName.set(n.name, node);
      continue;
    }
    const feeds = (n.in as number[]).map((i) => built.get(i)!);
    if (n.kind === "functor") {
      const op = makeFunctor(n.cls, n.params);
      const msg = (current() as any).functorInputsMessage(k, n.cls, op.numInputs, feeds.length) as string;
      if (msg) { op.dispose(); throw new RangeError(msg); }
      built.set(n.id, op(...feeds));
    } else {
      built.set(n.id, makeOperator(n.op, n.params, feeds));
    }
  }
  const opts: Dict = { alignOutputs: d.align_outputs };
  if (d.max_pending !== undefined) opts.maxPending = d.max_pending;
  return {
    inputs: (d.inputs as string[]).map((name) => byName.get(name)!),
    outputs: (d.outputs as number[]).map((id) => built.get(id)!),
    opts,
  };
}
