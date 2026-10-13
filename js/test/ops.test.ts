import { test } from "node:test";
import assert from "node:assert/strict";
import { ready, RollingMean, MovingAverage, EwMean, BacktestPortfolioTarget } from "../src/index.ts";

test("generated factory: ready() + RollingMean(3) scalar stream", async () => {
  await ready();
  const op = RollingMean(3);
  assert.ok(Number.isNaN(op(1))); // strict: NaN until the window fills
  assert.ok(Number.isNaN(op(2)));
  assert.equal(op(3), 2); // mean of [1,2,3]
  op.dispose();
});

test("generated factory: optional args default to NaN sentinel (EwMean)", async () => {
  await ready();
  // span provided, others left as the missing-optional NaN default.
  const op = EwMean(NaN, 3);
  assert.equal(typeof op(1), "number");
  op.dispose();
});

test("generated factory: vector<double> arg (MovingAverage taps)", async () => {
  await ready();
  const op = MovingAverage([0.5, 0.5]);
  assert.equal(typeof op(1), "number");
  op.dispose();
});

test("G3: mutating a parameter array after construction changes nothing", async () => {
  await ready();
  // A stream runs on a copy built after the array changed.
  const x = Array.from({ length: 50 }, (_, i) => Math.sin(i));
  const taps = [0.25, 0.5, 0.25];
  const ma = MovingAverage(taps);
  taps[0] = 9;
  const ref = MovingAverage([0.25, 0.5, 0.25]);
  assert.deepEqual([...ma(x.values())], [...ref(x.values())]);
});

test("G3: per-asset vector parameters are copied too", async () => {
  await ready();
  const T = 30, K = 2;
  const rows = Array.from({ length: T }, (_, t) => ({ w: new Float64Array(K).fill(0.5), px: Float64Array.from({ length: K }, (_, k) => 100 + t * K + k) }));
  const fee = [0.001, 0.002];
  const eng = BacktestPortfolioTarget(fee);
  fee[0] = 0.5;
  const ref = BacktestPortfolioTarget([0.001, 0.002]);
  const run = (e: any) => [...e(rows.map((r) => [r.w, r.px]).values())].map((r: any) => Array.from(r.data ?? r));
  assert.deepEqual(run(eng), run(ref));
  assert.notDeepEqual(run(eng), run(BacktestPortfolioTarget([0.5, 0.002])), "the fee must matter here");
});

// B3 part 4 (Task 23): one plain object as the only argument holds the settings.
test("options: EwMean({ span: 20 }) equals EwMean(undefined, 20)", async () => {
  await ready();
  const x = Array.from({ length: 60 }, (_, i) => Math.sin(i / 3) + i / 10);
  const want = Array.from(EwMean(undefined, 20)(x) as Float64Array);
  assert.deepEqual(Array.from(EwMean({ span: 20 })(x) as Float64Array), want);
  assert.deepEqual(Array.from(EwMean({ __proto__: null, span: 20 } as any)(x) as Float64Array), want);
  assert.deepEqual(Array.from(RollingMean({ windowSize: 5, startPolicy: "expanding" })(x) as Float64Array),
                   Array.from(RollingMean(5, "expanding")(x) as Float64Array));
});

test("options: an unknown key raises TypeError", async () => {
  await ready();
  assert.throws(() => EwMean({ spn: 20 } as any), { name: "TypeError",
    message: "EwMean: unknown option 'spn'; valid options are com, span, halflife, alpha" });
  // the Python name is not the JavaScript one
  assert.throws(() => RollingMean({ window_size: 5 } as any), { name: "TypeError",
    message: "RollingMean: unknown option 'window_size'; valid options are windowSize, startPolicy" });
  // an own __proto__ key (from JSON) is an unknown option
  assert.throws(() => EwMean(JSON.parse('{"__proto__": {"span": 20}, "com": 3}')), { name: "TypeError",
    message: "EwMean: unknown option '__proto__'; valid options are com, span, halflife, alpha" });
});

test("options: an inherited key is never read", async () => {
  await ready();
  Object.defineProperty(Object.prototype, "span", { value: 20, enumerable: true, configurable: true });
  try {
    assert.throws(() => EwMean({}), RangeError);       // span is not set, so no parameter is
    assert.equal(typeof EwMean({ com: 3 }), "function");
  } finally {
    delete (Object.prototype as any).span;
  }
});

test("options: an own non-enumerable key counts", async () => {
  await ready();
  const x = Array.from({ length: 30 }, (_, i) => Math.cos(i));
  const o = Object.defineProperty({}, "span", { value: 20, enumerable: false });
  assert.deepEqual(Array.from(EwMean(o)(x) as Float64Array), Array.from(EwMean(undefined, 20)(x) as Float64Array));
  const typo = Object.defineProperty({ span: 20 }, "spn", { value: 20, enumerable: false });
  assert.throws(() => EwMean(typo), { name: "TypeError",
    message: "EwMean: unknown option 'spn'; valid options are com, span, halflife, alpha" });
  // a symbol key is never a setting name and is ignored
  assert.equal(typeof EwMean({ span: 20, [Symbol("tag")]: 1 } as any), "function");
});

test("options: only the only argument, and only a plain object", async () => {
  await ready();
  assert.throws(() => (EwMean as any)({ span: 20 }, undefined), { name: "TypeError",
    message: "EwMean: an options object must be the only argument" });
  assert.throws(() => (RollingMean as any)({ windowSize: 5 }, "strict"), TypeError);
  // a class instance, a Map or an array is never an options object
  class Settings { span = 20; }
  assert.throws(() => EwMean(new Settings() as any), { name: "TypeError", message: "EwMean: com must be a number, got a Settings" });
  assert.throws(() => EwMean(new Map([["span", 20]]) as any), { name: "TypeError", message: "EwMean: com must be a number, got a Map" });
});

test("options: more arguments than settings raise TypeError", async () => {
  await ready();
  const { Abs } = await import("../src/index.ts");
  assert.throws(() => (RollingMean as any)(5, "strict", 1), { name: "TypeError",
    message: "RollingMean: takes at most 2 arguments, got 3" });
  assert.throws(() => (MovingAverage as any)([0.5, 0.5], 1), { name: "TypeError",
    message: "MovingAverage: takes at most 1 arguments, got 2" });
  assert.throws(() => (Abs as any)(5), { name: "TypeError", message: "Abs: takes no settings, got number" });
  assert.throws(() => (Abs as any)(undefined, undefined), { name: "TypeError", message: "Abs: takes no settings, got 2 arguments" });
  assert.throws(() => (Abs as any)({}, 1), { name: "TypeError", message: "Abs: an options object must be the only argument" });
  // trailing undefined up to the number of settings is a positional default
  assert.equal(typeof RollingMean(5, undefined), "function");
  assert.equal(typeof Abs(undefined), "function");
});

test("options: values pass the positional checks", async () => {
  await ready();
  assert.throws(() => EwMean({ span: "20" as any }), { name: "TypeError", message: "EwMean: span must be a number, got a string" });
  assert.throws(() => EwMean({ span: new String("20") as any }), { name: "TypeError", message: "EwMean: span must be a number, got a string" });
  assert.throws(() => RollingMean({ startPolicy: 3 as any }), { name: "TypeError", message: "RollingMean: startPolicy must be a string, got number" });
  assert.throws(() => RollingMean(5, new String("strict") as any), { name: "TypeError",
    message: "RollingMean: startPolicy must be a string, got a String object" });
  assert.throws(() => MovingAverage({ taps: new Set([0.5, 0.5]) as any }), { name: "TypeError",
    message: "MovingAverage: taps must be a number or an array of numbers, got a Set" });
  assert.throws(() => MovingAverage({ taps: ["0.5"] as any }), { name: "TypeError",
    message: "MovingAverage: taps must be a number or an array of numbers, got a string" });
  assert.throws(() => RollingMean({ windowSize: 0 }), RangeError);
});

test("options: a number setting takes only a number", async () => {
  await ready();
  const { Hold } = await import("../src/index.ts");
  const nd = { data: Float64Array.of(1), shape: [1] };
  const cases: [unknown, string][] = [[true, "boolean"], [[5], "an array"], [nd, "an NdArray"],
    [{}, "an Object"], [new Map(), "a Map"], [5n, "bigint"], [null, "null"], [Float64Array.of(5), "a Float64Array"]];
  for (const [bad, what] of cases) {
    const message = `Hold: release must be a number, got ${what}`;
    assert.throws(() => Hold(3, bad as any), { name: "TypeError", message }, `positional ${what}`);
    assert.throws(() => Hold({ n: 3, release: bad as any }), { name: "TypeError", message }, `option ${what}`);
  }
  assert.throws(() => RollingMean({ windowSize: [3] as any }), { name: "TypeError",
    message: "RollingMean: windowSize must be an integer, got an array" });
});

test("options: an NdArray where the settings belong says so", async () => {
  await ready();
  const { Abs } = await import("../src/index.ts");
  const nd = { data: Float64Array.of(1, 2), shape: [2] };
  assert.throws(() => EwMean(nd as any), { name: "TypeError",
    message: "EwMean: com must be a number or an options object, got an NdArray" });
  assert.throws(() => MovingAverage(nd as any), { name: "TypeError",
    message: "MovingAverage: taps must be a number or an array of numbers or an options object, got an NdArray" });
  assert.throws(() => (Abs as any)(nd), { name: "TypeError",
    message: "Abs: takes no settings, got an NdArray; apply the operator with Abs()(x)" });
});

test("options: an array first argument is still a positional value (MovingAverage)", async () => {
  await ready();
  const x = Array.from({ length: 20 }, (_, i) => i * i);
  const want = Array.from(MovingAverage([0.5, 0.5])(x) as Float64Array);
  assert.deepEqual(Array.from(MovingAverage({ taps: [0.5, 0.5] })(x) as Float64Array), want);
  assert.deepEqual(Array.from(MovingAverage({ taps: Float64Array.of(0.5, 0.5) })(x) as Float64Array), want);
  assert.deepEqual(Array.from(MovingAverage(Float64Array.of(0.5, 0.5) as any)(x) as Float64Array), want);
  // the options object's array is copied too
  const taps = [0.5, 0.5];
  const ma = MovingAverage({ taps });
  taps[0] = 9;
  assert.deepEqual(Array.from(ma(x) as Float64Array), want);
});

test("options: undefined is the default; null is only where Python takes None", async () => {
  await ready();
  const x = Array.from({ length: 30 }, (_, i) => Math.cos(i));
  const run = (op: any) => Array.from(op(x) as Float64Array);
  const want = run(EwMean(undefined, 20));
  assert.deepEqual(run(EwMean({ com: null, span: 20, alpha: undefined })), want);
  assert.deepEqual(run(EwMean(null, 20)), want);
  assert.deepEqual(run(RollingMean(undefined, undefined)), run(RollingMean()));
  assert.deepEqual(run(MovingAverage({ taps: undefined })), run(MovingAverage()));
  assert.throws(() => RollingMean(null as any), { name: "TypeError", message: "RollingMean: windowSize must be an integer, got null" });
  assert.throws(() => RollingMean(3, null as any), { name: "TypeError", message: "RollingMean: startPolicy must be a string, got null" });
  assert.throws(() => MovingAverage({ taps: null as any }), { name: "TypeError",
    message: "MovingAverage: taps must be a number or an array of numbers, got null" });
});

test("options: null agrees with Python None for every setting (null_oracle.json)", async () => {
  await ready();
  const { readFileSync } = await import("node:fs");
  const ops: any = await import("../src/generated/ops.ts");
  const oracle: { name: string; base: Record<string, number>; accepts_none: Record<string, boolean> }[] =
    JSON.parse(readFileSync(new URL("../../wasm/smoke/null_oracle.json", import.meta.url), "utf8"));
  const camel = (k: string) => k.replace(/_([a-z])/g, (_m, c) => c.toUpperCase());
  let n = 0;
  for (const { name, base, accepts_none } of oracle) {
    const opts: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(base)) opts[camel(k)] = v;
    for (const [k, accepts] of Object.entries(accepts_none)) {
      let refused = false;
      try { ops[name]({ ...opts, [camel(k)]: null }).dispose(); }
      catch (e) { refused = e instanceof TypeError; }
      assert.equal(!refused, accepts, `${name}(${k}=null): Python ${accepts ? "takes" : "refuses"} None`);
      n++;
    }
  }
  assert.ok(n > 450);
});

test("options: true agrees with Python True for every setting (null_oracle.json)", async () => {
  // A bool is not a number setting in either language (G25): Python refuses
  // True and NumPy bools with TypeError, as JavaScript refuses true.
  await ready();
  const { readFileSync } = await import("node:fs");
  const ops: any = await import("../src/generated/ops.ts");
  const oracle: { name: string; base: Record<string, number>; accepts_bool: Record<string, boolean>;
                  bool_message: Record<string, string | null> }[] =
    JSON.parse(readFileSync(new URL("../../wasm/smoke/null_oracle.json", import.meta.url), "utf8"));
  const camel = (k: string) => k.replace(/_([a-z])/g, (_m, c) => c.toUpperCase());
  // Python's message in JavaScript's words: the camelCase name, "boolean"
  // and "an array of numbers".
  const inJs = (msg: string, k: string) => msg.replace(`: ${k} must be`, `: ${camel(k)} must be`)
    .replace(/, got bool$/, ", got boolean").replace("a sequence of numbers", "an array of numbers");
  let n = 0;
  for (const { name, base, accepts_bool, bool_message } of oracle) {
    const opts: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(base)) opts[camel(k)] = v;
    for (const [k, accepts] of Object.entries(accepts_bool)) {
      let refused = false, message = "";
      try { ops[name]({ ...opts, [camel(k)]: true }).dispose(); }
      catch (e) { refused = e instanceof TypeError; message = (e as Error).message; }
      assert.equal(!refused, accepts, `${name}(${k}=true): Python ${accepts ? "takes" : "refuses"} True`);
      if (!accepts) assert.equal(message, inJs(bool_message[k] as string, k), `${name}(${k}=true) message`);
      n++;
    }
  }
  assert.ok(n > 450);
});

test("options: a stream setting refuses a boolean (G25)", async () => {
  await ready();
  const { Input, resample, select, delay } = await import("../src/index.ts");
  const x = Input("x");
  const msg = (what: string, want = "an integer") => ({ name: "TypeError", message: `${what} must be ${want}, got boolean` });
  assert.throws(() => resample(x, { count: true as any }), msg("resample: count"));
  assert.throws(() => resample(x, { every: true as any }), msg("resample: every"));
  assert.throws(() => resample(x, { threshold: true as any }), msg("resample: threshold", "a number"));
  assert.throws(() => select(x, true as any), msg("select: columns"));
  assert.throws(() => delay(x, true as any), msg("delay: duration"));
});

test("options: a required setting has no default", async () => {
  await ready();
  const { SchmittTrigger, Hold } = await import("../src/index.ts");
  const msg = { name: "TypeError", message: "SchmittTrigger: lower is required" };
  assert.throws(() => (SchmittTrigger as any)(), msg);
  assert.throws(() => (SchmittTrigger as any)(undefined, 3), msg);
  assert.throws(() => (SchmittTrigger as any)({ upper: 3 }), msg);
  assert.throws(() => SchmittTrigger(null as any, 3), { name: "TypeError", message: "SchmittTrigger: lower must be a number, got null" });
  assert.throws(() => (Hold as any)(), { name: "TypeError", message: "Hold: n is required" });
  const x = [0, 2, 4, 2, 0];
  assert.deepEqual(Array.from(SchmittTrigger({ lower: 1, upper: 3 })(x) as Float64Array),
                   Array.from(SchmittTrigger(1, 3)(x) as Float64Array));
});

test("options: an op without settings takes an empty options object", async () => {
  await ready();
  const { Abs } = await import("../src/index.ts");
  assert.deepEqual(Array.from(Abs({})([-1, 2]) as Float64Array), [1, 2]);
  assert.throws(() => Abs({ x: 1 } as any), { name: "TypeError", message: "Abs: unknown option 'x'; Abs has no options" });
});

test("options: every generated op builds from an empty options object as from no arguments", async () => {
  await ready();
  const ops: any = await import("../src/generated/ops.ts");
  let n = 0;
  for (const [name, f] of Object.entries(ops) as [string, any][]) {
    if (typeof f !== "function" || !/^[A-Z]/.test(name)) continue;
    let a: any;
    try { a = f(); } catch { continue; }                // needs arguments
    const b = f({});
    assert.equal(b.numInputs, a.numInputs, name);
    assert.equal(b.numOutputs, a.numOutputs, name);
    a.dispose(); b.dispose(); n++;
  }
  assert.ok(n > 200);
});

test("options: the <Op>Options types check both call forms (tsc)", async () => {
  const ts = (await import("typescript")).default;
  const path = await import("node:path");
  const file = path.resolve(import.meta.dirname, "types/options.types.ts");
  const program = ts.createProgram([file], {
    strict: true, noEmit: true, target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext,
    moduleResolution: ts.ModuleResolutionKind.Bundler, allowImportingTsExtensions: true,
    types: ["node"], lib: ["lib.es2022.d.ts", "lib.esnext.disposable.d.ts"],
  });
  const errors = ts.getPreEmitDiagnostics(program)
    .map((d) => ts.flattenDiagnosticMessageText(d.messageText, "\n"));
  assert.deepEqual(errors, []);
});
