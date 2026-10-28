// Randomization tests for backtest significance. Mirrors screamer/validation.py.
//
// `positionTest` keeps the market data and the strategy's trading profile,
// randomizes when the strategy holds its positions, and replays each copy
// through the same backtest engine. `strategyTest` keeps the strategy and
// randomizes its data (permuted bars or returns, or shifted features), re-running
// the whole strategy on each copy. `walkForward` picks the best of several
// candidates on each training window. These are batch-only validation tools, not
// streaming operators. Each runs in a C++ driver shared with Python
// (include/screamer/validation/position_test.h and strategy_test.h, bound in
// wasm/validation_bindings.cpp), which calls the engine, the strategy, and a
// custom metric back through wasmCallbacks; this module converts the arguments
// and the results. The same seed gives the same surrogates as Python.
import { current } from "./index.js";
import { asNd, type NdArray } from "./ndarray.js";
import { wasmCall, wasmCallbacks } from "./errors.js";
import { aKind, flattenRows } from "./kinds.js";
import type { ScreamerOp } from "./runtime.js";

// A target as a Float64Array, so the engine returns an NdArray: all number[]
// inputs would give an Array of per-event results (rule 3).
const asTyped = (x: ArrayLike<number>): ArrayLike<number> => (Array.isArray(x) ? Float64Array.from(x) : x);

// Numbers go to C++ as they are; anything else as NaN, which the C++ checks
// reject (a seed must be an integer in [0, 2^53)).
const num = (v: unknown): number => (typeof v === "number" ? v : NaN);

// A number argument is not a boolean: TypeError "<function>: <argument> must be
// an integer, got boolean", as in Python. Each entry is [value, integer].
function noBools(who: string, args: Record<string, [unknown, boolean]>): void {
  for (const [name, [v, integer]] of Object.entries(args))
    if (typeof v === "boolean" || (Array.isArray(v) && v.some((e) => typeof e === "boolean")))
      throw new TypeError(`${who}: ${name} must be ${integer ? "an integer" : "a number"}, got boolean`);
}

// Text settings: a string, or null/undefined for the default. Another kind is
// a TypeError, as in Python; C++ would refuse it with an Embind error.
function texts(who: string, args: Record<string, unknown>): void {
  for (const [name, v] of Object.entries(args))
    if (v != null && typeof v !== "string") throw new TypeError(`${who}: ${name} must be a string, got ${aKind(v)}`);
}

// Run a C++ surrogate generator: copy `x` to the heap, write a (T x ncols)
// row-major block, and copy it back out.
function surrogates(
  x: ArrayLike<number>,
  ncols: number,
  run: (xPtr: number, outPtr: number) => void,
): NdArray {
  const M = current();
  const T = x.length;
  let xPtr = 0, outPtr = 0;          // freeBuf(0) is a no-op; a failed alloc throws
  try {
    xPtr = M.allocF64(T); outPtr = M.allocF64(T * ncols);
    M.viewF64(xPtr, T).set(x as number[]);
    run(xPtr, outPtr);
    return { data: new Float64Array(M.viewF64(outPtr, T * ncols)), shape: [T, ncols] };
  } finally {
    M.freeBuf(xPtr); M.freeBuf(outPtr);
  }
}

// The C++ kernels check the length, min_shift, and the seed; their errors
// arrive as RangeError through wasmCall. min_shift -1 is the default.
function shiftBlock(x: ArrayLike<number>, minShift: number, seed: number,
                    col0: number, ncols: number): NdArray {
  return surrogates(x, ncols, (xp, op) => wasmCall(() =>
    current().circularShiftInto(xp, x.length, minShift, num(seed), col0, ncols, op, -1)));
}

function shuffleBlock(x: ArrayLike<number>, seed: number, col0: number,
                      ncols: number): NdArray {
  return surrogates(x, ncols, (xp, op) => wasmCall(() =>
    current().segmentShuffleInto(xp, x.length, num(seed), col0, ncols, op)));
}

/**
 * `n` circularly shifted copies of a series, as a (T, n) array.
 *
 * Each column is `x` rotated by a random offset in `[minShift, L - minShift]`,
 * where `L` is the length after the leading NaN warmup, which stays in place.
 * A rotation keeps every property of the series and changes only its alignment
 * in time. Without `minShift`, twice the mean run length is used, capped at a
 * quarter of `L`.
 */
export function circularShift(
  x: ArrayLike<number>, n: number,
  opts: { minShift?: number; seed?: number } = {},
): NdArray {
  noBools("circularShift", { n: [n, true], minShift: [opts.minShift, true], seed: [opts.seed, true] });
  const m = opts.minShift;
  if (m != null && (!Number.isInteger(m) || m < 1)) throw new RangeError("min_shift must be at least 1");
  return shiftBlock(x, m ?? -1, opts.seed ?? 0, 0, n);
}

/**
 * `n` copies of a series with its runs in random order, as a (T, n) array.
 *
 * Every copy holds exactly the same runs (trades and flat stretches) with the
 * same values and lengths, at random times. No two neighbouring runs have the
 * same value, so no two runs merge and every copy changes position as often
 * as the original. The NaN warmup stays in place.
 */
export function segmentShuffle(
  x: ArrayLike<number>, n: number, opts: { seed?: number } = {},
): NdArray {
  noBools("segmentShuffle", { n: [n, true], seed: [opts.seed, true] });
  return shuffleBlock(x, opts.seed ?? 0, 0, n);
}

/** The statistics computed for every run, named as in `backtestReport`'s summary. */
export type StatName =
  | "total_pnl" | "max_drawdown" | "total_cost" | "turnover" | "num_trades" | "sharpe"
  | "exposure" | "round_trips" | "win_rate" | "profit_factor" | "avg_win" | "avg_loss"
  | "avg_holding" | "sortino" | "calmar";

export type PositionTestMetric = StatName | ((engineOutput: NdArray) => number);

export interface PositionTestOptions {
  n?: number;
  method?: "shift" | "segments";
  metric?: PositionTestMetric;
  largerIsBetter?: boolean;
  minShift?: number;
  seed?: number;
  /** Annualizes the ratios, as in `backtestReport` (default 1: per step). */
  periodsPerYear?: number;
}

export interface PositionTestResult {
  observed: number;
  null: Float64Array;
  pValue: number;
  metric: string;
  method: "shift" | "segments";
  minShift: number | null;
  /** Every statistic of the real run. */
  observedStats: Record<StatName, number>;
  /** Every statistic per randomized run, in run order with NaN kept. */
  nullStats: Record<StatName, Float64Array>;
}

/** The result of `positionTest` for several strategies (one entry per column). */
export interface PositionFamilyResult {
  observed: Float64Array;
  /** (n, K) row-major; NaN where a run's metric is NaN. */
  null: NdArray;
  pValue: Float64Array;
  /** Each strategy against the best of the K randomized metrics per round. */
  familyPValue: Float64Array;
  metric: string;
  method: "shift";
  minShift: number;
  /** Every statistic of each real strategy (length K). */
  observedStats: Record<StatName, Float64Array>;
  /** Every statistic per randomized round and strategy, (n, K) row-major. */
  nullStats: Record<StatName, NdArray>;
}

// The statistic names, in the order of the C++ table (include/screamer/
// validation/scoring.h), the one source shared with Python; the drivers there
// also resolve a metric's name and direction.
const STAT_NAME_SET: Record<StatName, true> = {
  total_pnl: true, max_drawdown: true, total_cost: true, turnover: true, num_trades: true,
  sharpe: true, exposure: true, round_trips: true, win_rate: true, profit_factor: true,
  avg_win: true, avg_loss: true, avg_holding: true, sortino: true, calmar: true,
};

let statNameList: StatName[] | null = null;

// Read once from WASM, and checked against StatName.
function statNames(): StatName[] {
  if (statNameList) return statNameList;
  const rows = wasmCall(() => current().strategyStats()) as [StatName, string, number, number][];
  const names = rows.map((r) => r[0]);
  if (names.length !== Object.keys(STAT_NAME_SET).length || names.some((q) => !(q in STAT_NAME_SET)))
    throw new Error(`StatName does not match the C++ statistic table: ${names.join(", ")}`);
  statNameList = names;
  return names;
}

function statsRecord<V>(make: (i: number) => V): Record<StatName, V> {
  const out = {} as Record<StatName, V>;
  statNames().forEach((k, i) => { out[k] = make(i); });
  return out;
}

// The metric for C++: a statistic name (checked there), or "" with `custom`
// for a function. `name` is the result's metric name.
function metricArgs(metric: unknown, largerIsBetter: unknown) {
  const custom = typeof metric === "function";
  return {
    custom,
    name: custom ? (metric as Function).name || "custom" : String(metric),
    hasLarger: largerIsBetter != null,
    larger: !!largerIsBetter,
  };
}

// Strategy k of a (T, K, 4) output, as one run's (T, 4) output.
function strategyOf(out: NdArray, k: number): NdArray {
  const [T, K, F] = out.shape;
  const data = new Float64Array(T * F);
  for (let t = 0; t < T; t++) data.set(out.data.subarray((t * K + k) * F, (t * K + k + 1) * F), t * F);
  return { data, shape: [T, F] };
}

function column(a: NdArray, col: number): Float64Array {
  const [rows, cols] = a.shape;
  const out = new Float64Array(rows);
  for (let r = 0; r < rows; r++) out[r] = a.data[r * cols + col];
  return out;
}

// [ndim, T, K] of an input as the drivers check it: (T,) for a series, the
// shape of an NdArray, (T, K) for T rows of K values, and ndim 0 for anything
// else, which they reject.
function dims(x: any): [number, number, number] {
  const nd = asNd(x);
  if (nd !== null) return [nd.shape.length, nd.shape[0] ?? 0, nd.shape[1] ?? 1];
  if (x == null || typeof x === "string" || typeof x.length !== "number") return [0, 0, 0];
  const first = x.length > 0 ? x[0] : undefined;
  if (first != null && typeof first === "object" && typeof first.length === "number")
    return [2, x.length, first.length];
  return [1, x.length, 1];
}

// K columns of T values as a row-major (T, K) block (T rows of K values use
// flattenRows from kinds.ts).
function fromColumns(cols: ArrayLike<number>[], what: string): Float64Array {
  const K = cols.length, T = K > 0 ? cols[0].length : 0;
  const data = new Float64Array(T * K);
  cols.forEach((x, k) => {
    if (x == null || x.length !== T) throw new RangeError(`${what} must have the same length`);
    for (let t = 0; t < T; t++) data[t * K + k] = x[t];
  });
  return data;
}

// A target for the drivers: its row-major data, [ndim, T, K], and the object
// the engine gets for the real run.
interface Target { data: ArrayLike<number>; ndim: number; T: number; K: number; original: any }

const isColumns = (x: any): x is ArrayLike<number>[] =>
  Array.isArray(x) && x.length > 0 && typeof x[0] !== "number";

// positionTest: one series, K strategy columns, or for a portfolio engine a
// (T, K) NdArray or T rows of K values.
function positionTarget(target: any, portfolio: boolean): Target {
  const nd = asNd(target);
  if (nd !== null) {
    const [T, K] = [nd.shape[0] ?? 0, nd.shape[1] ?? 1];
    return { data: nd.data, ndim: nd.shape.length, T, K, original: nd.shape.length === 1 ? nd.data : nd };
  }
  if (isColumns(target)) {
    if (portfolio) {
      const { data } = flattenRows(target, "", "target");
      const [, T, K] = dims(target);
      return { data, ndim: 2, T, K, original: { data, shape: [T, K] } };
    }
    const data = fromColumns(target, "target and market inputs");
    const K = target.length, T = target[0].length;
    return { data, ndim: 2, T, K, original: { data, shape: [T, K] } };
  }
  const x = asTyped(target);
  return { data: x, ndim: 1, T: x.length, K: 1, original: x };
}

// walkForward: K candidate columns, a (T,) or (T, K) NdArray, or one series;
// always (T, K) for the driver. A portfolio engine takes a [T, K, C] or
// [T, K] NdArray, or T rows of K values, flattened to (T, K * C).
function candidates(targets: any, portfolio: boolean): Target & { C: number } {
  const nd = asNd(targets);
  if (portfolio) {
    if (nd !== null && (nd.shape.length === 2 || nd.shape.length === 3)) {
      const [T, K] = nd.shape, C = nd.shape.length === 3 ? nd.shape[2] : 1;
      return { data: nd.data, ndim: 2, T, K: K * C, original: nd, C };
    }
    if (isColumns(targets)) {
      const { data, T, K } = flattenRows(targets, "walkForward", "targets");
      return { data, ndim: 2, T, K, original: { data, shape: [T, K] }, C: 1 };
    }
    throw new RangeError("walkForward: a portfolio engine needs (T, K) or (T, K, C) targets");
  }
  if (nd !== null) {
    const T = nd.shape[0] ?? 0, K = nd.shape.length === 2 ? nd.shape[1] : 1;
    return { data: nd.data, ndim: nd.shape.length === 1 ? 2 : nd.shape.length, T, K,
             original: nd.shape.length === 2 ? nd : { data: nd.data, shape: [T, K] }, C: 1 };
  }
  const data = isColumns(targets) ? fromColumns(targets, "targets and price inputs")
    : Float64Array.from(targets as ArrayLike<number>);
  const K = isColumns(targets) ? targets.length : 1, T = data.length / Math.max(1, K);
  return { data, ndim: 2, T, K, original: { data, shape: [T, K] }, C: 1 };
}

// A custom metric's result as [first value, values given, kind] at out[at],
// for the C++ check (check_metric_result in include/screamer/validation/
// scoring.h; kinds as its MetricKind): 0 a number, 1 a 1-D array of numbers
// (a typed array of numbers or a flat Array of numbers), 2 anything else, such
// as a bool, a string, or a nested array, which C++ rejects as TypeError.
function metricEntry(v: unknown, out: Float64Array, at: number): void {
  let first = NaN, given = 0, kind = 2;
  if (typeof v === "number") {
    first = v; given = 1; kind = 0;
  } else if ((ArrayBuffer.isView(v) && !(v instanceof DataView) && !(v instanceof BigInt64Array) &&
              !(v instanceof BigUint64Array)) ||
             (Array.isArray(v) && v.every((x) => typeof x === "number"))) {
    const a = v as ArrayLike<number>;
    given = a.length; kind = 1;
    if (given > 0) first = a[0];
  }
  out[at] = first; out[at + 1] = given; out[at + 2] = kind;
}

// The result arrays of a randomization test (the C++ ResultSink): begin
// allocates n * K values per statistic and for the null, once C++ has checked
// n, and emit copies each batch in, so the WASM heap holds nothing sized by n.
function resultSink(M: any, names: StatName[]) {
  const sink = { n: 0, K: 1, count: 0, stats: [] as Float64Array[], null: new Float64Array(0) };
  const fns = {
    begin: (runs: number, k: number) => {
      sink.n = runs; sink.K = k;
      sink.stats = names.map(() => new Float64Array(runs * k));
      sink.null = new Float64Array(runs * k);
    },
    emit: (j0: number, runs: number, statsPtr: number, stride: number, metricsPtr: number, c: number) => {
      for (let q = 0; q < names.length; q++)
        sink.stats[q].set(M.viewF64(statsPtr + 8 * q * stride, runs * sink.K), j0 * sink.K);
      sink.null.set(M.viewF64(metricsPtr, c), sink.count);
      sink.count += c;
    },
  };
  // The finite null of one strategy: the first `count` values.
  const finiteNull = (count: number) => (count === sink.null.length ? sink.null : sink.null.slice(0, count));
  return { sink, fns, finiteNull };
}

// The engine side of positionTest and walkForward for the C++ drivers
// (JsEngineHost in wasm/validation_bindings.cpp). `state.out` is the engine's
// last output and `state.target` the last target it got.
function engineCallbacks(engine: ScreamerOp, original: any, market: any[], metric: unknown) {
  const M = current();
  const state = { out: null as any, last: null as NdArray | null, target: null as any };
  const fns = {
    run: (isOriginal: boolean, T: number, cols: number, ptr: number, hdr: number) => {
      const target = isOriginal ? original : cols === 0 ? new Float64Array(M.viewF64(ptr, T))
        : { data: new Float64Array(M.viewF64(ptr, T * cols)), shape: [T, cols] };
      engine.reset();
      state.out = engine(target, ...market);
      state.last = asNd(state.out);
      state.target = target;
      const shape = state.last ? state.last.shape : [];
      M.viewF64(hdr, 4).set([Math.min(shape.length, 4), shape[0] ?? 0, shape[1] ?? 0, shape[2] ?? 0]);
    },
    fetch: (outPtr: number, count: number) => {
      M.viewF64(outPtr, count).set((state.last as NdArray).data.subarray(0, count));
    },
    reset: () => { engine.reset(); },
    // Each of `count` strategies of `what`: 0 the last output, 1 the output
    // as one book, 2 the (rows, count, 4) window at windowPtr, 3 one
    // candidate's (rows, width, 4) book window at windowPtr. Writes
    // metricEntry's triple per strategy to resPtr, once at the end, up to the
    // first result of the wrong kind or count.
    metric: (what: number, count: number, rows: number, width: number, windowPtr: number,
             resPtr: number) => {
      const f = metric as (out: NdArray) => unknown;
      const last = state.last as NdArray;
      const res = new Float64Array(3 * count);
      for (let j = 0; j < count; j++) {
        const arg = what === 3
          ? { data: M.viewF64(windowPtr, rows * width * 4).slice(), shape: [rows, width, 4] }
          : what === 2     // a fresh view each time: the metric may grow the heap
          ? strategyOf({ data: M.viewF64(windowPtr, rows * count * 4), shape: [rows, count, 4] }, j)
          : what === 1 || last.shape.length === 2 ? last : strategyOf(last, j);
        metricEntry(f(arg), res, 3 * j);
        // The first result C++ will reject ends the batch, so its error is
        // raised before the metric runs again (later entries are not read).
        if (res[3 * j + 2] === 2 || res[3 * j + 1] !== 1) break;
      }
      M.viewF64(resPtr, 3 * count).set(res);
    },
  };
  return { fns, state };
}

// Row-major data of a market input: a series, an NdArray, or T rows of K values.
function marketData(x: any): Float64Array {
  const nd = asNd(x);
  if (nd !== null) return nd.data;
  if (dims(x)[0] === 2) return flattenRows(x, "walkForward", "market").data;
  return Float64Array.from(x as ArrayLike<number>);
}

// Heap copies of the target and of the [address, ndim, T, K] specs of the
// target and the market inputs. The market's data stays in JS except for
// input `withData` (the carry reads the mark price in C++; -1: none).
function heapSpecs(target: Target, market: any[], bufs: number[], withData = -1) {
  const data = heapCopy(target.data);
  bufs.push(data);
  const t = heapCopy([data, target.ndim, target.T, target.K]);
  bufs.push(t);
  const m = heapCopy(market.flatMap((x, i) => {
    let addr = 0;
    if (i === withData) { addr = heapCopy(marketData(x)); bufs.push(addr); }
    return [addr, ...dims(x)];
  }));
  bufs.push(m);
  return { target: t, market: m };
}

/**
 * Test whether a strategy's timing beats randomly timed copies of itself.
 *
 * `engine` is a target-position backtest engine, `target` the strategy's
 * target series, and `market` the engine's market inputs, exactly as in
 * `engine(target, ...market)`. With `method: "shift"` (default) each surrogate
 * rotates the whole target series; with `"segments"` it shuffles the order of
 * the runs. The null hypothesis is that the timing carries no information
 * about future returns, given this trading profile.
 *
 * `metric` is the name of a `backtestReport` summary statistic (`"sharpe"`,
 * per step, by default) or a function of one run's (T, 4) engine output.
 * Statistics with no natural direction, such as `"turnover"`, need
 * `largerIsBetter`. The p-value counts the observed result itself, so it is
 * never zero. `observedStats` and `nullStats` hold every statistic of the real
 * and the randomized runs, for plotting distributions and scatter plots. Mirrors Python's
 * `screamer.position_test`; the same seed gives the same null distribution.
 *
 * Pass an array of K target columns, or a (T, K) `NdArray` (the result is then
 * a `PositionFamilyResult`), to test several strategies jointly (method
 * "shift"): every round shifts all of them by the same offset, and
 * `familyPValue` compares each strategy with the best of the K per round.
 *
 * With a portfolio engine (`BacktestPortfolioTarget`,
 * `BacktestPortfolioOHLCTarget`), `target` is the (T, K) target of one book and
 * `market` its (T, K) inputs. Every round shifts all K assets by one offset and
 * scores the whole book, so the result is a single test.
 */
/** An `NdArray` of rank 1, (T,): a series. */
export type NdSeries = NdArray & { shape: [number] };
/** An `NdArray` of rank 2, (T, K): K columns or a panel of K assets. */
export type NdPanel = NdArray & { shape: [number, number] };
/** A market input of one strategy's engine: a series. */
export type SeriesInput = ArrayLike<number> | NdSeries;
/** A market input of a portfolio engine: a (T, K) panel, as T rows or an `NdArray`. */
export type PanelInput = ArrayLike<number>[] | NdPanel;

// Overloads. With series markets, a (T,) target is one strategy and a (T, K)
// target (K columns, or an NdArray of shape [T, K]) is K strategies. With
// panel markets the engine is a portfolio engine and the result is one test.
// A (T,) target is always one strategy. Where the rank is not in the types
// (an NdArray whose shape is number[]), the result is the union.
export function positionTest(
  engine: ScreamerOp, target: NdSeries, market: (SeriesInput | NdArray)[], opts?: PositionTestOptions,
): PositionTestResult;
export function positionTest(
  engine: ScreamerOp, target: NdPanel, market: SeriesInput[], opts?: PositionTestOptions,
): PositionFamilyResult;
export function positionTest(
  engine: ScreamerOp, target: ArrayLike<number>[], market: SeriesInput[], opts?: PositionTestOptions,
): PositionFamilyResult;
export function positionTest(
  engine: ScreamerOp, target: ArrayLike<number>, market: (SeriesInput | NdArray)[], opts?: PositionTestOptions,
): PositionTestResult;
export function positionTest(
  engine: ScreamerOp, target: NdPanel | ArrayLike<number>[], market: PanelInput[], opts?: PositionTestOptions,
): PositionTestResult;
export function positionTest(
  engine: ScreamerOp, target: NdArray | ArrayLike<number>[],
  market: (SeriesInput | PanelInput | NdArray)[], opts?: PositionTestOptions,
): PositionTestResult | PositionFamilyResult;
export function positionTest(
  engine: ScreamerOp,
  target: ArrayLike<number> | ArrayLike<number>[] | NdArray,
  market: any[],
  opts: PositionTestOptions = {},
): PositionTestResult | PositionFamilyResult {
  noBools("positionTest", { n: [opts.n, true], minShift: [opts.minShift, true], seed: [opts.seed, true],
                           periodsPerYear: [opts.periodsPerYear, false] });
  const M = current();
  const portfolio = !!(engine as any).rowOp;
  const tgt = positionTarget(target, portfolio);
  texts("positionTest", { method: opts.method });
  const method = opts.method ?? "shift";
  const metric = opts.metric ?? "sharpe";
  const { custom, name, hasLarger, larger } = metricArgs(metric, opts.largerIsBetter);
  const names = statNames();
  const { fns } = engineCallbacks(engine, tgt.original, market, metric);
  // The results arrive in batches (begin, emit) and the rest in done.
  const results = resultSink(M, names);
  let result: PositionTestResult | PositionFamilyResult | null = null;
  const callbacks = wasmCallbacks({
    ...fns,
    ...results.fns,
    // head = [observed (k), p (k), family p (k), observed stats (15 x k)]
    done: (headPtr: number, k: number, family: boolean, minShift: number, nullCount: number) => {
      const head = M.viewF64(headPtr, k * (3 + names.length)).slice();
      const part = (i: number) => head.slice(i * k, (i + 1) * k);
      const { n, stats } = results.sink;
      result = family
        ? { observed: part(0), null: { data: results.sink.null, shape: [n, k] }, pValue: part(1),
            familyPValue: part(2), metric: name, method: "shift", minShift,
            observedStats: statsRecord((i) => part(3 + i)),
            nullStats: statsRecord((i): NdArray => ({ data: stats[i], shape: [n, k] })) }
        : { observed: head[0], null: results.finiteNull(nullCount), pValue: head[1], metric: name,
            method: method as "shift" | "segments", minShift: minShift < 0 ? null : minShift,
            observedStats: statsRecord((i) => head[3 + i]), nullStats: statsRecord((i) => stats[i]) };
    },
  });
  const bufs: number[] = [];
  try {
    const spec = heapSpecs(tgt, market, bufs);
    // null and undefined both mean "not given", as None does in Python.
    callbacks.call((cb) => M.positionTestRun(cb, spec.target, spec.market, market.length,
      count(opts.n ?? 1000), String(method), custom ? "" : name, custom, hasLarger, larger,
      opts.minShift != null, int(opts.minShift ?? 0, 0), num(opts.seed ?? 0),
      opts.periodsPerYear ?? 1, portfolio));
    return result as unknown as PositionTestResult | PositionFamilyResult;
  } finally {
    bufs.forEach((p) => M.freeBuf(p));
  }
}

// Integers go to C++ as they are; a non-integer becomes `bad`, which the
// driver rejects with its usual message.
const int = (v: number, bad: number) => (Number.isInteger(v) ? v : bad);

// A run count: also +-Infinity as it is, which C++ clamps and rejects as too
// large (or too small).
const count = (v: number) => (v === Infinity || v === -Infinity ? v : int(v, 0));

// ---- data permutations and strategyTest (Python: permute_bars,
// permute_returns, strategy_test) ------------------------------------------

// Heap blocks of the given sizes (in doubles), read back as copies.
function heapBlocks(sizes: number[]) {
  const M = current();
  const ptrs: number[] = [];
  const free = () => ptrs.forEach((p) => M.freeBuf(p));
  try {
    for (const n of sizes) ptrs.push(M.allocF64(n));
  } catch (e) {
    free();
    throw e;
  }
  const read = (i: number) => new Float64Array(M.viewF64(ptrs[i], sizes[i]));
  return { ptrs, read, free };
}

function heapCopy(x: ArrayLike<number>): number {
  const M = current();
  const p = M.allocF64(x.length);
  M.viewF64(p, x.length).set(x as number[]);
  return p;
}

export type PermuteMode = "log" | "diff" | "level";
export type VolumeMode = "keep" | "notional";

function checkModes(mode: string, volumeMode: string): void {
  if (mode !== "log" && mode !== "diff" && mode !== "level")
    throw new RangeError('mode must be "log", "diff", or "level"');
  if (volumeMode !== "keep" && volumeMode !== "notional")
    throw new RangeError('volume_mode must be "keep" or "notional"');
}

// Inputs to the permutation kernels are row-major (T x K) blocks of K assets;
// K = 1 is one series. The C++ kernels check the values (finite, positive,
// warmup, K >= 1) and their errors arrive as RangeError through wasmCall. The
// kernels get raw pointers, so the buffer lengths are checked here.
function checkLengths(xs: ArrayLike<number>[], T: number, K: number, what: string): void {
  if (xs.some((x) => x.length !== xs[0].length))
    throw new RangeError(`${what} must have the same length.`);
  if (xs[0].length !== T * K)
    throw new RangeError(`${what} must hold T * K = ${T * K} values, got ${xs[0].length}.`);
}

// Outputs: open, high, low, close (T x ncols x K), source (T x ncols), and the
// moved volume (T x ncols x K) when `vol` is given.
function barsBlock(o: ArrayLike<number>, h: ArrayLike<number>, l: ArrayLike<number>,
                   c: ArrayLike<number>, T: number, K: number, mode: PermuteMode, seed: number,
                   col0: number, ncols: number, vol: ArrayLike<number> | null, notional: boolean) {
  const M = current();
  checkLengths([o, h, l, c], T, K, "open, high, low, and close");
  if (vol !== null && vol.length !== o.length)
    throw new RangeError("volume must have the same length as the prices.");
  const ins: number[] = [];
  let outs: ReturnType<typeof heapBlocks> | null = null;
  try {
    for (const a of [o, h, l, c]) ins.push(heapCopy(a));
    const vp = vol === null ? 0 : heapCopy(vol);
    ins.push(vp);
    const N = T * ncols * K;
    outs = heapBlocks(vol === null ? [N, N, N, N, T * ncols] : [N, N, N, N, T * ncols, N]);
    const p = outs.ptrs;
    wasmCall(() => M.permuteBarsInto(ins[0], ins[1], ins[2], ins[3], T, K, mode, seed, col0,
                                     ncols, p[0], p[1], p[2], p[3], p[4], vp, notional,
                                     vol === null ? 0 : p[5]));
    return p.map((_, k) => (outs as ReturnType<typeof heapBlocks>).read(k));
  } finally {
    ins.forEach((p) => M.freeBuf(p)); outs?.free();
  }
}

// Outputs: prices (T x ncols x K), source (T x ncols), and the moved volume
// (T x ncols x K) when `vol` is given.
function returnsBlock(x: ArrayLike<number>, T: number, K: number, mode: PermuteMode,
                      seed: number, col0: number, ncols: number,
                      vol: ArrayLike<number> | null, notional: boolean) {
  const M = current();
  checkLengths([x], T, K, "x");
  if (vol !== null && vol.length !== x.length)
    throw new RangeError("volume must have the same length as the prices.");
  let xp = 0, vp = 0;
  let outs: ReturnType<typeof heapBlocks> | null = null;
  try {
    xp = heapCopy(x);
    if (vol !== null) vp = heapCopy(vol);
    const N = T * ncols * K;
    outs = heapBlocks(vol === null ? [N, T * ncols] : [N, T * ncols, N]);
    const p = outs.ptrs;
    wasmCall(() => M.permuteReturnsInto(xp, T, K, mode, seed, col0, ncols, p[0], p[1],
                                        vp, notional, vol === null ? 0 : p[2]));
    return p.map((_, k) => (outs as ReturnType<typeof heapBlocks>).read(k));
  } finally {
    M.freeBuf(xp); M.freeBuf(vp); outs?.free();
  }
}

// The public permute functions take one series, as in Python: an NdArray of
// rank 1 is read as its data, and a panel (an NdArray of rank 2 or more, or an
// array of rows) throws the RangeError Python raises as ValueError.
function series(x: any, name: string): ArrayLike<number> {
  const nd = asNd(x);
  if (nd !== null) {
    if (nd.shape.length !== 1) throw new RangeError(`${name} must be a 1-D series`);
    return nd.data;
  }
  if (x == null || typeof x === "string" || typeof x.length !== "number")
    throw new TypeError(`${name} must be an array of numbers or an NdArray`);
  const first = x.length > 0 ? x[0] : undefined;
  if (first != null && typeof first === "object") throw new RangeError(`${name} must be a 1-D series`);
  return x;
}

export interface PermuteOptions {
  mode?: PermuteMode;
  seed?: number;
  volume?: ArrayLike<number>;
  volumeMode?: VolumeMode;
}

/**
 * `n` permuted copies of an OHLC series: `[open, high, low, close]` (and the
 * moved volume when `volume` is given), each a (T, n) array. `mode` is "log"
 * (shuffle log gaps and intrabar shapes, Masters 2020; the default), "diff"
 * (the same with differences, for prices that can be zero or negative), or
 * "level" (shuffle whole bars). `volumeMode` "keep" moves volume with its bar;
 * "notional" also rescales it so volume times the typical price stays that of
 * its source bar. Mirrors Python's `permute_bars`.
 */
export function permuteBars(
  open: ArrayLike<number> | NdArray, high: ArrayLike<number> | NdArray,
  low: ArrayLike<number> | NdArray, close: ArrayLike<number> | NdArray, n: number,
  opts: PermuteOptions = {},
): NdArray[] {
  noBools("permuteBars", { n: [n, true], seed: [opts.seed, true] });
  const seed = num(opts.seed ?? 0), mode = opts.mode ?? "log", vm = opts.volumeMode ?? "keep";
  checkModes(mode, vm);
  const o = series(open, "open"), h = series(high, "high"), l = series(low, "low");
  const c = series(close, "close");
  const vol = opts.volume === undefined ? null : series(opts.volume, "volume");
  const out = barsBlock(o, h, l, c, o.length, 1, mode, seed, 0, n, vol, vm === "notional");
  const keep = vol === null ? [0, 1, 2, 3] : [0, 1, 2, 3, 5];
  return keep.map((k) => ({ data: out[k], shape: [o.length, n] }));
}

/**
 * `n` permuted copies of a price series, as a (T, n) array; with `volume`,
 * `[prices, volume]`. `mode` is "log" (shuffle log returns; the default),
 * "diff" (shuffle differences), or "level" (shuffle the values). Mirrors
 * Python's `permute_returns`.
 */
export function permuteReturns(
  x: ArrayLike<number> | NdArray, n: number, opts: PermuteOptions = {},
): NdArray | NdArray[] {
  noBools("permuteReturns", { n: [n, true], seed: [opts.seed, true] });
  const seed = num(opts.seed ?? 0), mode = opts.mode ?? "log", vm = opts.volumeMode ?? "keep";
  checkModes(mode, vm);
  const xs = series(x, "x");
  const vol = opts.volume === undefined ? null : series(opts.volume, "volume");
  const out = returnsBlock(xs, xs.length, 1, mode, seed, 0, n, vol, vm === "notional");
  const nd = (d: Float64Array): NdArray => ({ data: d, shape: [xs.length, n] });
  return vol === null ? nd(out[0]) : [nd(out[0]), nd(out[2])];
}

export interface StrategyTestOptions {
  n?: number;
  method?: "bars" | "returns" | "shift";
  ohlc?: [number, number, number, number];
  price?: number;
  mode?: PermuteMode;
  volume?: number;
  volumeMode?: VolumeMode;
  columns?: number[];
  fixed?: number[];
  metric?: PositionTestMetric;
  largerIsBetter?: boolean;
  minShift?: number;
  seed?: number;
  /** Annualizes the ratios, as in `backtestReport` (default 1: per step). */
  periodsPerYear?: number;
}

export interface StrategyTestResult {
  observed: number;
  null: Float64Array;
  pValue: number;
  metric: string;
  method: "bars" | "returns" | "shift";
  /** Every statistic of the real run. */
  observedStats: Record<StatName, number>;
  /** Every statistic per randomized run, in run order with NaN kept. */
  nullStats: Record<StatName, Float64Array>;
}

/** One strategyTest input: a series of T values, or a (T, K) panel as an
 * NdArray or as T rows of K values. */
export type StrategyInput = ArrayLike<number> | ArrayLike<number>[] | NdArray;

// A strategyTest input as row-major data with its time length T and asset
// count K; `panel` is false for a 1-D series.
interface Block { data: Float64Array; T: number; K: number; panel: boolean }

function toBlock(x: any, i: number): Block {
  const where = `input ${i}`;
  const nd = asNd(x);
  if (nd !== null) {
    if (nd.shape.length === 1) return { data: nd.data, T: nd.shape[0], K: 1, panel: false };
    if (nd.shape.length === 2)
      return { data: nd.data, T: nd.shape[0], K: nd.shape[1], panel: true };
    throw new RangeError(`${where} must be 1-D (T,) or 2-D (T, K), got ${nd.shape.length}-D`);
  }
  if (x == null || typeof x === "string" || typeof x.length !== "number")
    throw new TypeError(`${where} must be an array of numbers, an array of rows, or an NdArray`);
  const T = x.length;
  const first = T > 0 ? x[0] : undefined;
  if (first != null && typeof first === "object" && typeof first.length === "number") {
    const { data, K } = flattenRows(x, "", where);
    return { data, T, K, panel: true };
  }
  return { data: Float64Array.from(x as ArrayLike<number>), T, K: 1, panel: false };
}

// What the strategy receives: a Float64Array for a series, an NdArray for a panel.
const asArg = (b: Block, data: Float64Array): Float64Array | NdArray =>
  b.panel ? { data, shape: [b.T, b.K] } : data;

/**
 * Test a whole strategy against the same strategy run on randomized data.
 * `strategy(...inputs)` must return a backtest engine output of shape (T, 4)
 * or a portfolio output of shape (T, K, 4) (or a pipeline `{values, index}`
 * holding one). `method` is `"bars"` (permute the OHLC inputs at `ohlc`),
 * `"returns"` (permute the price input at `price`), or `"shift"` (rotate the
 * feature inputs at `columns` by one random offset). With bars and returns,
 * `mode` is "log", "diff", or "level", and other inputs move with their bar
 * unless listed in `fixed`; the input at `volume` is moved and, with
 * `volumeMode: "notional"`, rescaled to keep the traded amount.
 *
 * Each input is a series of T values or a (T, K) panel of K assets (an
 * NdArray or T rows of K values); a panel reaches the strategy as an NdArray.
 * Each randomized copy applies one time permutation to every asset, so the
 * correlation between assets is kept. Mirrors Python's `strategy_test`; the
 * same seed gives the same randomized inputs.
 */
export function strategyTest(
  strategy: (...inputs: any[]) => any,
  inputs: StrategyInput[],
  opts: StrategyTestOptions = {},
): StrategyTestResult {
  noBools("strategyTest", { n: [opts.n, true], ohlc: [opts.ohlc, true], price: [opts.price, true],
                           volume: [opts.volume, true], minShift: [opts.minShift, true],
                           seed: [opts.seed, true], periodsPerYear: [opts.periodsPerYear, false] });
  // The text settings go to C++ as strings: another kind is a TypeError here.
  texts("strategyTest", { method: opts.method, mode: opts.mode, volumeMode: opts.volumeMode });
  const n = opts.n ?? 500;
  const method = opts.method ?? "bars";
  const metric = opts.metric ?? "sharpe";
  const { custom, name, hasLarger, larger } = metricArgs(metric, opts.largerIsBetter);
  const blocks = inputs.map((x, i) => toBlock(x, i));
  const M = current();
  const k = blocks.length;
  // Positions go to C++ as integers; a non-integer becomes -1, which the
  // driver rejects with its usual message.
  const pos = (xs: readonly number[]) => xs.map((i) => int(i, -1));

  // The callbacks C++ makes (wasm/validation_bindings.cpp); wasmCallbacks
  // keeps the user's error and rethrows it once the driver has unwound.
  let last: NdArray | null = null;
  let result: StrategyTestResult | null = null;
  const names = statNames();
  const results = resultSink(M, names);
  const entry = new Float64Array(3);
  const callbacks = wasmCallbacks({
    run: (argsPtr: number, hdrPtr: number) => {
      const ptrs = M.viewF64(argsPtr, k);
      const args = blocks.map((b, i) => asArg(b, ptrs[i] === 0 ? b.data
        : new Float64Array(M.viewF64(ptrs[i], b.T * b.K))));
      const out = strategy(...args);
      last = asNd(out && out.values !== undefined ? out.values : out);   // Pipeline Output
      const shape = last ? last.shape : [];
      M.viewF64(hdrPtr, 4).set([shape.length, shape[0] ?? 0, shape[1] ?? 0, shape[2] ?? 0]);
    },
    fetch: (outPtr: number, count: number) => {
      M.viewF64(outPtr, count).set((last as NdArray).data.subarray(0, count));
    },
    metric: (hdrPtr: number) => {
      metricEntry((metric as (out: NdArray) => unknown)(last as NdArray), entry, 0);
      M.viewF64(hdrPtr, 3).set(entry);
    },
    ...results.fns,
    // [observed, p_value, observed stats (15)]
    done: (headPtr: number, nullCount: number) => {
      const head = M.viewF64(headPtr, 2 + names.length).slice();
      result = {
        observed: head[0], null: results.finiteNull(nullCount), pValue: head[1],
        metric: name, method, observedStats: statsRecord((i) => head[2 + i]),
        nullStats: statsRecord((i) => results.sink.stats[i]),
      };
    },
  });

  const bufs: number[] = [];
  try {
    const spec = new Float64Array(4 * k);
    blocks.forEach((b, i) => {
      bufs.push(heapCopy(b.data));
      spec.set([bufs[i], b.T, b.K, b.panel ? 1 : 0], 4 * i);
    });
    const specPtr = heapCopy(spec);
    bufs.push(specPtr);
    // null and undefined both mean "not given", as None does in Python.
    callbacks.call((cb) => M.strategyTestRun(
      cb, specPtr, k, method, count(n), pos(opts.ohlc ?? [0, 1, 2, 3]), int(opts.price ?? 0, -1),
      opts.mode ?? "log", opts.volume != null, int(opts.volume ?? 0, -1),
      opts.volumeMode ?? "keep", pos(opts.columns ?? []), pos(opts.fixed ?? []),
      custom ? "" : name, custom, hasLarger, larger, opts.minShift != null,
      int(opts.minShift ?? 0, 0), num(opts.seed ?? 0), opts.periodsPerYear ?? 1));
    return result as unknown as StrategyTestResult;
  } finally {
    bufs.forEach((p) => M.freeBuf(p));
  }
}

// ---- probabilistic and deflated Sharpe ratios (Python: probabilistic_sharpe,
// deflated_sharpe) ---------------------------------------------------------

function pnlOf(pnl: ArrayLike<number> | NdArray): Float64Array {
  const nd = asNd(pnl);
  if (nd && nd.shape.length === 2 && nd.shape[1] === 4) return column(nd, 1);
  return Float64Array.from(pnl as ArrayLike<number>);
}

function checkPnl(x: Float64Array): void {
  const f = Array.from(x).filter(Number.isFinite);
  if (f.length < 3) throw new RangeError("need at least three finite pnl values.");
  if (f.every((v) => v === f[0])) throw new RangeError("pnl has no dispersion.");
}

/**
 * Probability that the true Sharpe ratio is above `benchmark` (per step),
 * adjusting the observed per-step Sharpe ratio for sample length, skewness,
 * and kurtosis (Bailey and Lopez de Prado 2012). `pnl` is a per-step pnl
 * series or a backtest engine's (T, 4) output. Mirrors Python's
 * `probabilistic_sharpe`.
 */
export function probabilisticSharpe(pnl: ArrayLike<number> | NdArray, benchmark = 0): number {
  noBools("probabilisticSharpe", { benchmark: [benchmark, false] });
  const M = current();
  const x = pnlOf(pnl);
  checkPnl(x);
  const p = heapCopy(x);
  try { return wasmCall(() => M.probabilisticSharpe(p, x.length, benchmark)); } finally { M.freeBuf(p); }
}

/**
 * Probabilistic Sharpe ratio against the Sharpe ratio expected from the best of
 * the tried strategies (Bailey and Lopez de Prado 2014). `trialSharpes` holds
 * the per-step Sharpe ratio of every variant tried, this one included.
 * Mirrors Python's `deflated_sharpe`.
 */
export function deflatedSharpe(
  pnl: ArrayLike<number> | NdArray, trialSharpes: ArrayLike<number>,
): { probability: number; benchmark: number } {
  const M = current();
  const x = pnlOf(pnl);
  checkPnl(x);
  if (Array.from(trialSharpes).filter(Number.isFinite).length < 2)
    throw new RangeError("deflated_sharpe needs the Sharpe ratios of at least two trials.");
  let p = 0, t = 0, out = 0;
  try {
    p = heapCopy(x); t = heapCopy(trialSharpes); out = M.allocF64(2);
    wasmCall(() => M.deflatedSharpeInto(p, x.length, t, trialSharpes.length, out));
    const v = M.viewF64(out, 2);
    return { probability: v[0], benchmark: v[1] };
  } finally {
    M.freeBuf(p); M.freeBuf(t); M.freeBuf(out);
  }
}

// ---- walk-forward selection (Python: walk_forward) ----------------------

export interface WalkForwardOptions {
  /** Length of each training window (at least 2, shorter than the series). */
  train: number;
  /** Length of each test window (at least 1). */
  test: number;
  metric?: PositionTestMetric;
  /** Train on everything before the test window instead of the last `train` steps. */
  anchored?: boolean;
  /** Annualizes the ratios, as in `backtestReport` (default 1: per step). */
  periodsPerYear?: number;
  largerIsBetter?: boolean;
  /**
   * Per-step carry rate (funding, borrow), one value per row, with the sign
   * and timing `BacktestCarry` documents. Every run is scored after
   * `BacktestCarry`, marked with the engine's `markInput`. A `[T, K]`
   * NdArray, or T rows of K values, for a portfolio engine.
   */
  carry?: ArrayLike<number> | ArrayLike<number>[] | NdArray;
  /** The carry's contract multiplier (default 1). */
  carryMultiplier?: number;
}

export interface WalkForwardResult {
  /**
   * The stitched out-of-sample target: 0 before the first test window;
   * `[T, K]` for a portfolio engine.
   */
  target: Float64Array | NdArray;
  /** The chosen candidate per fold; -1 when every score was NaN (flat). */
  choice: Int32Array;
  /** The first step of each test window. */
  foldStart: Int32Array;
  /** Every candidate's metric on every training window, (nFolds, C) row-major. */
  trainScores: NdArray;
  /**
   * The engine output on `target`, (T, 4), or (T, K, 4) for a portfolio
   * engine, after carry when `carry` is given.
   */
  output: NdArray;
  /** Every statistic of `output` from the first test window to the end. */
  stats: Record<StatName, number>;
  metric: string;
}

/**
 * Walk-forward selection: pick the best candidate on each training window and
 * trade it on the next test window.
 *
 * `targets` holds one target series per candidate (an array of K columns, a
 * (T, K) `NdArray`, or one series), and `market` the engine's inputs, as in
 * `engine(target, ...market)`. Folds start at `t0 = train, train + test, ...`
 * while `t0 < T`; fold i tests on `[t0, min(t0 + test, T))` and trains on
 * `[t0 - train, t0)`, or on `[0, t0)` with `anchored`. A window is scored as a
 * backtest of its own. NaN scores lose, ties go to the lowest column, and a
 * fold where every score is NaN stays flat with choice -1. The stitched target
 * (0 before the first test window) runs through the engine once, and `stats`
 * covers it from the first test window on.
 *
 * `carry` is a per-step carry rate (funding, borrow), one value per row, with
 * the sign and timing `BacktestCarry` documents. With it, every candidate's
 * run and the stitched run pass through `BacktestCarry` (multiplier
 * `carryMultiplier`) before they are scored, marked with the engine's
 * `markInput` (the price for `BacktestPriceTarget` and
 * `BacktestPortfolioTarget`, the close for the OHLC engines, the trade price
 * for `BacktestTradesTarget`, the ask for `BacktestL1Target`), so `output`
 * and `stats` are after carry. An engine without `markInput` raises with
 * `carry`.
 *
 * Portfolio engines (`BacktestPortfolioTarget`, `BacktestPortfolioOHLCTarget`)
 * take a `[T, K, C]` target, C candidates for K assets (`[T, K]` is one
 * candidate), and `[T, K]` prices. Each window is scored on the book, the pnl
 * summed across assets; `target` is `[T, K]`, `output` `[T, K, 4]`, and a
 * callable metric gets one candidate's `[L, K, 4]` window and returns one
 * number. `carry` is then `[T, K]`. A portfolio engine reads a NaN target as
 * "hold". When a test window changes the book (the first one, a switch of
 * candidate, or the first after a flat fold), a NaN on its first row is
 * restated as the new candidate's last finite target for that asset (0 when
 * it has none), so the window starts from that candidate's book; when the
 * same candidate continues, the held book carries over without a trade.
 * Mirrors Python's `screamer.walk_forward`.
 */
export function walkForward(
  engine: ScreamerOp,
  targets: ArrayLike<number> | ArrayLike<number>[] | NdArray,
  market: (ArrayLike<number> | NdArray)[],
  opts: WalkForwardOptions,
): WalkForwardResult {
  noBools("walkForward", { train: [opts.train, true], test: [opts.test, true],
                          periodsPerYear: [opts.periodsPerYear, false],
                          carryMultiplier: [opts.carryMultiplier, false] });
  const M = current();
  const metric = opts.metric ?? "sharpe";
  const { custom, name, hasLarger, larger } = metricArgs(metric, opts.largerIsBetter);
  const portfolio = !!(engine as any).rowOp;
  const tgt = candidates(targets, portfolio);
  const { fns, state } = engineCallbacks(engine, tgt.original, market, metric);
  let result: WalkForwardResult | null = null;
  const callbacks = wasmCallbacks({
    ...fns,
    done: (statsPtr: number, foldPtr: number, choicePtr: number, scoresPtr: number, F: number, K: number,
           outPtr: number, outLen: number) => {
      const stats = M.viewF64(statsPtr, statNames().length).slice();
      result = {
        target: state.target, choice: Int32Array.from(M.viewF64(choicePtr, F)),
        foldStart: Int32Array.from(M.viewF64(foldPtr, F)),
        trainScores: { data: M.viewF64(scoresPtr, F * K).slice(), shape: [F, K] },
        output: outLen > 0
          ? { data: M.viewF64(outPtr, outLen).slice(),
              shape: portfolio ? [tgt.T, tgt.K / tgt.C, 4] : [tgt.T, 4] }
          : state.out,
        stats: statsRecord((i) => stats[i]), metric: name,
      };
    },
  });
  const bufs: number[] = [];
  try {
    const markInput: number = opts.carry != null ? (engine as any).markInput ?? -1 : -1;
    const spec = heapSpecs(tgt, market, bufs, markInput);
    let carrySpec = 0;
    if (opts.carry != null) {
      // A series, an NdArray, or T rows of K values (a portfolio's (T, K)).
      const cnd = asNd(opts.carry) ?? (isColumns(opts.carry)
        ? (({ data, T, K }) => ({ data, shape: [T, K] }))(flattenRows(opts.carry, "walkForward", "carry"))
        : null);
      const cdata = cnd !== null ? cnd.data : Float64Array.from(opts.carry as ArrayLike<number>);
      const cp = heapCopy(cdata);
      bufs.push(cp);
      carrySpec = cnd !== null
        ? heapCopy([cp, cnd.shape.length, cnd.shape[0] ?? 0, cnd.shape[1] ?? 1])
        : heapCopy([cp, 1, cdata.length, 1]);
      bufs.push(carrySpec);
    }
    callbacks.call((cb) => M.walkForwardRun(cb, spec.target, spec.market, market.length,
      num(opts.train), num(opts.test), custom ? "" : name, custom, hasLarger, larger, !!opts.anchored,
      opts.periodsPerYear ?? 1, portfolio, tgt.C, carrySpec, opts.carryMultiplier ?? 1, markInput,
      (engine as any).opName ?? ""));
    return result as unknown as WalkForwardResult;
  } finally {
    bufs.forEach((p) => M.freeBuf(p));
  }
}

// ---- time-series folds (Python: time_splits) -----------------------------

export interface TimeSplitsOptions {
  /** Rows in each test block (default `Math.floor(n / (nSplits + 1))`). */
  testSize?: number;
  /**
   * The rows after row t that its label covers: one number for every row
   * (default 0), or n values, such as `TripleBarrier`'s `holding` output.
   */
  labelLength?: number | ArrayLike<number> | NdArray;
  /** Keep at most this many training rows per fold, the most recent ones. */
  maxTrainSize?: number;
}

/**
 * Train and test rows of time-ordered folds, with the training rows whose
 * labels reach into the test block left out.
 *
 * Returns `nSplits` pairs `[train, test]` of row indices. The folds are those
 * of scikit-learn's `TimeSeriesSplit`: the `nSplits` test blocks of `testSize`
 * rows end at row `n`, and each fold trains on the rows before its test
 * block, the most recent `maxTrainSize` of them when it is given. A training
 * row t is left out when `t + labelLength[t]` reaches the first test row,
 * because its label would use test data. A NaN or negative length raises, and
 * so does a fold left with no training rows. Mirrors Python's
 * `screamer.time_splits`.
 */
export function timeSplits(n: number, nSplits = 5, opts: TimeSplitsOptions = {}): [Int32Array, Int32Array][] {
  const who = "timeSplits";
  const ll = opts.labelLength ?? 0;
  noBools(who, { n: [n, true], nSplits: [nSplits, true], testSize: [opts.testSize, true],
                 maxTrainSize: [opts.maxTrainSize, true], labelLength: [ll, false] });
  if (typeof ll === "string") throw new TypeError(`${who}: labelLength must be a number or an array, got string`);
  let label: ArrayLike<number> | null = null;
  if (typeof ll !== "number") {
    const nd = asNd(ll);
    if (nd !== null && nd.shape.length !== 1)
      throw new RangeError(`${who}: labelLength must be a number or a 1-D array of n values`);
    label = nd !== null ? nd.data : Float64Array.from(ll as ArrayLike<number>);
  }
  const M = current();
  let p = 0;
  try {
    if (label !== null) {
      p = M.allocF64(Math.max(1, label.length));
      M.viewF64(p, label.length).set(label as number[]);
    }
    const r = wasmCall(() => M.timeSplits(num(n), num(nSplits), opts.testSize != null, num(opts.testSize),
      opts.maxTrainSize != null, num(opts.maxTrainSize), p, label?.length ?? 0,
      typeof ll === "number" ? ll : 0)) as { offsets: Int32Array; index: Int32Array };
    const folds: [Int32Array, Int32Array][] = [];
    for (let f = 0; f + 2 < r.offsets.length; f += 2)
      folds.push([r.index.subarray(r.offsets[f], r.offsets[f + 1]),
                  r.index.subarray(r.offsets[f + 1], r.offsets[f + 2])]);
    return folds;
  } finally {
    M.freeBuf(p);
  }
}
