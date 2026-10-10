// Randomization tests for backtest significance. Mirrors screamer/validation.py.
//
// `positionTest` keeps the market data and the strategy's trading profile,
// randomizes when the strategy holds its positions, and replays each copy
// through the same backtest engine. `strategyTest` keeps the strategy and
// randomizes its data (permuted bars or returns, or shifted features), re-running
// the whole strategy on each copy. These are batch-only validation tools, not streaming
// operators. The surrogates are built in C++ (validation_bindings.cpp), and the
// backtests and the report are the ordinary C++ engines; this module only
// chains them and counts. The same seed gives the same surrogates as Python.
import { current } from "./index.js";
import type { NdArray } from "./ndarray.js";
import type { ScreamerOp } from "./runtime.js";
import { BacktestReport } from "./generated/ops.js";

// Leading NaN warmup length and the length of the series after it.
function validLength(x: ArrayLike<number>): number {
  let p = 0;
  while (p < x.length && Number.isNaN(x[p])) p++;
  return x.length - p;
}

function checkSeed(seed: number): void {
  if (!Number.isInteger(seed) || seed < 0 || seed > Number.MAX_SAFE_INTEGER)
    throw new RangeError("seed must be an integer in [0, 2^53)");
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
  const xPtr = M.allocF64(T), outPtr = M.allocF64(T * ncols);
  try {
    M.viewF64(xPtr, T).set(x as number[]);
    run(xPtr, outPtr);
    return { data: new Float64Array(M.viewF64(outPtr, T * ncols)), shape: [T, ncols] };
  } finally {
    M.freeBuf(xPtr); M.freeBuf(outPtr);
  }
}

function defaultMinShift(x: ArrayLike<number>): number {
  const M = current();
  const ptr = M.allocF64(x.length);
  try {
    M.viewF64(ptr, x.length).set(x as number[]);
    return M.defaultMinShift(ptr, x.length);
  } finally {
    M.freeBuf(ptr);
  }
}

function shiftBlock(x: ArrayLike<number>, minShift: number, seed: number,
                    col0: number, ncols: number, start = -1): NdArray {
  const L = start < 0 ? validLength(x) : x.length - start;
  if (L < 2)
    throw new RangeError("circular_shift needs at least two values after the NaN warmup.");
  if (2 * minShift > L)
    throw new RangeError("min_shift is too large: it must be at most half the series " +
                         "length after the NaN warmup.");
  return surrogates(x, ncols, (xp, op) =>
    current().circularShiftInto(xp, x.length, minShift, seed, col0, ncols, op, start));
}

function shuffleBlock(x: ArrayLike<number>, seed: number, col0: number,
                      ncols: number): NdArray {
  return surrogates(x, ncols, (xp, op) =>
    current().segmentShuffleInto(xp, x.length, seed, col0, ncols, op));
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
  const seed = opts.seed ?? 0;
  checkSeed(seed);
  let m = opts.minShift;
  if (m === undefined) m = defaultMinShift(x);
  else if (!Number.isInteger(m) || m < 1) throw new RangeError("min_shift must be at least 1");
  return shiftBlock(x, m, seed, 0, n);
}

/**
 * `n` copies of a series with its runs in random order, as a (T, n) array.
 *
 * Every copy holds exactly the same runs (trades and flat stretches) with the
 * same values and lengths, at random times. The NaN warmup stays in place.
 */
export function segmentShuffle(
  x: ArrayLike<number>, n: number, opts: { seed?: number } = {},
): NdArray {
  const seed = opts.seed ?? 0;
  checkSeed(seed);
  return shuffleBlock(x, seed, 0, n);
}

export type PositionTestMetric =
  | "sharpe" | "total_pnl" | "max_drawdown"
  | ((engineOutput: NdArray) => number);

export interface PositionTestOptions {
  n?: number;
  method?: "shift" | "segments";
  metric?: PositionTestMetric;
  largerIsBetter?: boolean;
  minShift?: number;
  seed?: number;
}

export interface PositionTestResult {
  observed: number;
  null: Float64Array;
  pValue: number;
  metric: string;
  method: "shift" | "segments";
  minShift: number | null;
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
}

// Engine output columns: [equity, pnl, position, cost].
// BacktestReport columns: [drawdown, cum_cost, turnover, trades, max_drawdown, sharpe].
const METRICS: Record<string, { report: boolean; col: number; larger: boolean }> = {
  total_pnl: { report: false, col: 0, larger: true },
  sharpe: { report: true, col: 5, larger: true },
  max_drawdown: { report: true, col: 4, larger: true },
};

function lastFinite(a: NdArray, col: number): number {
  const [rows, cols] = a.shape;
  for (let r = rows - 1; r >= 0; r--) {
    const v = a.data[r * cols + col];
    if (Number.isFinite(v)) return v;
  }
  return NaN;
}

function column(a: NdArray, col: number): Float64Array {
  const [rows, cols] = a.shape;
  const out = new Float64Array(rows);
  for (let r = 0; r < rows; r++) out[r] = a.data[r * cols + col];
  return out;
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
 * `metric` is `"sharpe"` (per step, default), `"total_pnl"`, `"max_drawdown"`,
 * or a function of one run's (T, 4) engine output. The p-value counts the
 * observed result itself, so it is never zero. Mirrors Python's
 * `screamer.position_test`; the same seed gives the same null distribution.
 *
 * Pass an array of target columns to test several strategies jointly (method
 * "shift"): every round shifts all of them by the same offset, and
 * `familyPValue` compares each strategy with the best of the K per round.
 */
export function positionTest(
  engine: ScreamerOp,
  target: ArrayLike<number>[],
  market: ArrayLike<number>[],
  opts?: PositionTestOptions,
): PositionFamilyResult;
export function positionTest(
  engine: ScreamerOp,
  target: ArrayLike<number>,
  market: ArrayLike<number>[],
  opts?: PositionTestOptions,
): PositionTestResult;
export function positionTest(
  engine: ScreamerOp,
  target: ArrayLike<number> | ArrayLike<number>[],
  market: ArrayLike<number>[],
  opts: PositionTestOptions = {},
): PositionTestResult | PositionFamilyResult {
  if (Array.isArray(target) && target.length > 0 && typeof target[0] !== "number")
    return positionFamily(engine, target as ArrayLike<number>[], market, opts);
  return positionSingle(engine, target as ArrayLike<number>, market, opts);
}

function positionSingle(
  engine: ScreamerOp,
  target: ArrayLike<number>,
  market: ArrayLike<number>[],
  opts: PositionTestOptions,
): PositionTestResult {
  const n = opts.n ?? 1000;
  const method = opts.method ?? "shift";
  const metric = opts.metric ?? "sharpe";
  const seed = opts.seed ?? 0;
  if (method !== "shift" && method !== "segments")
    throw new RangeError('method must be "shift" or "segments"');
  if (!Number.isInteger(n) || n < 1) throw new RangeError("n must be at least 1");
  checkSeed(seed);
  let name: string, larger: boolean;
  if (typeof metric === "function") {
    name = metric.name || "custom";
    larger = opts.largerIsBetter ?? true;
  } else {
    if (!(metric in METRICS))
      throw new RangeError(`metric must be one of ${Object.keys(METRICS).sort().join(", ")} or a function`);
    name = metric;
    larger = opts.largerIsBetter ?? METRICS[metric].larger;
  }
  const T = target.length;
  if (market.some((m) => m.length !== T))
    throw new RangeError("target and market inputs must have the same length");

  const report = BacktestReport();
  const score = (out: NdArray): number => {
    if (typeof metric === "function") return metric(out);
    const spec = METRICS[metric];
    if (!spec.report) return lastFinite(out, spec.col);
    const rep = report(column(out, 0), column(out, 1), column(out, 2), column(out, 3)) as NdArray;
    report.reset();
    return lastFinite(rep, spec.col);
  };

  try {
    engine.reset();
    const observed = score(engine(target, ...market) as NdArray);

    let shift: number | null = null;
    if (method === "shift") {
      shift = opts.minShift ?? defaultMinShift(target);
      if (!Number.isInteger(shift) || shift < 1) throw new RangeError("min_shift must be at least 1");
    }

    // Generate the surrogates in chunks of columns, as the Python driver does;
    // each column has its own seeded generator, so the chunk size does not
    // change the result.
    const chunk = Math.max(1, Math.min(n, Math.floor(4_000_000 / Math.max(1, T))));
    const nul: number[] = [];
    const col = new Float64Array(T);
    for (let c0 = 0; c0 < n; c0 += chunk) {
      const k = Math.min(chunk, n - c0);
      const block = method === "shift"
        ? shiftBlock(target, shift as number, seed, c0, k)
        : shuffleBlock(target, seed, c0, k);
      for (let j = 0; j < k; j++) {
        for (let t = 0; t < T; t++) col[t] = block.data[t * k + j];
        engine.reset();
        const v = score(engine(col, ...market) as NdArray);
        if (Number.isFinite(v)) nul.push(v);
      }
    }
    engine.reset();

    let hits = 0;
    for (const v of nul) if (larger ? v >= observed : v <= observed) hits++;
    const pValue = Number.isFinite(observed) ? (1 + hits) / (1 + nul.length) : NaN;
    return { observed, null: Float64Array.from(nul), pValue, metric: name, method, minShift: shift };
  } finally {
    report.dispose();
  }
}

// ---- data permutations and strategyTest (Python: permute_bars,
// permute_returns, strategy_test) ------------------------------------------

function heapBlocks(T: number, ncols: number, count: number) {
  const M = current();
  const ptrs = Array.from({ length: count }, () => M.allocF64(T * ncols));
  const read = (i: number) => new Float64Array(M.viewF64(ptrs[i], T * ncols));
  const free = () => ptrs.forEach((p) => M.freeBuf(p));
  return { ptrs, read, free };
}

function heapCopy(x: ArrayLike<number>): number {
  const M = current();
  const p = M.allocF64(x.length);
  M.viewF64(p, x.length).set(x as number[]);
  return p;
}

function checkBars(o: ArrayLike<number>, h: ArrayLike<number>, l: ArrayLike<number>,
                   c: ArrayLike<number>): void {
  const T = o.length;
  if (h.length !== T || l.length !== T || c.length !== T)
    throw new RangeError("open, high, low, and close must have the same length.");
  if (T < 2) throw new RangeError("permute_bars needs at least two bars.");
  for (let t = 0; t < T; t++) {
    if (!(o[t] > 0 && h[t] > 0 && l[t] > 0 && c[t] > 0) ||
        !Number.isFinite(o[t] + h[t] + l[t] + c[t]))
      throw new RangeError("permute_bars needs finite, positive open, high, low, and close prices.");
  }
}

function barsBlock(o: ArrayLike<number>, h: ArrayLike<number>, l: ArrayLike<number>,
                   c: ArrayLike<number>, seed: number, col0: number, ncols: number) {
  const M = current();
  const T = o.length;
  checkBars(o, h, l, c);
  const ins = [o, h, l, c].map(heapCopy);
  const outs = heapBlocks(T, ncols, 5);
  try {
    M.permuteBarsInto(ins[0], ins[1], ins[2], ins[3], T, seed, col0, ncols, ...outs.ptrs);
    return [0, 1, 2, 3, 4].map(outs.read);
  } finally {
    ins.forEach((p) => M.freeBuf(p)); outs.free();
  }
}

function checkReturns(x: ArrayLike<number>, log: boolean): void {
  let p = 0;
  while (p < x.length && Number.isNaN(x[p])) p++;
  if (x.length - p < 2)
    throw new RangeError("permute_returns needs at least two values after the NaN warmup.");
  for (let t = p; t < x.length; t++) {
    if (!Number.isFinite(x[t]) || (log && !(x[t] > 0)))
      throw new RangeError(log
        ? "permute_returns with log=True needs finite, positive values."
        : "permute_returns needs finite values after the NaN warmup.");
  }
}

function returnsBlock(x: ArrayLike<number>, log: boolean, seed: number,
                      col0: number, ncols: number) {
  const M = current();
  checkReturns(x, log);
  const xp = heapCopy(x);
  const outs = heapBlocks(x.length, ncols, 2);
  try {
    M.permuteReturnsInto(xp, x.length, log, seed, col0, ncols, ...outs.ptrs);
    return [outs.read(0), outs.read(1)];
  } finally {
    M.freeBuf(xp); outs.free();
  }
}

/**
 * `n` bar-permuted copies of an OHLC series: `[open, high, low, close]`, each a
 * (T, n) array. Gaps (open over previous close) and intrabar shapes are
 * shuffled independently in log space and the path is rebuilt from the first
 * bar (Masters 2020). Mirrors Python's `permute_bars`.
 */
export function permuteBars(
  open: ArrayLike<number>, high: ArrayLike<number>, low: ArrayLike<number>,
  close: ArrayLike<number>, n: number, opts: { seed?: number } = {},
): NdArray[] {
  const seed = opts.seed ?? 0;
  checkSeed(seed);
  return barsBlock(open, high, low, close, seed, 0, n).slice(0, 4)
    .map((data) => ({ data, shape: [open.length, n] }));
}

/**
 * `n` copies of a price series with its returns in random order, as a (T, n)
 * array. Mirrors Python's `permute_returns`.
 */
export function permuteReturns(
  x: ArrayLike<number>, n: number, opts: { log?: boolean; seed?: number } = {},
): NdArray {
  const seed = opts.seed ?? 0;
  checkSeed(seed);
  const [data] = returnsBlock(x, opts.log ?? true, seed, 0, n);
  return { data, shape: [x.length, n] };
}

export interface StrategyTestOptions {
  n?: number;
  method?: "bars" | "returns" | "shift";
  ohlc?: [number, number, number, number];
  price?: number;
  log?: boolean;
  columns?: number[];
  fixed?: number[];
  metric?: PositionTestMetric;
  largerIsBetter?: boolean;
  minShift?: number;
  seed?: number;
}

export interface StrategyTestResult {
  observed: number;
  null: Float64Array;
  pValue: number;
  metric: string;
  method: "bars" | "returns" | "shift";
}

function engineOutput(out: any): NdArray {
  const v = out && out.values !== undefined ? out.values : out;   // Pipeline Output
  if (!v || !v.shape || v.shape.length !== 2 || v.shape[1] !== 4)
    throw new RangeError(
      "strategy must return a backtest engine output of shape (T, 4) " +
      "[equity, pnl, position, cost], or a pipeline output holding one");
  return v as NdArray;
}

/**
 * Test a whole strategy against the same strategy run on randomized data.
 * `strategy(...inputs)` must return a backtest engine output of shape (T, 4)
 * (or a pipeline `{values, index}` holding one). `method` is `"bars"` (permute
 * the OHLC inputs at `ohlc`), `"returns"` (permute the price input at
 * `price`), or `"shift"` (rotate the feature inputs at `columns` by one random
 * offset). With bars and returns, other inputs move with their bar unless listed
 * in `fixed`. Mirrors Python's `strategy_test`; the same seed gives the same
 * randomized inputs.
 */
export function strategyTest(
  strategy: (...inputs: Float64Array[]) => any,
  inputs: ArrayLike<number>[],
  opts: StrategyTestOptions = {},
): StrategyTestResult {
  const n = opts.n ?? 500;
  const method = opts.method ?? "bars";
  const metric = opts.metric ?? "sharpe";
  const seed = opts.seed ?? 0;
  if (method !== "bars" && method !== "returns" && method !== "shift")
    throw new RangeError('method must be "bars", "returns", or "shift"');
  if (!Number.isInteger(n) || n < 1) throw new RangeError("n must be at least 1");
  checkSeed(seed);
  let name: string, larger: boolean;
  if (typeof metric === "function") {
    name = metric.name || "custom";
    larger = opts.largerIsBetter ?? true;
  } else {
    if (!(metric in METRICS))
      throw new RangeError(`metric must be one of ${Object.keys(METRICS).sort().join(", ")} or a function`);
    name = metric;
    larger = opts.largerIsBetter ?? METRICS[metric].larger;
  }
  const data = inputs.map((x) => Float64Array.from(x as ArrayLike<number>));
  if (data.length === 0) throw new RangeError("strategyTest needs at least one input");
  const T = data[0].length, k = data.length;
  if (data.some((x) => x.length !== T)) throw new RangeError("all inputs must have the same length");
  const check = (pos: number[], what: string) => {
    if (pos.some((i) => !Number.isInteger(i) || i < 0 || i >= k))
      throw new RangeError(`${what} refers to an input that does not exist`);
    return pos;
  };
  const fixed = new Set(check(opts.fixed ?? [], "fixed"));
  let priceCols: number[] = [], shifted = new Set<number>(), shift = 0, log = true;
  if (method === "bars") {
    priceCols = check([...(opts.ohlc ?? [0, 1, 2, 3])], "ohlc");
    if (priceCols.length !== 4) throw new RangeError("ohlc must list the positions of open, high, low, close");
  } else if (method === "returns") {
    priceCols = check([opts.price ?? 0], "price");
    const px = data[priceCols[0]];
    log = opts.log ?? Array.from(px).filter(Number.isFinite).every((v) => v > 0);
  } else {
    if (!opts.columns || opts.columns.length === 0)
      throw new RangeError('method "shift" needs the feature positions in columns');
    shifted = new Set(check(opts.columns, "columns"));
    shift = opts.minShift ?? Math.max(1, Math.floor(T / 10));
    if (!Number.isInteger(shift) || shift < 1 || 2 * shift > T)
      throw new RangeError("min_shift must be in [1, T / 2]");
  }

  const report = BacktestReport();
  const score = (out: NdArray): number => {
    if (typeof metric === "function") return metric(out);
    const spec = METRICS[metric];
    if (!spec.report) return lastFinite(out, spec.col);
    const rep = report(column(out, 0), column(out, 1), column(out, 2), column(out, 3)) as NdArray;
    report.reset();
    return lastFinite(rep, spec.col);
  };

  try {
    const observed = score(engineOutput(strategy(...data)));
    const positions = Float64Array.from({ length: T }, (_, i) => i);
    const nul: number[] = [];
    const chunk = Math.max(1, Math.min(n, 64));
    for (let c0 = 0; c0 < n; c0 += chunk) {
      const m = Math.min(chunk, n - c0);
      let replaced = new Map<number, Float64Array>(), src: Float64Array;
      if (method === "bars") {
        const b = barsBlock(data[priceCols[0]], data[priceCols[1]], data[priceCols[2]],
                            data[priceCols[3]], seed, c0, m);
        priceCols.forEach((i, q) => replaced.set(i, b[q]));
        src = b[4];
      } else if (method === "returns") {
        const r = returnsBlock(data[priceCols[0]], log, seed, c0, m);
        replaced.set(priceCols[0], r[0]);
        src = r[1];
      } else {
        src = shiftBlock(positions, shift, seed, c0, m).data;
      }
      for (let j = 0; j < m; j++) {
        const run = data.map((x, i) => {
          const col = new Float64Array(T);
          if (replaced.has(i)) {
            const blk = replaced.get(i) as Float64Array;
            for (let t = 0; t < T; t++) col[t] = blk[t * m + j];
          } else if (fixed.has(i) || (method === "shift" && !shifted.has(i))) {
            return x;
          } else {
            for (let t = 0; t < T; t++) col[t] = x[src[t * m + j]];
          }
          return col;
        });
        const v = score(engineOutput(strategy(...run)));
        if (Number.isFinite(v)) nul.push(v);
      }
    }
    let hits = 0;
    for (const v of nul) if (larger ? v >= observed : v <= observed) hits++;
    const pValue = Number.isFinite(observed) ? (1 + hits) / (1 + nul.length) : NaN;
    return { observed, null: Float64Array.from(nul), pValue, metric: name, method };
  } finally {
    report.dispose();
  }
}

function positionFamily(
  engine: ScreamerOp,
  targets: ArrayLike<number>[],
  market: ArrayLike<number>[],
  opts: PositionTestOptions,
): PositionFamilyResult {
  const n = opts.n ?? 1000;
  const method = opts.method ?? "shift";
  const metric = opts.metric ?? "sharpe";
  const seed = opts.seed ?? 0;
  if (method !== "shift")
    throw new RangeError('several strategies need method "shift", which shifts them jointly');
  if (!Number.isInteger(n) || n < 1) throw new RangeError("n must be at least 1");
  checkSeed(seed);
  let name: string, larger: boolean;
  if (typeof metric === "function") {
    name = metric.name || "custom";
    larger = opts.largerIsBetter ?? true;
  } else {
    if (!(metric in METRICS))
      throw new RangeError(`metric must be one of ${Object.keys(METRICS).sort().join(", ")} or a function`);
    name = metric;
    larger = opts.largerIsBetter ?? METRICS[metric].larger;
  }
  const K = targets.length, T = targets[0].length;
  if (targets.some((x) => x.length !== T) || market.some((m) => m.length !== T))
    throw new RangeError("target and market inputs must have the same length");
  const start = Math.max(...targets.map((x) => x.length - validLength(x)));
  let shift = opts.minShift ?? Math.max(1, Math.min(
    Math.max(...targets.map(defaultMinShift)), Math.floor((T - start) / 4)));
  if (!Number.isInteger(shift) || shift < 1) throw new RangeError("min_shift must be at least 1");

  const report = BacktestReport();
  const score = (out: NdArray): number => {
    if (typeof metric === "function") return metric(out);
    const spec = METRICS[metric];
    if (!spec.report) return lastFinite(out, spec.col);
    const rep = report(column(out, 0), column(out, 1), column(out, 2), column(out, 3)) as NdArray;
    report.reset();
    return lastFinite(rep, spec.col);
  };
  const run = (x: ArrayLike<number>): number => {
    engine.reset();
    return score(engine(x, ...market) as NdArray);
  };
  try {
    const observed = Float64Array.from(targets, run);
    const nul = new Float64Array(n * K);
    const chunk = Math.max(1, Math.min(n, Math.floor(4_000_000 / Math.max(1, T))));
    const col = new Float64Array(T);
    for (let c0 = 0; c0 < n; c0 += chunk) {
      const m = Math.min(chunk, n - c0);
      targets.forEach((x, k) => {
        const block = shiftBlock(x, shift, seed, c0, m, start);
        for (let j = 0; j < m; j++) {
          for (let t = 0; t < T; t++) col[t] = block.data[t * m + j];
          nul[(c0 + j) * K + k] = run(col);
        }
      });
    }
    engine.reset();
    const sign = larger ? 1 : -1;
    const pValue = new Float64Array(K), familyPValue = new Float64Array(K);
    const best: number[] = [];
    for (let r = 0; r < n; r++) {
      let b = -Infinity;
      for (let k = 0; k < K; k++) { const v = sign * nul[r * K + k]; if (v > b) b = v; }
      if (Number.isFinite(b)) best.push(b);
    }
    for (let k = 0; k < K; k++) {
      let hits = 0, count = 0;
      for (let r = 0; r < n; r++) {
        const v = nul[r * K + k];
        if (!Number.isFinite(v)) continue;
        count++;
        if (sign * v >= sign * observed[k]) hits++;
      }
      const o = observed[k];
      pValue[k] = Number.isFinite(o) ? (1 + hits) / (1 + count) : NaN;
      familyPValue[k] = Number.isFinite(o)
        ? (1 + best.filter((b) => b >= sign * o).length) / (1 + best.length) : NaN;
    }
    return { observed, null: { data: nul, shape: [n, K] }, pValue, familyPValue,
             metric: name, method: "shift", minShift: shift };
  } finally {
    report.dispose();
  }
}

// ---- probabilistic and deflated Sharpe ratios (Python: probabilistic_sharpe,
// deflated_sharpe) ---------------------------------------------------------

function pnlOf(pnl: ArrayLike<number> | NdArray): Float64Array {
  const nd = pnl as NdArray;
  if (nd && nd.shape && nd.shape.length === 2 && nd.shape[1] === 4) return column(nd, 1);
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
  const M = current();
  const x = pnlOf(pnl);
  checkPnl(x);
  const p = heapCopy(x);
  try { return M.probabilisticSharpe(p, x.length, benchmark); } finally { M.freeBuf(p); }
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
  const p = heapCopy(x), t = heapCopy(trialSharpes), out = M.allocF64(2);
  try {
    M.deflatedSharpeInto(p, x.length, t, trialSharpes.length, out);
    const v = M.viewF64(out, 2);
    return { probability: v[0], benchmark: v[1] };
  } finally {
    M.freeBuf(p); M.freeBuf(t); M.freeBuf(out);
  }
}
