// Randomization tests for backtest significance. Mirrors screamer/validation.py.
//
// `positionTest` keeps the market data and the strategy's trading profile,
// randomizes when the strategy holds its positions, replays each randomized
// copy through the same backtest engine, and compares the real metric with
// that null distribution. These are batch-only validation tools, not streaming
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
                    col0: number, ncols: number): NdArray {
  const L = validLength(x);
  if (L < 2)
    throw new RangeError("circular_shift needs at least two values after the NaN warmup.");
  if (2 * minShift > L)
    throw new RangeError("min_shift is too large: it must be at most half the series " +
                         "length after the NaN warmup.");
  return surrogates(x, ncols, (xp, op) =>
    current().circularShiftInto(xp, x.length, minShift, seed, col0, ncols, op));
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
 */
export function positionTest(
  engine: ScreamerOp,
  target: ArrayLike<number>,
  market: ArrayLike<number>[],
  opts: PositionTestOptions = {},
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
