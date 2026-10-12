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
import { asNd, type NdArray } from "./ndarray.js";
import type { ScreamerOp } from "./runtime.js";
import { BacktestReport, BacktestStats, PortfolioReport } from "./generated/ops.js";

// Leading NaN warmup length and the length of the series after it.
// A target as a Float64Array, so the engine returns an NdArray: all number[]
// inputs would give an Array of per-event results (rule 3).
const asTyped = (x: ArrayLike<number>): ArrayLike<number> => (Array.isArray(x) ? Float64Array.from(x) : x);

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
  let xPtr = 0, outPtr = 0;          // freeBuf(0) is a no-op, so a failed alloc is safe
  try {
    xPtr = M.allocF64(T); outPtr = M.allocF64(T * ncols);
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

// Engine output columns: [equity, pnl, position, cost].
// BacktestReport columns: [drawdown, cum_cost, turnover, trades, max_drawdown, sharpe].
// BacktestStats columns: [exposure, round_trips, win_rate, profit_factor, avg_win,
//                         avg_loss, avg_holding, sortino, calmar].
// `larger` is the direction as a test metric; null: none, largerIsBetter is needed.
type Source = "engine" | "report" | "stats";
const METRICS: Record<StatName, { src: Source; col: number; larger: boolean | null }> = {
  total_pnl: { src: "engine", col: 0, larger: true },
  max_drawdown: { src: "report", col: 4, larger: true },
  total_cost: { src: "report", col: 1, larger: false },
  turnover: { src: "report", col: 2, larger: null },
  num_trades: { src: "report", col: 3, larger: null },
  sharpe: { src: "report", col: 5, larger: true },
  exposure: { src: "stats", col: 0, larger: null },
  round_trips: { src: "stats", col: 1, larger: null },
  win_rate: { src: "stats", col: 2, larger: true },
  profit_factor: { src: "stats", col: 3, larger: true },
  avg_win: { src: "stats", col: 4, larger: true },
  avg_loss: { src: "stats", col: 5, larger: true },
  avg_holding: { src: "stats", col: 6, larger: null },
  sortino: { src: "stats", col: 7, larger: true },
  calmar: { src: "stats", col: 8, larger: true },
};
const STAT_NAMES = Object.keys(METRICS) as StatName[];

function metricSpec(metric: PositionTestMetric, largerIsBetter?: boolean)
  : { name: string; larger: boolean } {
  if (typeof metric === "function")
    return { name: metric.name || "custom", larger: largerIsBetter ?? true };
  if (!(metric in METRICS))
    throw new RangeError(`metric must be one of ${[...STAT_NAMES].sort().join(", ")} or a function`);
  const larger = largerIsBetter ?? METRICS[metric].larger;
  if (larger === null)
    throw new RangeError(`metric "${metric}" has no natural direction; pass largerIsBetter`);
  return { name: metric, larger };
}

// Scores runs: every statistic of one run's output, and the metric. A (T, 4)
// output is one strategy; a (T, K, 4) output is a portfolio, scored as one
// book: equity, pnl, and cost summed over assets, the gross position as its
// position, and turnover and trades from PortfolioReport (per-asset changes).
function makeScorer(metric: PositionTestMetric, periodsPerYear: number) {
  const report = BacktestReport(periodsPerYear);
  const stats = BacktestStats(periodsPerYear);
  const portfolio = PortfolioReport();
  const score = (out: NdArray): { value: number; stats: Float64Array } => {
    let cols: Float64Array[], engine: NdArray, turnover: NdArray | null = null;
    if (out.shape.length === 3) {
      const [T, K] = out.shape;
      cols = [0, 1, 2, 3].map(() => new Float64Array(T));
      for (let t = 0; t < T; t++)
        for (let k = 0; k < K; k++) {
          const b = (t * K + k) * 4;
          cols[0][t] += out.data[b]; cols[1][t] += out.data[b + 1];
          cols[2][t] += Math.abs(out.data[b + 2]); cols[3][t] += out.data[b + 3];
        }
      const flat = new Float64Array(T * 4);
      for (let t = 0; t < T; t++) for (let c = 0; c < 4; c++) flat[t * 4 + c] = cols[c][t];
      engine = { data: flat, shape: [T, 4] };
      turnover = portfolio(out) as NdArray;
      portfolio.reset();
    } else {
      cols = [0, 1, 2, 3].map((i) => column(out, i));
      engine = out;
    }
    const rep = report(...cols) as NdArray;
    const st = stats(...cols) as NdArray;
    report.reset();
    stats.reset();
    if (turnover) {                         // per-asset turnover and trade count
      const [T] = rep.shape;
      for (let t = 0; t < T; t++)
        for (const c of [2, 3]) rep.data[t * 6 + c] = turnover.data[t * 6 + c];
    }
    const src = { engine, report: rep, stats: st };
    const all = Float64Array.from(STAT_NAMES, (k) => lastFinite(src[METRICS[k].src], METRICS[k].col));
    const value = typeof metric === "function"
      ? metric(out) : all[STAT_NAMES.indexOf(metric)];
    return { value, stats: all };
  };
  const dispose = () => { report.dispose(); stats.dispose(); portfolio.dispose(); };
  return { score, dispose };
}

function statsRecord<V>(make: (i: number) => V): Record<StatName, V> {
  const out = {} as Record<StatName, V>;
  STAT_NAMES.forEach((k, i) => { out[k] = make(i); });
  return out;
}

// Every NdArray test goes through asNd, so a malformed array throws here.
const isNdArray = (x: any): x is NdArray => asNd(x) !== null;

function lastFinite(a: NdArray, col: number): number {
  const [rows, cols] = a.shape;
  for (let r = rows - 1; r >= 0; r--) {
    const v = a.data[r * cols + col];
    if (Number.isFinite(v)) return v;
  }
  return NaN;
}

// Column k of a (T, K, 4) engine output, as one run's (T, 4) output.
function run(out: NdArray, k: number): NdArray {
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
 * Pass an array of target columns to test several strategies jointly (method
 * "shift"): every round shifts all of them by the same offset, and
 * `familyPValue` compares each strategy with the best of the K per round.
 *
 * With a portfolio engine (`BacktestPortfolioTarget`,
 * `BacktestPortfolioOHLCTarget`), `target` is the (T, K) target of one book and
 * `market` its (T, K) inputs. Every round shifts all K assets by one offset and
 * scores the whole book, so the result is a single test.
 */
export function positionTest(
  engine: ScreamerOp,
  target: NdArray,
  market: (NdArray | ArrayLike<number>[])[],
  opts?: PositionTestOptions,
): PositionTestResult;
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
  target: ArrayLike<number> | ArrayLike<number>[] | NdArray,
  market: any[],
  opts: PositionTestOptions = {},
): PositionTestResult | PositionFamilyResult {
  if ((engine as any).rowOp) return positionPortfolio(engine, target, market, opts);
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
  const { name, larger } = metricSpec(metric, opts.largerIsBetter);
  const T = target.length;
  if (market.some((m) => m.length !== T))
    throw new RangeError("target and market inputs must have the same length");

  const scorer = makeScorer(metric, opts.periodsPerYear ?? 1);
  try {
    engine.reset();
    const obs = scorer.score(engine(asTyped(target), ...market) as NdArray);
    const observed = obs.value;
    const nullStats = statsRecord(() => new Float64Array(n));

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
    for (let c0 = 0; c0 < n; c0 += chunk) {
      const k = Math.min(chunk, n - c0);
      const block = method === "shift"
        ? shiftBlock(target, shift as number, seed, c0, k)
        : shuffleBlock(target, seed, c0, k);
      engine.reset();
      const out = engine({ data: block.data, shape: [T, k] }, ...market) as NdArray;
      for (let j = 0; j < k; j++) {
        const r = scorer.score(run(out, j));
        STAT_NAMES.forEach((name, i) => { nullStats[name][c0 + j] = r.stats[i]; });
        if (Number.isFinite(r.value)) nul.push(r.value);
      }
    }
    engine.reset();

    let hits = 0;
    for (const v of nul) if (larger ? v >= observed : v <= observed) hits++;
    const pValue = Number.isFinite(observed) ? (1 + hits) / (1 + nul.length) : NaN;
    return { observed, null: Float64Array.from(nul), pValue, metric: name, method, minShift: shift,
             observedStats: statsRecord((i) => obs.stats[i]), nullStats };
  } finally {
    scorer.dispose();
  }
}

// One book of K assets traded by a portfolio engine: all columns shifted
// together, scored on the summed book. Mirrors Python's position_test with a
// portfolio engine.
function positionPortfolio(
  engine: ScreamerOp, target: any, market: any[], opts: PositionTestOptions,
): PositionTestResult {
  const n = opts.n ?? 1000;
  const method = opts.method ?? "shift";
  const metric = opts.metric ?? "sharpe";
  const seed = opts.seed ?? 0;
  if (method !== "shift")
    throw new RangeError('a portfolio is tested with method "shift", which moves all assets together');
  if (!Number.isInteger(n) || n < 1) throw new RangeError("n must be at least 1");
  checkSeed(seed);
  const { name, larger } = metricSpec(metric, opts.largerIsBetter);
  let T: number, K: number, flat: ArrayLike<number>;
  if (isNdArray(target) && target.shape.length === 2) {
    [T, K] = target.shape; flat = target.data;
  } else if (Array.isArray(target) && target.length > 0 && typeof target[0] !== "number") {
    T = target.length; K = (target[0] as ArrayLike<number>).length;
    const f = new Float64Array(T * K);
    (target as ArrayLike<number>[]).forEach((r, t) => f.set(r as number[], t * K));
    flat = f;
  } else {
    throw new RangeError("a portfolio engine needs a (T, K) target");
  }
  const cols = Array.from({ length: K }, (_, k) => Float64Array.from({ length: T }, (_, t) => flat[t * K + k]));
  const start = Math.max(...cols.map((x) => x.length - validLength(x)));
  const shift = opts.minShift ?? Math.max(1, Math.min(
    Math.max(...cols.map(defaultMinShift)), Math.floor((T - start) / 4)));
  if (!Number.isInteger(shift) || shift < 1) throw new RangeError("min_shift must be at least 1");

  const scorer = makeScorer(metric, opts.periodsPerYear ?? 1);
  const run = (tgt: NdArray) => {
    engine.reset();
    return scorer.score(engine(tgt, ...market) as NdArray);
  };
  try {
    const obs = run({ data: Float64Array.from(flat), shape: [T, K] });
    const nullStats = statsRecord(() => new Float64Array(n));
    const nul: number[] = [];
    const chunk = Math.max(1, Math.min(n, 256));
    for (let c0 = 0; c0 < n; c0 += chunk) {
      const m = Math.min(chunk, n - c0);
      const blocks = cols.map((x) => shiftBlock(x, shift, seed, c0, m, start));
      for (let j = 0; j < m; j++) {
        const data = new Float64Array(T * K);
        for (let t = 0; t < T; t++) for (let k = 0; k < K; k++) data[t * K + k] = blocks[k].data[t * m + j];
        const r = run({ data, shape: [T, K] });
        STAT_NAMES.forEach((q, i) => { nullStats[q][c0 + j] = r.stats[i]; });
        if (Number.isFinite(r.value)) nul.push(r.value);
      }
    }
    engine.reset();
    let hits = 0;
    for (const v of nul) if (larger ? v >= obs.value : v <= obs.value) hits++;
    const pValue = Number.isFinite(obs.value) ? (1 + hits) / (1 + nul.length) : NaN;
    return { observed: obs.value, null: Float64Array.from(nul), pValue, metric: name, method: "shift",
             minShift: shift, observedStats: statsRecord((i) => obs.stats[i]), nullStats };
  } finally {
    scorer.dispose();
  }
}

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
// K = 1 is one series. The checks repeat the C++ ones so that a bad input
// throws a RangeError here instead of aborting inside WASM.
function checkBars(o: ArrayLike<number>, h: ArrayLike<number>, l: ArrayLike<number>,
                   c: ArrayLike<number>, T: number, mode: PermuteMode,
                   notionalVolume: boolean): void {
  const N = o.length;
  if (h.length !== N || l.length !== N || c.length !== N)
    throw new RangeError("open, high, low, and close must have the same length.");
  if (T < 2) throw new RangeError("permute_bars needs at least two bars.");
  for (let i = 0; i < N; i++) {
    if (!Number.isFinite(o[i] + h[i] + l[i] + c[i]))
      throw new RangeError("permute_bars needs finite open, high, low, and close prices.");
    if ((mode === "log" || notionalVolume) && !(o[i] > 0 && h[i] > 0 && l[i] > 0 && c[i] > 0))
      throw new RangeError(mode === "log"
        ? 'permute_bars with mode="log" needs positive prices; use mode="diff".'
        : 'volume_mode="notional" needs positive prices.');
  }
}

// Outputs: open, high, low, close (T x ncols x K), source (T x ncols), and the
// moved volume (T x ncols x K) when `vol` is given.
function barsBlock(o: ArrayLike<number>, h: ArrayLike<number>, l: ArrayLike<number>,
                   c: ArrayLike<number>, T: number, K: number, mode: PermuteMode, seed: number,
                   col0: number, ncols: number, vol: ArrayLike<number> | null, notional: boolean) {
  const M = current();
  checkBars(o, h, l, c, T, mode, notional && vol !== null);
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
    M.permuteBarsInto(ins[0], ins[1], ins[2], ins[3], T, K, mode, seed, col0, ncols,
                      p[0], p[1], p[2], p[3], p[4], vp, notional, vol === null ? 0 : p[5]);
    return p.map((_, k) => (outs as ReturnType<typeof heapBlocks>).read(k));
  } finally {
    ins.forEach((p) => M.freeBuf(p)); outs?.free();
  }
}

function checkReturns(x: ArrayLike<number>, T: number, K: number, mode: PermuteMode,
                      notionalVolume: boolean): void {
  let p = 0;
  for (let a = 0; a < K; a++) {
    let q = 0;
    while (q < T && Number.isNaN(x[q * K + a])) q++;
    p = Math.max(p, q);
  }
  if (T - p < 2)
    throw new RangeError("permute_returns needs at least two values after the NaN warmup.");
  for (let i = p * K; i < T * K; i++) {
    if (!Number.isFinite(x[i]))
      throw new RangeError("permute_returns needs finite values after the NaN warmup.");
    if ((mode === "log" || notionalVolume) && !(x[i] > 0))
      throw new RangeError(mode === "log"
        ? 'permute_returns with mode="log" needs positive values; use mode="diff".'
        : 'volume_mode="notional" needs positive prices.');
  }
}

// Outputs: prices (T x ncols x K), source (T x ncols), and the moved volume
// (T x ncols x K) when `vol` is given.
function returnsBlock(x: ArrayLike<number>, T: number, K: number, mode: PermuteMode,
                      seed: number, col0: number, ncols: number,
                      vol: ArrayLike<number> | null, notional: boolean) {
  const M = current();
  checkReturns(x, T, K, mode, notional && vol !== null);
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
    M.permuteReturnsInto(xp, T, K, mode, seed, col0, ncols, p[0], p[1],
                         vp, notional, vol === null ? 0 : p[2]);
    return p.map((_, k) => (outs as ReturnType<typeof heapBlocks>).read(k));
  } finally {
    M.freeBuf(xp); M.freeBuf(vp); outs?.free();
  }
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
  open: ArrayLike<number>, high: ArrayLike<number>, low: ArrayLike<number>,
  close: ArrayLike<number>, n: number, opts: PermuteOptions = {},
): NdArray[] {
  const seed = opts.seed ?? 0, mode = opts.mode ?? "log", vm = opts.volumeMode ?? "keep";
  checkSeed(seed);
  checkModes(mode, vm);
  const vol = opts.volume ?? null;
  const out = barsBlock(open, high, low, close, open.length, 1, mode, seed, 0, n, vol,
                        vm === "notional");
  const keep = vol === null ? [0, 1, 2, 3] : [0, 1, 2, 3, 5];
  return keep.map((k) => ({ data: out[k], shape: [open.length, n] }));
}

/**
 * `n` permuted copies of a price series, as a (T, n) array; with `volume`,
 * `[prices, volume]`. `mode` is "log" (shuffle log returns; the default),
 * "diff" (shuffle differences), or "level" (shuffle the values). Mirrors
 * Python's `permute_returns`.
 */
export function permuteReturns(
  x: ArrayLike<number>, n: number, opts: PermuteOptions = {},
): NdArray | NdArray[] {
  const seed = opts.seed ?? 0, mode = opts.mode ?? "log", vm = opts.volumeMode ?? "keep";
  checkSeed(seed);
  checkModes(mode, vm);
  const vol = opts.volume ?? null;
  const out = returnsBlock(x, x.length, 1, mode, seed, 0, n, vol, vm === "notional");
  const nd = (d: Float64Array): NdArray => ({ data: d, shape: [x.length, n] });
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

function engineOutput(out: any): NdArray {
  const v = out && out.values !== undefined ? out.values : out;   // Pipeline Output
  const ok = asNd(v) && ((v.shape.length === 2 && v.shape[1] === 4) ||
                              (v.shape.length === 3 && v.shape[2] === 4));
  if (!ok)
    throw new RangeError(
      "strategy must return a backtest engine output of shape (T, 4) " +
      "[equity, pnl, position, cost], a portfolio output of shape (T, K, 4), " +
      "or a pipeline output holding one");
  return v as NdArray;
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
    const K = first.length;
    const data = new Float64Array(T * K);
    for (let t = 0; t < T; t++) {
      const row = x[t];
      if (row == null || row.length !== K)
        throw new RangeError(`${where}: every row must hold ${K} values`);
      for (let a = 0; a < K; a++) data[t * K + a] = row[a];
    }
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
  const n = opts.n ?? 500;
  const method = opts.method ?? "bars";
  const metric = opts.metric ?? "sharpe";
  const seed = opts.seed ?? 0;
  if (method !== "bars" && method !== "returns" && method !== "shift")
    throw new RangeError('method must be "bars", "returns", or "shift"');
  if (!Number.isInteger(n) || n < 1) throw new RangeError("n must be at least 1");
  checkSeed(seed);
  const { name, larger } = metricSpec(metric, opts.largerIsBetter);
  const blocks = inputs.map((x, i) => toBlock(x, i));
  if (blocks.length === 0) throw new RangeError("strategyTest needs at least one input");
  const T = blocks[0].T, k = blocks.length;
  if (blocks.some((b) => b.T !== T)) throw new RangeError("all inputs must have the same length");
  const sameShape = (a: Block, b: Block) => a.panel === b.panel && a.K === b.K;
  const check = (pos: number[], what: string) => {
    if (pos.some((i) => !Number.isInteger(i) || i < 0 || i >= k))
      throw new RangeError(`${what} refers to an input that does not exist`);
    return pos;
  };
  const fixed = new Set(check(opts.fixed ?? [], "fixed"));
  const mode = opts.mode ?? "log", volumeMode = opts.volumeMode ?? "keep";
  checkModes(mode, volumeMode);
  let volCol = opts.volume === undefined ? null : check([opts.volume], "volume")[0];
  if (volCol !== null && method === "shift")
    throw new RangeError('volume applies to method "bars" or "returns"');
  const notional = volumeMode === "notional";
  let priceCols: number[] = [], shifted = new Set<number>(), shift = 0;
  if (method === "bars") {
    priceCols = check([...(opts.ohlc ?? [0, 1, 2, 3])], "ohlc");
    if (priceCols.length !== 4) throw new RangeError("ohlc must list the positions of open, high, low, close");
    if (priceCols.some((i) => !sameShape(blocks[i], blocks[priceCols[0]])))
      throw new RangeError("open, high, low, and close must have the same shape");
  } else if (method === "returns") {
    priceCols = check([opts.price ?? 0], "price");
  } else {
    if (!opts.columns || opts.columns.length === 0)
      throw new RangeError('method "shift" needs the feature positions in columns');
    shifted = new Set(check(opts.columns, "columns"));
    shift = opts.minShift ?? Math.max(1, Math.floor(T / 10));
    if (!Number.isInteger(shift) || shift < 1 || 2 * shift > T)
      throw new RangeError("min_shift must be in [1, T / 2]");
  }
  // A volume with the shape of the prices goes to the permutation kernel.
  // Otherwise "keep" moves it with its bar like any other input, and
  // "notional" cannot rescale it.
  if (volCol !== null && !sameShape(blocks[volCol], blocks[priceCols[0]])) {
    if (notional)
      throw new RangeError('volume_mode="notional" needs a volume input with the shape of the prices');
    volCol = null;
  }
  const P = priceCols.length ? blocks[priceCols[0]] : blocks[0];
  const vol = volCol === null ? null : blocks[volCol].data;

  const scorer = makeScorer(metric, opts.periodsPerYear ?? 1);
  try {
    const obs = scorer.score(engineOutput(strategy(...blocks.map((b) => asArg(b, b.data)))));
    const observed = obs.value;
    const nullStats = statsRecord(() => new Float64Array(n));
    const positions = Float64Array.from({ length: T }, (_, i) => i);
    const nul: number[] = [];
    const chunk = Math.max(1, Math.min(n, 64));
    for (let c0 = 0; c0 < n; c0 += chunk) {
      const m = Math.min(chunk, n - c0);
      let replaced = new Map<number, Float64Array>(), src: Float64Array;
      if (method === "bars") {
        const [o, h, l, c] = priceCols.map((i) => blocks[i].data);
        const b = barsBlock(o, h, l, c, T, P.K, mode, seed, c0, m, vol, notional);
        priceCols.forEach((i, q) => replaced.set(i, b[q]));
        if (volCol !== null) replaced.set(volCol, b[5]);
        src = b[4];
      } else if (method === "returns") {
        const r = returnsBlock(P.data, T, P.K, mode, seed, c0, m, vol, notional);
        replaced.set(priceCols[0], r[0]);
        if (volCol !== null) replaced.set(volCol, r[2]);
        src = r[1];
      } else {
        src = shiftBlock(positions, shift, seed, c0, m).data;
      }
      for (let j = 0; j < m; j++) {
        const run = blocks.map((b, i) => {
          if (fixed.has(i) || (method === "shift" && !shifted.has(i))) return asArg(b, b.data);
          const K = b.K, col = new Float64Array(T * K);
          const blk = replaced.get(i);
          if (blk !== undefined) {
            // copy j of a (T x m x K) block
            for (let t = 0; t < T; t++)
              col.set(blk.subarray((t * m + j) * K, (t * m + j + 1) * K), t * K);
          } else {
            for (let t = 0; t < T; t++) {
              const s = src[t * m + j] * K;
              col.set(b.data.subarray(s, s + K), t * K);
            }
          }
          return asArg(b, col);
        });
        const r = scorer.score(engineOutput(strategy(...run)));
        STAT_NAMES.forEach((q, i) => { nullStats[q][c0 + j] = r.stats[i]; });
        if (Number.isFinite(r.value)) nul.push(r.value);
      }
    }
    let hits = 0;
    for (const v of nul) if (larger ? v >= observed : v <= observed) hits++;
    const pValue = Number.isFinite(observed) ? (1 + hits) / (1 + nul.length) : NaN;
    return { observed, null: Float64Array.from(nul), pValue, metric: name, method,
             observedStats: statsRecord((i) => obs.stats[i]), nullStats };
  } finally {
    scorer.dispose();
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
  const { name, larger } = metricSpec(metric, opts.largerIsBetter);
  const K = targets.length, T = targets[0].length;
  if (targets.some((x) => x.length !== T) || market.some((m) => m.length !== T))
    throw new RangeError("target and market inputs must have the same length");
  const start = Math.max(...targets.map((x) => x.length - validLength(x)));
  let shift = opts.minShift ?? Math.max(1, Math.min(
    Math.max(...targets.map(defaultMinShift)), Math.floor((T - start) / 4)));
  if (!Number.isInteger(shift) || shift < 1) throw new RangeError("min_shift must be at least 1");

  const scorer = makeScorer(metric, opts.periodsPerYear ?? 1);
  const observe = (x: ArrayLike<number>) => {
    engine.reset();
    return scorer.score(engine(asTyped(x), ...market) as NdArray);
  };
  try {
    const obs = targets.map(observe);
    const observed = Float64Array.from(obs, (r) => r.value);
    const observedStats = statsRecord((i) => Float64Array.from(obs, (r) => r.stats[i]));
    const nullStats = statsRecord((): NdArray => ({ data: new Float64Array(n * K), shape: [n, K] }));
    const nul = new Float64Array(n * K);
    const chunk = Math.max(1, Math.min(n, Math.floor(4_000_000 / Math.max(1, T))));
    for (let c0 = 0; c0 < n; c0 += chunk) {
      const m = Math.min(chunk, n - c0);
      targets.forEach((x, k) => {
        const block = shiftBlock(x, shift, seed, c0, m, start);
        engine.reset();
        const out = engine({ data: block.data, shape: [T, m] }, ...market) as NdArray;
        for (let j = 0; j < m; j++) {
          const r = scorer.score(run(out, j));
          nul[(c0 + j) * K + k] = r.value;
          STAT_NAMES.forEach((q, i) => { nullStats[q].data[(c0 + j) * K + k] = r.stats[i]; });
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
             metric: name, method: "shift", minShift: shift, observedStats, nullStats };
  } finally {
    scorer.dispose();
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
  let p = 0, t = 0, out = 0;
  try {
    p = heapCopy(x); t = heapCopy(trialSharpes); out = M.allocF64(2);
    M.deflatedSharpeInto(p, x.length, t, trialSharpes.length, out);
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
}

export interface WalkForwardResult {
  /** The stitched out-of-sample target: 0 before the first test window. */
  target: Float64Array;
  /** The chosen candidate per fold; -1 when every score was NaN (flat). */
  choice: Int32Array;
  /** The first step of each test window. */
  foldStart: Int32Array;
  /** Every candidate's metric on every training window, (nFolds, K) row-major. */
  trainScores: NdArray;
  /** The engine output on `target`, (T, 4). */
  output: NdArray;
  /** Every statistic of `output` from the first test window to the end. */
  stats: Record<StatName, number>;
  metric: string;
}

// Rows [a, b) of a (T, 4) engine output, scored as a backtest of its own: the
// equity is shifted so the window starts from the equity the run started from.
// Mirrors `_window` in validation.py.
function engineWindow(out: NdArray, a: number, b: number): NdArray {
  const data = out.data.slice(a * 4, b * 4);
  if (a > 0) {
    let prior = NaN;
    for (let t = a - 1; t >= 0; t--)
      if (Number.isFinite(out.data[t * 4])) { prior = out.data[t * 4]; break; }
    if (!Number.isNaN(prior)) {
      let first = 0;
      while (first < out.shape[0] && Number.isNaN(out.data[first * 4])) first++;
      const shift = prior - (out.data[first * 4] - out.data[first * 4 + 1]);
      for (let r = 0; r < b - a; r++) data[r * 4] -= shift;
    }
  }
  return { data, shape: [b - a, 4] };
}

// The best of one fold's scores: NaN loses, ties go to the lowest index, -1
// when every score is NaN.
function pickBest(scores: ArrayLike<number>, larger: boolean): number {
  let best = -1, bestV = -Infinity;
  for (let k = 0; k < scores.length; k++) {
    if (Number.isNaN(scores[k])) continue;
    const v = larger ? scores[k] : -scores[k];
    if (best < 0 || v > bestV) { best = k; bestV = v; }
  }
  return best;
}

function targetColumns(targets: ArrayLike<number> | ArrayLike<number>[] | NdArray): Float64Array[] {
  if (isNdArray(targets)) {
    if (targets.shape.length === 1) return [Float64Array.from(targets.data)];
    if (targets.shape.length !== 2) throw new RangeError("targets must be a (T,) or (T, K) array");
    const [T, K] = targets.shape;
    return Array.from({ length: K }, (_, k) =>
      Float64Array.from({ length: T }, (_, t) => targets.data[t * K + k]));
  }
  if (Array.isArray(targets) && targets.length > 0 && typeof targets[0] !== "number")
    return (targets as ArrayLike<number>[]).map((x) => Float64Array.from(x));
  return [Float64Array.from(targets as ArrayLike<number>)];
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
 * covers it from the first test window on. Mirrors Python's
 * `screamer.walk_forward`.
 */
export function walkForward(
  engine: ScreamerOp,
  targets: ArrayLike<number> | ArrayLike<number>[] | NdArray,
  market: ArrayLike<number>[],
  opts: WalkForwardOptions,
): WalkForwardResult {
  if ((engine as any).rowOp)
    throw new RangeError("walkForward needs a single-asset engine; portfolio engines are not supported");
  const metric = opts.metric ?? "sharpe";
  const { name, larger } = metricSpec(metric, opts.largerIsBetter);
  const cols = targetColumns(targets);
  const K = cols.length, T = cols[0].length;
  if (cols.some((x) => x.length !== T) || market.some((m) => m.length !== T))
    throw new RangeError("targets and price inputs must have the same length");
  const { train, test } = opts;
  if (!Number.isInteger(train) || train < 2) throw new RangeError("train must be an integer of at least 2");
  if (!Number.isInteger(test) || test < 1) throw new RangeError("test must be an integer of at least 1");
  if (train >= T) throw new RangeError("train must be shorter than the series");

  const starts: number[] = [];
  for (let t0 = train; t0 < T; t0 += test) starts.push(t0);
  const F = starts.length;
  const scorer = makeScorer(metric, opts.periodsPerYear ?? 1);
  try {
    const block = new Float64Array(T * K);
    cols.forEach((x, k) => { for (let t = 0; t < T; t++) block[t * K + k] = x[t]; });
    engine.reset();
    const all = engine({ data: block, shape: [T, K] }, ...market) as NdArray;
    const full = cols.map((_, k) => run(all, k));
    const scores = new Float64Array(F * K);
    const choice = new Int32Array(F);
    const target = new Float64Array(T);
    starts.forEach((t0, i) => {
      const a = opts.anchored ? 0 : t0 - train;
      for (let k = 0; k < K; k++) scores[i * K + k] = scorer.score(engineWindow(full[k], a, t0)).value;
      const c = pickBest(scores.subarray(i * K, (i + 1) * K), larger);
      choice[i] = c;
      if (c >= 0) for (let t = t0; t < Math.min(t0 + test, T); t++) target[t] = cols[c][t];
    });
    engine.reset();
    const output = engine(target, ...market) as NdArray;
    engine.reset();
    const oos = scorer.score(engineWindow(output, starts[0], T));
    return { target, choice, foldStart: Int32Array.from(starts),
             trainScores: { data: scores, shape: [F, K] }, output,
             stats: statsRecord((i) => oos.stats[i]), metric: name };
  } finally {
    scorer.dispose();
  }
}
