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
import { BacktestReport, BacktestStats, PortfolioReport } from "./generated/ops.js";

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

const isNdArray = (x: any): x is NdArray =>
  x != null && x.data instanceof Float64Array && Array.isArray(x.shape);

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
    const obs = scorer.score(engine(target, ...market) as NdArray);
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
    const col = new Float64Array(T);
    for (let c0 = 0; c0 < n; c0 += chunk) {
      const k = Math.min(chunk, n - c0);
      const block = method === "shift"
        ? shiftBlock(target, shift as number, seed, c0, k)
        : shuffleBlock(target, seed, c0, k);
      for (let j = 0; j < k; j++) {
        for (let t = 0; t < T; t++) col[t] = block.data[t * k + j];
        engine.reset();
        const r = scorer.score(engine(col, ...market) as NdArray);
        STAT_NAMES.forEach((k, i) => { nullStats[k][c0 + j] = r.stats[i]; });
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

function heapBlocks(T: number, ncols: number, count: number) {
  const M = current();
  const ptrs: number[] = [];
  const free = () => ptrs.forEach((p) => M.freeBuf(p));
  try {
    for (let i = 0; i < count; i++) ptrs.push(M.allocF64(T * ncols));
  } catch (e) {
    free();
    throw e;
  }
  const read = (i: number) => new Float64Array(M.viewF64(ptrs[i], T * ncols));
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

function checkBars(o: ArrayLike<number>, h: ArrayLike<number>, l: ArrayLike<number>,
                   c: ArrayLike<number>, mode: PermuteMode, notionalVolume: boolean): void {
  const T = o.length;
  if (h.length !== T || l.length !== T || c.length !== T)
    throw new RangeError("open, high, low, and close must have the same length.");
  if (T < 2) throw new RangeError("permute_bars needs at least two bars.");
  for (let t = 0; t < T; t++) {
    if (!Number.isFinite(o[t] + h[t] + l[t] + c[t]))
      throw new RangeError("permute_bars needs finite open, high, low, and close prices.");
    if ((mode === "log" || notionalVolume) && !(o[t] > 0 && h[t] > 0 && l[t] > 0 && c[t] > 0))
      throw new RangeError(mode === "log"
        ? 'permute_bars with mode="log" needs positive prices; use mode="diff".'
        : 'volume_mode="notional" needs positive prices.');
  }
}

function barsBlock(o: ArrayLike<number>, h: ArrayLike<number>, l: ArrayLike<number>,
                   c: ArrayLike<number>, mode: PermuteMode, seed: number, col0: number,
                   ncols: number, vol: ArrayLike<number> | null, notional: boolean) {
  const M = current();
  const T = o.length;
  checkBars(o, h, l, c, mode, notional && vol !== null);
  if (vol !== null && vol.length !== T)
    throw new RangeError("volume must have the same length as the prices.");
  const ins: number[] = [];
  let outs: ReturnType<typeof heapBlocks> | null = null;
  try {
    for (const a of [o, h, l, c]) ins.push(heapCopy(a));
    const vp = vol === null ? 0 : heapCopy(vol);
    ins.push(vp);
    outs = heapBlocks(T, ncols, vol === null ? 5 : 6);
    const p = outs.ptrs;
    M.permuteBarsInto(ins[0], ins[1], ins[2], ins[3], T, mode, seed, col0, ncols,
                      p[0], p[1], p[2], p[3], p[4], vp, notional, vol === null ? 0 : p[5]);
    return p.map((_, k) => (outs as ReturnType<typeof heapBlocks>).read(k));
  } finally {
    ins.forEach((p) => M.freeBuf(p)); outs?.free();
  }
}

function checkReturns(x: ArrayLike<number>, mode: PermuteMode, notionalVolume: boolean): void {
  let p = 0;
  while (p < x.length && Number.isNaN(x[p])) p++;
  if (x.length - p < 2)
    throw new RangeError("permute_returns needs at least two values after the NaN warmup.");
  for (let t = p; t < x.length; t++) {
    if (!Number.isFinite(x[t]))
      throw new RangeError("permute_returns needs finite values after the NaN warmup.");
    if ((mode === "log" || notionalVolume) && !(x[t] > 0))
      throw new RangeError(mode === "log"
        ? 'permute_returns with mode="log" needs positive values; use mode="diff".'
        : 'volume_mode="notional" needs positive prices.');
  }
}

function returnsBlock(x: ArrayLike<number>, mode: PermuteMode, seed: number, col0: number,
                      ncols: number, vol: ArrayLike<number> | null, notional: boolean) {
  const M = current();
  checkReturns(x, mode, notional && vol !== null);
  if (vol !== null && vol.length !== x.length)
    throw new RangeError("volume must have the same length as the prices.");
  let xp = 0, vp = 0;
  let outs: ReturnType<typeof heapBlocks> | null = null;
  try {
    xp = heapCopy(x);
    if (vol !== null) vp = heapCopy(vol);
    outs = heapBlocks(x.length, ncols, vol === null ? 2 : 3);
    const p = outs.ptrs;
    M.permuteReturnsInto(xp, x.length, mode, seed, col0, ncols, p[0], p[1],
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
  const out = barsBlock(open, high, low, close, mode, seed, 0, n, vol, vm === "notional");
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
  const out = returnsBlock(x, mode, seed, 0, n, vol, vm === "notional");
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
  const ok = v && v.shape && ((v.shape.length === 2 && v.shape[1] === 4) ||
                              (v.shape.length === 3 && v.shape[2] === 4));
  if (!ok)
    throw new RangeError(
      "strategy must return a backtest engine output of shape (T, 4) " +
      "[equity, pnl, position, cost], a portfolio output of shape (T, K, 4), " +
      "or a pipeline output holding one");
  return v as NdArray;
}

/**
 * Test a whole strategy against the same strategy run on randomized data.
 * `strategy(...inputs)` must return a backtest engine output of shape (T, 4)
 * (or a pipeline `{values, index}` holding one). `method` is `"bars"` (permute
 * the OHLC inputs at `ohlc`), `"returns"` (permute the price input at
 * `price`), or `"shift"` (rotate the feature inputs at `columns` by one random
 * offset). With bars and returns, `mode` is "log", "diff", or "level", and
 * other inputs move with their bar unless listed in `fixed`; the input at
 * `volume` is moved and, with `volumeMode: "notional"`, rescaled to keep the
 * traded amount. Mirrors Python's `strategy_test`; the same seed gives the same
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
  const { name, larger } = metricSpec(metric, opts.largerIsBetter);
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
  const mode = opts.mode ?? "log", volumeMode = opts.volumeMode ?? "keep";
  checkModes(mode, volumeMode);
  const volCol = opts.volume === undefined ? null : check([opts.volume], "volume")[0];
  if (volCol !== null && method === "shift")
    throw new RangeError('volume applies to method "bars" or "returns"');
  const notional = volumeMode === "notional";
  let priceCols: number[] = [], shifted = new Set<number>(), shift = 0;
  if (method === "bars") {
    priceCols = check([...(opts.ohlc ?? [0, 1, 2, 3])], "ohlc");
    if (priceCols.length !== 4) throw new RangeError("ohlc must list the positions of open, high, low, close");
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

  const scorer = makeScorer(metric, opts.periodsPerYear ?? 1);
  try {
    const obs = scorer.score(engineOutput(strategy(...data)));
    const observed = obs.value;
    const nullStats = statsRecord(() => new Float64Array(n));
    const positions = Float64Array.from({ length: T }, (_, i) => i);
    const nul: number[] = [];
    const chunk = Math.max(1, Math.min(n, 64));
    for (let c0 = 0; c0 < n; c0 += chunk) {
      const m = Math.min(chunk, n - c0);
      let replaced = new Map<number, Float64Array>(), src: Float64Array;
      if (method === "bars") {
        const vol = volCol === null ? null : data[volCol];
        const b = barsBlock(data[priceCols[0]], data[priceCols[1]], data[priceCols[2]],
                            data[priceCols[3]], mode, seed, c0, m, vol, notional);
        priceCols.forEach((i, q) => replaced.set(i, b[q]));
        if (volCol !== null) replaced.set(volCol, b[5]);
        src = b[4];
      } else if (method === "returns") {
        const vol = volCol === null ? null : data[volCol];
        const r = returnsBlock(data[priceCols[0]], mode, seed, c0, m, vol, notional);
        replaced.set(priceCols[0], r[0]);
        if (volCol !== null) replaced.set(volCol, r[2]);
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
  const run = (x: ArrayLike<number>) => {
    engine.reset();
    return scorer.score(engine(x, ...market) as NdArray);
  };
  try {
    const obs = targets.map(run);
    const observed = Float64Array.from(obs, (r) => r.value);
    const observedStats = statsRecord((i) => Float64Array.from(obs, (r) => r.stats[i]));
    const nullStats = statsRecord((): NdArray => ({ data: new Float64Array(n * K), shape: [n, K] }));
    const nul = new Float64Array(n * K);
    const chunk = Math.max(1, Math.min(n, Math.floor(4_000_000 / Math.max(1, T))));
    const col = new Float64Array(T);
    for (let c0 = 0; c0 < n; c0 += chunk) {
      const m = Math.min(chunk, n - c0);
      targets.forEach((x, k) => {
        const block = shiftBlock(x, shift, seed, c0, m, start);
        for (let j = 0; j < m; j++) {
          for (let t = 0; t < T; t++) col[t] = block.data[t * m + j];
          const r = run(col);
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
