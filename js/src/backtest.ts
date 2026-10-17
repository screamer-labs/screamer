// The report of a backtest engine's output. Mirrors screamer/backtest.py: the
// running columns and the summary are computed in C++
// (include/screamer/backtest_summary.h, bound in wasm/backtest_bindings.cpp),
// and this module only labels them with the names C++ gives, so a report reads
// the same in Python and JavaScript.
import { current } from "./index.js";
import { asNd, type NdArray } from "./ndarray.js";
import { wasmCall } from "./errors.js";
import { flattenRows, throwMessage, throwWrongKind, toF64Checked } from "./kinds.js";

/**
 * Market-order price sentinel for the backtest engines (Python's
 * `screamer.MARKET`). A quote or limit price of `MARKET` (or any non-finite
 * price) is a market order in the aggressive direction; NaN works as a
 * side-agnostic shorthand.
 */
export const MARKET = Infinity;

export interface BacktestReportOptions {
  /**
   * An index for the rows, returned as `running.index`, a Float64Array. Its
   * values must be numbers (a string is a TypeError); Python passes any index
   * through.
   */
  index?: ArrayLike<number>;
  /**
   * Annualizes `sharpe`, `sortino`, and `calmar` (for example 252 for daily
   * bars). The default 1 gives per-step ratios.
   */
  periodsPerYear?: number;
  /**
   * One label per row, numbers or strings (Python's `by=`). The result then
   * holds `byGroup`.
   */
  by?: ArrayLike<number | string>;
}

export interface BacktestReportResult {
  /**
   * The engine columns `equity`, `pnl`, `position`, `cost`, the
   * `BacktestReport` columns, the `BacktestStats` columns, and `index` when
   * one was given. Each is a causal series whose last finite value is the
   * summary.
   */
  running: Record<string, Float64Array>;
  /** The final statistics, keyed as Python's `backtest_report` summary. */
  summary: Record<string, number>;
  /**
   * With `by`: for each label, in order of first appearance, `total_pnl`,
   * `total_cost`, `turnover`, `sharpe`, `max_drawdown`, and `steps`, as
   * Python's `summary["by_group"]`.
   */
  byGroup?: Map<number | string, Record<string, number>>;
}

const WHO = "backtestReport";

// The names from WASM, read once: the engine columns, the computed columns
// (BacktestReport then BacktestStats), and the summary.
interface Names { engine: string[]; computed: string[]; summary: string[]; groups: string[] }
let names: Names | null = null;

function reportNames(): Names {
  if (names) return names;
  const [engine, report, stats, summary, groups] =
    wasmCall(() => current().backtestReportNames()) as string[][];
  names = { engine, computed: [...report, ...stats], summary, groups };
  return names;
}

// The labels of `by` as codes numbered in order of first appearance.
function groupCodes(by: ArrayLike<number | string>): { codes: Float64Array; labels: (number | string)[] } {
  const codes = new Float64Array(by.length);
  const labels: (number | string)[] = [];
  const seen = new Map<number | string, number>();
  for (let t = 0; t < by.length; t++) {
    const v = by[t];
    if (typeof v !== "number" && typeof v !== "string")
      throw new TypeError(`${WHO}: by item ${t} must be a number or a string, got ${typeof v}`);
    let c = seen.get(v);
    if (c === undefined) { c = labels.length; seen.set(v, c); labels.push(v); }
    codes[t] = c;
  }
  return { codes, labels };
}

interface EngineRows { data: Float64Array; ndim: number; T: number; ncols: number }

// The engine output as row-major data with the dims C++ checks: an NdArray as
// it is, or an array of T rows (flattenRows: null is NaN, a string is a
// TypeError). A flat array, a number, null, or undefined is not
// two-dimensional, which C++ rejects as Python does.
function engineRows(values: unknown, who: string = WHO): EngineRows {
  const nd = asNd(values);
  if (nd !== null)
    return { data: nd.data, ndim: nd.shape.length, T: nd.shape[0] ?? 0, ncols: nd.shape[1] ?? 0 };
  const notRows = { data: new Float64Array(0), ndim: 1, T: 0, ncols: 0 };
  if (values == null || typeof values === "number") return { ...notRows, ndim: 0 };
  if (ArrayBuffer.isView(values)) return notRows;
  if (!Array.isArray(values))
    throwWrongKind(who, "values", values, "an NdArray or an array of rows");
  if (values.length === 0 || values[0] === null || typeof values[0] === "number") return notRows;
  const { data, T, K } = flattenRows(values, who, "values");
  return { data, ndim: 2, T, ncols: K };
}

/**
 * Running report columns and a summary for a backtest engine's output.
 *
 * `values` is the `[T, 4]` output of a backtest engine, with columns
 * `[equity, pnl, position, cost]`, as an `NdArray` or an array of T rows.
 * `running` holds the four engine columns, the `BacktestReport` columns
 * (`drawdown`, `cum_cost`, `turnover`, `trades`, `max_drawdown`, `sharpe`),
 * and the `BacktestStats` columns (`exposure`, `round_trips`, `win_rate`,
 * `profit_factor`, `avg_win`, `avg_loss`, `avg_holding`, `sortino`,
 * `calmar`). `summary` holds `total_pnl`, `max_drawdown`, `total_cost`,
 * `turnover`, `num_trades`, `sharpe`, and the nine `BacktestStats` values.
 * The keys are those of Python's `backtest_report`, and so are the values.
 */
export function backtestReport(
  values: NdArray | ArrayLike<ArrayLike<number | null>>,
  opts: BacktestReportOptions | null = {},
): BacktestReportResult {
  const M = current();
  // The checks run in Python's order: the values, their shape (in C++), the
  // kind of periodsPerYear, and its value (in C++, in the call).
  const { data, ndim, T, ncols } = engineRows(values);
  wasmCall(() => M.checkEngineRows(ndim, ncols));
  const o = opts ?? {};
  const ppy = o.periodsPerYear === undefined ? 1 : o.periodsPerYear;
  if (typeof ppy === "boolean") throw new TypeError(`${WHO}: periodsPerYear must be a number, got boolean`);
  if (typeof ppy !== "number") throwWrongKind(WHO, "periodsPerYear", ppy);
  const index = o.index == null ? null
    : toF64Checked(o.index as ArrayLike<unknown>, WHO, (i) => `index item ${i}`);
  const { engine, computed, summary: summaryNames } = reportNames();
  let inPtr = 0, outPtr = 0, sumPtr = 0, codesPtr = 0, groupPtr = 0;          // freeBuf(0) is a no-op
  try {
    inPtr = M.allocF64(Math.max(1, data.length));
    outPtr = M.allocF64(Math.max(1, T * computed.length));
    sumPtr = M.allocF64(summaryNames.length);
    M.viewF64(inPtr, data.length).set(data);
    wasmCall(() => M.backtestReportInto(inPtr, T, ndim, ncols, ppy, outPtr, sumPtr));
    const running: Record<string, Float64Array> = {};
    // The engine columns come from the input; C++ writes the computed ones.
    engine.forEach((name, c) => {
      const col = new Float64Array(T);
      for (let t = 0; t < T; t++) col[t] = data[t * ncols + c];
      running[name] = col;
    });
    const out = M.viewF64(outPtr, T * computed.length);
    computed.forEach((name, i) => { running[name] = out.slice(i * T, (i + 1) * T); });
    if (index !== null) {
      // One index value per row (checked_index.h), as a Pipeline feed.
      throwMessage(M.feedLengthMessage(WHO, -1, T, index.length));
      running.index = index;
    }
    const sums = M.viewF64(sumPtr, summaryNames.length);
    const summary: Record<string, number> = {};
    summaryNames.forEach((name, i) => { summary[name] = sums[i]; });
    let byGroup: Map<number | string, Record<string, number>> | undefined;
    if (o.by != null) {
      const { codes, labels } = groupCodes(o.by);
      const { groups } = reportNames();
      codesPtr = M.allocF64(Math.max(1, codes.length));
      M.viewF64(codesPtr, codes.length).set(codes);
      groupPtr = M.allocF64(Math.max(1, labels.length * groups.length));
      wasmCall(() => M.backtestGroupsInto(inPtr, T, ndim, ncols, codesPtr, codes.length,
                                          labels.length, ppy, groupPtr));
      const g = M.viewF64(groupPtr, labels.length * groups.length);
      byGroup = new Map(labels.map((label, k) =>
        [label, Object.fromEntries(groups.map((name, j) => [name, g[k * groups.length + j]]))]));
    }
    return byGroup === undefined ? { running, summary } : { running, summary, byGroup };
  } finally {
    M.freeBuf(inPtr); M.freeBuf(outPtr); M.freeBuf(sumPtr); M.freeBuf(codesPtr); M.freeBuf(groupPtr);
  }
}

export interface BacktestTradesOptions {
  /**
   * An index for the rows: `entry` and `exit` hold its values instead of row
   * numbers. Its values must be numbers.
   */
  index?: ArrayLike<number>;
}

export interface BacktestTradesResult {
  entry: Float64Array;
  exit: Float64Array;
  /** +1 long, -1 short. */
  side: Float64Array;
  /** exit - entry, in rows. */
  bars: Float64Array;
  pnl: Float64Array;
  cost: Float64Array;
  /** 0 for a trade still open on the last row, else 1. */
  closed: Uint8Array;
}

/**
 * One row per round trip of a backtest engine's output, as
 * `BacktestStats` counts trades: flat to flat, and a sign flip closes one
 * trade and opens the next, with the flip row's pnl and cost going to the
 * closing trade. `values` is the `[T, 4]` engine output. Rows with
 * a NaN field are skipped; a trade open on the last row has `exit = T - 1`
 * and `closed = 0`. A `[T, K, 4]` portfolio output throws; call it once per
 * asset. Mirrors Python's `backtest_trades`.
 */
export function backtestTrades(
  values: NdArray | ArrayLike<ArrayLike<number | null>>,
  opts: BacktestTradesOptions | null = {},
): BacktestTradesResult {
  const who = "backtestTrades";
  const M = current();
  const { data, ndim, T, ncols } = engineRows(values, who);
  let inPtr = 0;
  try {
    inPtr = M.allocF64(Math.max(1, data.length));
    M.viewF64(inPtr, data.length).set(data);
    const cols = wasmCall(() => M.backtestTrades(inPtr, T, ndim, ncols)) as Float64Array;
    const n = cols.length / 7;
    const col = (k: number) => cols.slice(k * n, (k + 1) * n);
    const entry = col(0), exit = col(1);
    const o = opts ?? {};
    if (o.index != null) {
      const index = toF64Checked(o.index as ArrayLike<unknown>, who, (i) => `index item ${i}`);
      throwMessage(M.feedLengthMessage(who, -1, T, index.length));
      for (let i = 0; i < n; i++) { entry[i] = index[entry[i]]; exit[i] = index[exit[i]]; }
    }
    return { entry, exit, side: col(2), bars: col(3), pnl: col(4), cost: col(5),
             closed: Uint8Array.from(col(6)) };
  } finally {
    M.freeBuf(inPtr);
  }
}
