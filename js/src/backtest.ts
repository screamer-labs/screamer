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
}

const WHO = "backtestReport";

// The names from WASM, read once: the engine columns, the computed columns
// (BacktestReport then BacktestStats), and the summary.
interface Names { engine: string[]; computed: string[]; summary: string[] }
let names: Names | null = null;

function reportNames(): Names {
  if (names) return names;
  const [engine, report, stats, summary] =
    wasmCall(() => current().backtestReportNames()) as string[][];
  names = { engine, computed: [...report, ...stats], summary };
  return names;
}

interface EngineRows { data: Float64Array; ndim: number; T: number; ncols: number }

// The engine output as row-major data with the dims C++ checks: an NdArray as
// it is, or an array of T rows (flattenRows: null is NaN, a string is a
// TypeError). A flat array, a number, null, or undefined is not
// two-dimensional, which C++ rejects as Python does.
function engineRows(values: unknown): EngineRows {
  const nd = asNd(values);
  if (nd !== null)
    return { data: nd.data, ndim: nd.shape.length, T: nd.shape[0] ?? 0, ncols: nd.shape[1] ?? 0 };
  const notRows = { data: new Float64Array(0), ndim: 1, T: 0, ncols: 0 };
  if (values == null || typeof values === "number") return { ...notRows, ndim: 0 };
  if (ArrayBuffer.isView(values)) return notRows;
  if (!Array.isArray(values))
    throwWrongKind(WHO, "values", values, "an NdArray or an array of rows");
  if (values.length === 0 || values[0] === null || typeof values[0] === "number") return notRows;
  const { data, T, K } = flattenRows(values, WHO, "values");
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
  let inPtr = 0, outPtr = 0, sumPtr = 0;          // freeBuf(0) is a no-op
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
    return { running, summary };
  } finally {
    M.freeBuf(inPtr); M.freeBuf(outPtr); M.freeBuf(sumPtr);
  }
}
