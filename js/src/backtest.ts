// The report of a backtest engine's output. Mirrors screamer/backtest.py: the
// running columns and the summary are computed in C++
// (include/screamer/backtest_summary.h, bound in wasm/backtest_bindings.cpp),
// and this module only labels them with the names C++ gives, so a report reads
// the same in Python and JavaScript.
import { current } from "./index.js";
import { asNd, type NdArray } from "./ndarray.js";
import { wasmCall } from "./errors.js";
import { feedValue, throwWrongKind, toF64Checked } from "./kinds.js";

export interface BacktestReportOptions {
  /** An index for the rows, returned as `running.index`. */
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

// [engine columns, BacktestReport columns, BacktestStats columns, summary],
// read once from WASM.
let names: { columns: string[]; summary: string[] } | null = null;

function reportNames(): { columns: string[]; summary: string[] } {
  if (names) return names;
  const [engine, report, stats, summary] = wasmCall(() => current().backtestReportNames()) as string[][];
  names = { columns: [...engine, ...report, ...stats], summary };
  return names;
}

// The engine output as row-major data with the dims C++ checks: an NdArray as
// it is, or an array of T rows (null is NaN, a string is a TypeError). A flat
// array is one-dimensional, which C++ rejects as Python does.
function engineRows(values: unknown): { data: Float64Array; ndim: number; T: number; ncols: number } {
  const nd = asNd(values);
  if (nd !== null)
    return { data: nd.data, ndim: nd.shape.length, T: nd.shape[0] ?? 0, ncols: nd.shape[1] ?? 0 };
  const flat = { data: new Float64Array(0), ndim: 1, T: 0, ncols: 0 };
  if (ArrayBuffer.isView(values)) return flat;
  if (!Array.isArray(values)) throwWrongKind(WHO, "values", values, "an NdArray or an array of rows");
  const rows = values as unknown[];
  const T = rows.length;
  if (T === 0 || rows[0] === null || typeof rows[0] === "number") return flat;
  const ncols = (rows[0] as ArrayLike<unknown> | null)?.length ?? -1;
  const data = new Float64Array(T * Math.max(ncols, 0));
  for (let t = 0; t < T; t++) {
    const r = rows[t] as ArrayLike<unknown>;
    if (r == null || typeof r !== "object" || typeof r.length !== "number")
      throwWrongKind(WHO, `row ${t}`, r, "an array of numbers");
    if (r.length !== ncols) throw new RangeError(`${WHO}: every row must hold ${ncols} values`);
    for (let c = 0; c < ncols; c++) {
      const v = r[c];
      data[t * ncols + c] = typeof v === "number" ? v : feedValue(v, WHO, () => `item [${t}][${c}] of values`);
    }
  }
  return { data, ndim: 2, T, ncols };
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
  opts: BacktestReportOptions = {},
): BacktestReportResult {
  const M = current();
  const ppy = opts.periodsPerYear === undefined ? 1 : opts.periodsPerYear;
  if (typeof ppy !== "number") throwWrongKind(WHO, "periodsPerYear", ppy);
  const index = opts.index == null ? null
    : toF64Checked(opts.index as ArrayLike<unknown>, WHO, (i) => `index item ${i}`);
  const { data, ndim, T, ncols } = engineRows(values);
  const { columns, summary: summaryNames } = reportNames();
  let inPtr = 0, outPtr = 0, sumPtr = 0;          // freeBuf(0) is a no-op
  try {
    inPtr = M.allocF64(Math.max(1, data.length));
    outPtr = M.allocF64(Math.max(1, T * columns.length));
    sumPtr = M.allocF64(summaryNames.length);
    M.viewF64(inPtr, data.length).set(data);
    wasmCall(() => M.backtestReportInto(inPtr, T, ndim, ncols, ppy, outPtr, sumPtr));
    const out = M.viewF64(outPtr, T * columns.length);
    const sums = M.viewF64(sumPtr, summaryNames.length);
    const running: Record<string, Float64Array> = {};
    columns.forEach((name, i) => { running[name] = out.slice(i * T, (i + 1) * T); });
    if (index !== null) running.index = index;
    const summary: Record<string, number> = {};
    summaryNames.forEach((name, i) => { summary[name] = sums[i]; });
    return { running, summary };
  } finally {
    M.freeBuf(inPtr); M.freeBuf(outPtr); M.freeBuf(sumPtr);
  }
}
