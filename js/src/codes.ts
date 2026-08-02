// Resample / aggregation code tables, ported VERBATIM from screamer/dag.py so the
// JS graph builder emits the exact same integer codes as the Python bindings.
// The `tests/test_dag_codes_sync.py` gate fails if these drift from dag.py.
//
// Source of truth: screamer/dag.py
//   _RESAMPLE_AGG_CODE, _RESAMPLE_MODE_CODE, _RESAMPLE_FILL_CODE, _PLAN_AGG,
//   _BAR_AGG_FIXED_PLANS. The label code is derived inline in dag.py
//   (`1 if label == "right" else 0`); it is spelled out here as a table.

// ResampleAgg enum codes for the top-level agg= selector (matching the C++ enum
// ResampleAgg). 10 = OhlcvBars (dynamic plan, built in C++).
export const RESAMPLE_AGG_CODE: Record<string, number> = {
  first: 0, last: 1, min: 2, max: 3,
  sum: 4, count: 5, mean: 6, ohlc: 7,
  ohlcv_bars: 10,
};

// ResampleMode enum codes (matching C++ enum ResampleMode).
export const RESAMPLE_MODE_CODE: Record<string, number> = {
  by_index: 0, by_count: 1, by_cumulative: 2, by_clock: 3,
};

// ResampleFill enum codes (matching C++ enum ResampleFill).
export const RESAMPLE_FILL_CODE: Record<string, number> = {
  skip: 0, nan: 1, carry: 2,
};

// ResampleLabel: dag.py derives this inline (`1 if label == "right" else 0`).
// Spelled out here as the canonical table for the JS side and the sync gate.
export const RESAMPLE_LABEL_CODE: Record<string, number> = {
  left: 0, right: 1,
};

// ResampleAgg enum codes for plan entries (matching the C++ enum).
export const PLAN_AGG_CODE: Record<string, number> = {
  first: 0, last: 1, min: 2, max: 3,
  sum: 4, sum_pos: 8, sum_neg: 9,
};

// Per-column reducer plans for multi-column bar aggs. Each entry is a
// [agg_code, input_col_index] pair mapping an output column to a reducer.
//   ohlc_bars: [first O, max H, min L, last C]           - 4 in, 4 out
//   ohlcv:     [first, max, min, last (col0), sum (col1)] - 2 in, 5 out
//   ohlcv2:    [OHLC (col0), sumPos (col1), sumNeg (col1)]- 2 in, 6 out
// ohlcv_bars uses agg code 10 (OhlcvBars) with an EMPTY plan; C++ builds the
// plan dynamically from the resolved input width.
export const BAR_AGG_FIXED_PLANS: Record<string, Array<[number, number]>> = {
  ohlc_bars: [[0, 0], [3, 1], [2, 2], [1, 3]],
  ohlcv: [[0, 0], [3, 0], [2, 0], [1, 0], [4, 1]],
  ohlcv2: [[0, 0], [3, 0], [2, 0], [1, 0], [8, 1], [9, 1]],
};

// Aggregated export for the parity gate (`--dump` -> JSON). Keys mirror the
// Python dict names (minus the leading underscore) so the sync test can compare
// them field by field.
export const CODES = {
  RESAMPLE_AGG_CODE,
  RESAMPLE_MODE_CODE,
  RESAMPLE_FILL_CODE,
  RESAMPLE_LABEL_CODE,
  PLAN_AGG_CODE,
  BAR_AGG_FIXED_PLANS,
};

// `node --import tsx src/codes.ts --dump` prints CODES as JSON for the gate.
if (process.argv.includes("--dump")) {
  process.stdout.write(JSON.stringify(CODES));
}
