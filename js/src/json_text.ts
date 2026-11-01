// JSON text of a value, with a non-finite number spelled as Python's json
// module and the C++ reader (include/screamer/common/json.h) write it (NaN,
// Infinity, -Infinity) and -0 as -0.0; JSON.stringify alone would write null
// and 0. Such a number goes through a marker string that no setting holds.
const MARK = "\u0000";
const MARKED = /"\\u0000(NaN|Infinity|-Infinity|-0\.0)"/g;
export function jsonText(x: unknown): string {
  return JSON.stringify(x, (_k, v) => (typeof v !== "number" ? v
    : !Number.isFinite(v) ? MARK + String(v) : Object.is(v, -0) ? MARK + "-0.0" : v))
    .replace(MARKED, "$1");
}
