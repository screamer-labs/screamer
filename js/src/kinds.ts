// Wrong-kind and wrong-shape elements (rule A4: a wrong kind is a TypeError, a
// shape mismatch a RangeError). The dispatch tests its inputs with cheap
// checks; these helpers run only after one failed, so the happy path does no
// extra work and every message is built on the error path. The messages match
// the Python binding's, e.g. "RollingMean: event 1 is a M, expected a number"
// and "RollingMean: event 1 has shape (2,), expected a number".

// The name of x's type: its class name, or typeof for a primitive.
// The error a C++ rule reports as a message (wasm/dag_embind.h message_of):
// "" for none, else its class ('T' TypeError, 'R' RangeError) and its text.
// The bindings return messages because a throw out of an Embind function
// leaks its string arguments.
export function throwMessage(msg: string): void {
  if (!msg) return;
  throw msg[0] === "T" ? new TypeError(msg.slice(1)) : new RangeError(msg.slice(1));
}

// The engine's TypeError for an index that is not a number (checked_index.h).
export function throwIndexKind(M: any, index: unknown): never {
  throwMessage(M.indexKindMessage(kindName(index)));
  throw new TypeError("index must be a number");   // not reached
}

export function kindName(x: unknown): string {
  if (x === null) return "null";
  if (typeof x === "object" || typeof x === "function")
    return (x as any).constructor?.name || "object";
  return typeof x;
}

// x's type name with its article: "a string", "an Object", "an undefined".
export function aKind(x: unknown): string {
  const k = kindName(x);
  return (/^[aeiou]/i.test(k) ? "an " : "a ") + k;
}

// "event 3", "event 3 of input 1", or `prefix` alone.
export function whereText(prefix: string, index = -1, input = -1): string {
  let s = prefix;
  if (index >= 0) s += ` ${index}`;
  if (input >= 0) s += ` of input ${input}`;
  return s;
}

export function throwWrongKind(who: string, where: string, x: unknown, expected = "a number"): never {
  throw new TypeError(`${who ? `${who}: ` : ""}${where} is ${aKind(x)}, expected ${expected}`);
}

const isArrayLike = (x: any) =>
  x instanceof Float64Array || (x != null && x.data instanceof Float64Array && Array.isArray(x.shape));

// The search is bounded: each array is entered once (an array that holds
// itself, or one shared array repeated, is searched once), nesting stops at
// depth 64, and the whole search stops after BUDGET elements.
const BUDGET = 1_000_000;
type Search = { path: number[]; entered: Set<unknown>; budget: number; exhausted: boolean };

// The first element of x, depth first through arrays, that is not a number, an
// array or a Float64Array/NdArray.
function findWrongKind(x: unknown, s: Search, depth = 0): { bad: unknown } | null {
  if (--s.budget < 0) { s.exhausted = true; return null; }
  if (Array.isArray(x)) {
    if (depth >= 64 || s.entered.has(x)) return null;
    s.entered.add(x);
    for (let i = 0; i < x.length; i++) {
      s.path.push(i);
      const r = findWrongKind(x[i], s, depth + 1);
      if (r || s.exhausted) return r;
      s.path.pop();
    }
    return null;
  }
  if (typeof x === "number" || x === null || isArrayLike(x)) return null;   // null is NaN
  return { bad: x };
}

// Throws the wrong-kind TypeError for the first such element of x and returns
// when there is none, so the caller throws its own error. `where` names x
// itself; a nested element reads "item [1][0] of event 3". When the search
// runs out of budget it returns too: nothing wrong was found, so the caller's
// shape or conversion error stands.
export function throwIfWrongKind(who: string, x: unknown, where: string): void {
  const s: Search = { path: [], entered: new Set(), budget: BUDGET, exhausted: false };
  const r = findWrongKind(x, s);
  const self = where || "the input";
  if (s.exhausted || !r) return;
  if (s.path.length === 0) throwWrongKind(who, self, r.bad);
  const item = "item " + s.path.map((i) => `[${i}]`).join("");
  throwWrongKind(who, where ? `${item} of ${where}` : item, r.bad);
}

// One failed stream event, `items` holding one item per source.
export function throwIfWrongKindEvent(who: string, items: unknown[], event: number): void {
  for (let i = 0; i < items.length; i++)
    throwIfWrongKind(who, items[i], whereText("event", event, items.length > 1 ? i : -1));
}

// The shape of a nested array, Float64Array or NdArray as "(2, 3)", read along
// the first items, or "" when it is not regular enough to tell.
export function shapeText(x: unknown): string {
  const dims: number[] = [];
  let cur: any = x;
  for (let depth = 0; depth < 32; depth++) {
    if (Array.isArray(cur)) {
      dims.push(cur.length);
      if (cur.length === 0) break;
      cur = cur[0];
      continue;
    }
    if (cur instanceof Float64Array) { dims.push(cur.length); break; }
    if (isArrayLike(cur)) { dims.push(...cur.shape); break; }
    break;
  }
  if (dims.length === 0 || dims.length >= 32) return "";
  return `(${dims.join(", ")}${dims.length === 1 ? "," : ""})`;
}

// An element holds only numbers but has the wrong shape:
// "who: where has shape (2,), expected a number".
export function throwBadShape(who: string, where: string, x: unknown, expected = "a number"): never {
  const s = shapeText(x);
  throw new RangeError(`${who}: ${where} ${s ? `has shape ${s}` : "has an irregular shape"}, expected ${expected}`);
}

// An element that should be a number is not one: TypeError for a wrong kind
// inside it, else RangeError giving its shape.
export function throwNotANumber(who: string, x: unknown, where: string): never {
  throwIfWrongKind(who, x, where);
  throwBadShape(who, where, x);
}

// One element of a numeric feed. A boolean or undefined goes through ToNumber,
// as Float64Array.from did. A string is never a number: any string ("abc" or
// "3") is a wrong kind, as in Python (R-a). An object is a wrong kind, except
// one that converts to a number (a Date, an object with valueOf); an array, a
// function, a bigint or a symbol is always a wrong kind.
export function feedNumber(v: unknown, who: string, where: () => string): number {
  if (typeof v === "string" || typeof v === "function" || typeof v === "bigint" || typeof v === "symbol")
    throwWrongKind(who, where(), v);
  if (typeof v === "object" && v !== null) {
    if (Array.isArray(v) || v instanceof String) throwWrongKind(who, where(), v);
    const n = Number(v);
    if (Number.isNaN(n)) throwWrongKind(who, where(), v);
    return n;
  }
  return v as number;   // the Float64Array store applies ToNumber, as before
}

// One value of a feed: null is a missing value, NaN, as Python's None is
// (decision A9). Everything else is coerced as feedNumber. An index keeps
// feedNumber's rule.
export function feedValue(v: unknown, who: string, where: () => string): number {
  return v === null ? NaN : feedNumber(v, who, where);
}

// A numeric feed as a Float64Array, coerced element by element as feedNumber,
// or as feedValue when `values` is set.
export function toF64Checked(x: Float64Array | ArrayLike<unknown>, who: string, where: (i: number) => string,
                             values = false): Float64Array {
  if (x instanceof Float64Array) return x;
  return fillF64Checked(x, new Float64Array(x.length), who, where, values);
}

// The same, written into `out` (x.length items, for example a view of the
// WASM heap, which saves a copy). Returns `out`.
export function fillF64Checked(x: ArrayLike<unknown>, out: Float64Array, who: string,
                               where: (i: number) => string, values = false): Float64Array {
  const n = x.length;
  const one = values ? feedValue : feedNumber;
  for (let i = 0; i < n; i++) {
    const v = x[i];
    out[i] = typeof v === "number" ? v : one(v, who, () => where(i));
  }
  return out;
}

// T rows of K values as a row-major (T, K) block; K is the length of the first
// row. A row that is not an array is a wrong kind (TypeError), a row of another
// length a RangeError, and each item is read as feedValue reads it (null is
// NaN, a string a TypeError). `who` names the caller (may be empty) and
// `where` the input, as in "who: item [2][1] of values is a string, expected
// a number".
export function flattenRows(rows: ArrayLike<unknown>, who: string, where: string):
    { data: Float64Array; T: number; K: number } {
  const T = rows.length;
  const prefix = who ? `${who}: ` : "";
  const isRow = (r: any) => r != null && typeof r === "object" && typeof r.length === "number";
  const first = T > 0 ? (rows[0] as any) : undefined;
  const K = isRow(first) ? first.length : 0;
  const data = new Float64Array(T * K);
  for (let t = 0; t < T; t++) {
    const r = rows[t] as any;
    if (!isRow(r))
      throw new TypeError(`${prefix}row ${t} of ${where} is ${aKind(r)}, expected an array of numbers`);
    if (r.length !== K)
      throw new RangeError(`${prefix}every row of ${where} must hold ${K} values, got ${r.length} at row ${t}`);
    if (r instanceof Float64Array) { data.set(r, t * K); continue; }
    for (let c = 0; c < K; c++) {
      const v = r[c];
      data[t * K + c] = typeof v === "number" ? v
        : feedValue(v, who, () => `item [${t}][${c}] of ${where}`);
    }
  }
  return { data, T, K };
}
