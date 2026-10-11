// Wrong-kind elements (rule A4: a wrong kind is a TypeError). The dispatch
// tests its inputs with cheap checks; these helpers run only after one failed,
// so the happy path does no extra work and every message is built on the error
// path. The messages match the Python binding's, e.g.
// "RollingMean: event 1 is a M, expected a number".

// The name of x's type: its class name, or typeof for a primitive.
export function kindName(x: unknown): string {
  if (x === null) return "null";
  if (typeof x === "object" || typeof x === "function")
    return (x as any).constructor?.name || "object";
  return typeof x;
}

// "event 3", "event 3 of input 1", or `prefix` alone.
export function whereText(prefix: string, index = -1, input = -1): string {
  let s = prefix;
  if (index >= 0) s += ` ${index}`;
  if (input >= 0) s += ` of input ${input}`;
  return s;
}

export function throwWrongKind(who: string, where: string, x: unknown, expected = "a number"): never {
  throw new TypeError(`${who}: ${where} is a ${kindName(x)}, expected ${expected}`);
}

const isArrayLike = (x: any) =>
  x instanceof Float64Array || (x != null && x.data instanceof Float64Array && Array.isArray(x.shape));

// The first element of x, depth first through arrays, that is not a number, an
// array or a Float64Array/NdArray; `path` receives the indices to it.
function findWrongKind(x: unknown, path: number[], depth = 0): { bad: unknown } | null {
  if (Array.isArray(x)) {
    if (depth >= 64) return null;
    for (let i = 0; i < x.length; i++) {
      path.push(i);
      const r = findWrongKind(x[i], path, depth + 1);
      if (r) return r;
      path.pop();
    }
    return null;
  }
  if (typeof x === "number" || isArrayLike(x)) return null;
  return { bad: x };
}

// Throws the wrong-kind TypeError for the first such element of x and returns
// when there is none, so the caller throws its own error. `where` names x
// itself; a nested element reads "item [1][0] of event 3".
export function throwIfWrongKind(who: string, x: unknown, where: string): void {
  const path: number[] = [];
  const r = findWrongKind(x, path);
  if (!r) return;
  if (path.length === 0) throwWrongKind(who, where, r.bad);
  const item = "item " + path.map((i) => `[${i}]`).join("");
  throwWrongKind(who, where ? `${item} of ${where}` : item, r.bad);
}

// One failed stream event, `items` holding one item per source.
export function throwIfWrongKindEvent(who: string, items: unknown[], event: number): void {
  for (let i = 0; i < items.length; i++)
    throwIfWrongKind(who, items[i], whereText("event", event, items.length > 1 ? i : -1));
}

// A number[] or Float64Array of numbers as a Float64Array; any other element
// raises the wrong-kind TypeError naming its position.
export function toF64Checked(x: Float64Array | ArrayLike<unknown>, who: string, where: (i: number) => string): Float64Array {
  if (x instanceof Float64Array) return x;
  const n = x.length, out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const v = x[i];
    if (typeof v !== "number") throwWrongKind(who, where(i), v);
    out[i] = v as number;
  }
  return out;
}
