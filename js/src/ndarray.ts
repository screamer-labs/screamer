export interface NdArray { data: Float64Array; shape: number[]; }

// Throws unless the shape is a list of non-negative integers whose product is
// data.length, so C++ never reads past the end of an argument.
export function checkNd(a: NdArray): void {
  let n = 1;
  for (const d of a.shape) {
    if (!Number.isInteger(d) || d < 0) throw new TypeError(`shape [${a.shape}] must hold non-negative integers`);
    n *= d;
  }
  if (n !== a.data.length)
    throw new TypeError(`shape [${a.shape}] holds ${n} values, but data holds ${a.data.length}`);
}
export function toNested(a: NdArray): number[] | number[][] {
  if (a.shape.length <= 1) return Array.from(a.data);
  const [rows, cols] = a.shape;
  const out: number[][] = [];
  for (let r = 0; r < rows; r++) out.push(Array.from(a.data.subarray(r * cols, r * cols + cols)));
  return out;
}
