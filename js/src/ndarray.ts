export interface NdArray { data: Float64Array; shape: number[]; }
export function toNested(a: NdArray): number[] | number[][] {
  if (a.shape.length <= 1) return Array.from(a.data);
  const [rows, cols] = a.shape;
  const out: number[][] = [];
  for (let r = 0; r < rows; r++) out.push(Array.from(a.data.subarray(r * cols, r * cols + cols)));
  return out;
}
