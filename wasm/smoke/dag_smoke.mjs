// DAG smoke test for the screamer WASM build.
//
//   node wasm/smoke/dag_smoke.mjs
//
// Builds a trivial graph  input -> RollingMean(3) -> output  via GraphBuilder,
// drives [1,2,3,4,5] through runBatchFlat, and asserts the output values equal
// [NaN,NaN,2,3,4] (NaN-aware). A build/link error or a wrong value is a real
// finding. Prints DAG SMOKE OK on success.

import init from "../build/screamer.mjs";

// Write a Float64Array's contents into a fresh heap double[]; return its pointer.
function heapF64(M, arr) {
  const ptr = M.allocF64(arr.length);
  M.viewF64(ptr, arr.length).set(arr);
  return ptr;
}

// Write a JS number array into a fresh heap uint32[]; return its byte pointer.
// Reuses the double allocator (8-byte units) and writes via a Uint32Array over
// the exported HEAPF64 buffer, fetched fresh in case memory grew.
function heapU32(M, arr) {
  const ptr = M.allocF64(Math.ceil((arr.length * 4) / 8) || 1);
  const view = new Uint32Array(M.HEAPF64.buffer, ptr, arr.length);
  view.set(arr);
  return ptr;
}

function nanEq(a, b, tol = 1e-9) {
  if (Number.isNaN(b)) return Number.isNaN(a);
  return Math.abs(a - b) <= tol;
}

async function main() {
  const M = await init();

  const gb = new M.GraphBuilder();
  const x = gb.addInput();

  // Functor node: RollingMean(3, "strict"). Hand its raw EvalOp* to addFunctor.
  const rm = new M.RollingMean(3, "strict");
  const rmPtr = M.opPtr(rm);

  const fnInputs = new M.VectorSizeT();
  fnInputs.push_back(x);
  const y = gb.addFunctor(rmPtr, fnInputs);
  fnInputs.delete();

  const outs = new M.VectorSizeT();
  outs.push_back(y);
  gb.setOutputs(outs);
  outs.delete();

  const cg = gb.compile();

  // Feed one input stream: index [0..4], value [1,2,3,4,5].
  const values = [1, 2, 3, 4, 5];
  const indices = [0, 1, 2, 3, 4];
  const N = values.length;

  const idxBuf = heapF64(M, indices);
  const valBuf = heapF64(M, values);
  const idxPtrs = heapU32(M, [idxBuf]);
  const valPtrs = heapU32(M, [valBuf]);
  const lens = heapU32(M, [N]);
  const widths = heapU32(M, [1]);

  const out = cg.runBatchFlat(idxPtrs, valPtrs, lens, widths, 1, 0);

  const gotVals = Array.from(M.viewF64(out.valuePtr, out.rows * out.width));
  const gotIdx = Array.from(M.viewF64(out.indexPtr, out.rows));

  // Free everything.
  M.freeBuf(out.indexPtr);
  M.freeBuf(out.valuePtr);
  M.freeBuf(idxBuf);
  M.freeBuf(valBuf);
  M.freeBuf(idxPtrs);
  M.freeBuf(valPtrs);
  M.freeBuf(lens);
  M.freeBuf(widths);
  cg.delete();
  rm.delete();
  gb.delete();

  const expect = [NaN, NaN, 2, 3, 4];
  console.log(`RollingMean(3) output: ${JSON.stringify(gotVals)}`);
  console.log(`indices: ${JSON.stringify(gotIdx)} rows=${out.rows} width=${out.width}`);

  let ok = out.width === 1 && gotVals.length === expect.length;
  if (ok) {
    for (let i = 0; i < expect.length; i++) {
      if (!nanEq(gotVals[i], expect[i])) ok = false;
    }
  }
  // Indices should be the right-labelled event indices [0..4].
  if (ok) {
    for (let i = 0; i < N; i++) if (gotIdx[i] !== indices[i]) ok = false;
  }

  if (!ok) {
    console.log(
      `DAG SMOKE FAILED: got=${JSON.stringify(gotVals)} expected=${JSON.stringify(expect)}`,
    );
    process.exit(1);
  }
  console.log("DAG SMOKE OK");
}

main().catch((err) => {
  console.error("dag smoke harness error:", err);
  process.exit(1);
});
