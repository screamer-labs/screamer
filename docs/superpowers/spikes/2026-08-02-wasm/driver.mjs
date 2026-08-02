// WASM spike driver. Run:  node docs/superpowers/spikes/2026-08-02-wasm/driver.mjs
//
// Exercises the six questions against the REAL screamer kernel compiled to
// WASM. Every number printed comes from screamer::detail::RollingMean::append.

import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const factory = (await import(join(here, 'screamer_spike.mjs'))).default;
const mod = await factory();

const line = (s) => console.log(s);
const fmt = (x) => (Number.isNaN(x) ? 'NaN' : String(x));
const fmtArr = (a) => '[' + Array.from(a, fmt).join(', ') + ']';

// ---------------------------------------------------------------------------
// Q2 batch marshalling. JS Float64Array -> WASM heap -> kernel -> Float64Array.
// The output buffer is malloc'd in C++ (the owner-capsule analogue); JS reads
// it through a zero-copy typed_memory_view, copies out, then frees both heap
// buffers. No leak, no use-after-free.
// ---------------------------------------------------------------------------
function batchRollingMean(input, size, policy = 'strict', { zeroCopy = false } = {}) {
  const arr = input instanceof Float64Array ? input : Float64Array.from(input);
  const n = arr.length;

  const inPtr = mod.allocF64(n);
  // Write via a fresh typed_memory_view over the just-allocated heap region.
  // (Re-fetch after every alloc: ALLOW_MEMORY_GROWTH can detach older views.)
  mod.viewF64(inPtr, n).set(arr);

  const outPtr = mod.rollingMeanBatchInto(inPtr, n, size, policy);

  // viewF64 returns a Float64Array aliasing the WASM heap (zero copy).
  const view = mod.viewF64(outPtr, n);
  const out = zeroCopy ? view : Float64Array.from(view);

  if (!zeroCopy) {
    // Safe to free once we have copied out.
    mod.freeBuf(inPtr);
    mod.freeBuf(outPtr);
  } else {
    // Caller owns the view; must copy before the next alloc and then free.
    // We free the input now (already consumed) but keep outPtr alive.
    mod.freeBuf(inPtr);
    return { view, free: () => mod.freeBuf(outPtr) };
  }
  return out;
}

// ---------------------------------------------------------------------------
// Q3 polymorphic factory. One callable dispatching on input type, mirroring
// screamer's Python Op(config)(data).
//   number            -> scalar path, advances a persistent stateful op
//   Float64Array/Array-> batch path, returns a Float64Array
//   other iterable    -> lazy generator, one append() per pulled event
// ---------------------------------------------------------------------------
function rollingMean(size, policy = 'strict') {
  const scalarOp = new mod.RollingMean(size, policy); // persistent stream state

  const fn = (data) => {
    if (typeof data === 'number') {
      return scalarOp.append(data); // scalar -> scalar, streaming
    }
    if (data instanceof Float64Array || Array.isArray(data)) {
      return batchRollingMean(data, size, policy); // batch -> Float64Array
    }
    if (data != null && typeof data[Symbol.iterator] === 'function') {
      return lazyRollingMean(data, size, policy); // iterable -> generator
    }
    throw new TypeError('rollingMean: unsupported input ' + typeof data);
  };
  fn.dispose = () => scalarOp.delete(); // Embind objects are manually freed
  return fn;
}

// ---------------------------------------------------------------------------
// Q4 lazy streaming. A generator that pulls one value at a time and drives a
// persistent WASM op per event.
// ---------------------------------------------------------------------------
function* lazyRollingMean(iterable, size, policy = 'strict') {
  const op = new mod.RollingMean(size, policy);
  try {
    for (const x of iterable) {
      yield op.append(x);
    }
  } finally {
    op.delete();
  }
}

// Q4 async streaming. An async generator over an async iterable, using JS's
// native for await...of. No coroutine bridge needed on the WASM object.
async function* lazyRollingMeanAsync(asyncIterable, size, policy = 'strict') {
  const op = new mod.RollingMean(size, policy);
  try {
    for await (const x of asyncIterable) {
      yield op.append(x);
    }
  } finally {
    op.delete();
  }
}

// ===========================================================================
// Demonstrations
// ===========================================================================
const INPUT = [1, 2, 3, 4, 5, NaN, 7];
// Reference from the actual Python screamer.RollingMean(3), strict policy:
const PY_REF = [NaN, NaN, 2, 3, 4, NaN, 5.333333333333333];

line('=== Q1: compile ===');
line('module loaded, RollingMean bound: ' + (typeof mod.RollingMean));

line('\n=== Q2: data types ===');
// Scalar round trip: number -> double -> append -> number.
const s = new mod.RollingMean(3, 'strict');
const scalarOut = INPUT.map((x) => s.append(x));
s.delete();
line('scalar per-event : ' + fmtArr(scalarOut));

// Batch: Float64Array in -> Float64Array out.
const batchOut = batchRollingMean(Float64Array.from(INPUT), 3);
line('batch (copied)   : ' + fmtArr(batchOut) + '  (typeof ' + batchOut.constructor.name + ')');

// Zero-copy typed_memory_view variant.
const zc = batchRollingMean(Float64Array.from(INPUT), 3, 'strict', { zeroCopy: true });
line('batch (zero-copy): ' + fmtArr(zc.view) + '  (aliases WASM heap)');
zc.free();

line('\n=== Q3: polymorphic single callable ===');
const rm = rollingMean(3);
line('number  1 -> ' + fmt(rm(1)));
line('number  2 -> ' + fmt(rm(2)));
line('number  3 -> ' + fmt(rm(3)) + '   (persistent stream state advances)');
const rm2 = rollingMean(3);
line('Array     -> ' + fmtArr(rm2([1, 2, 3, 4, 5])));
const rm3 = rollingMean(3);
line('Float64Ar -> ' + fmtArr(rm3(Float64Array.from([10, 20, 30, 40]))));
const rm4 = rollingMean(3);
const genOut = [...rm4(new Set([1, 2, 3, 4, 5]))]; // Set is iterable, not Array
line('iterable  -> ' + fmtArr(genOut) + '   (lazy generator)');
rm.dispose(); rm2.dispose(); rm3.dispose(); rm4.dispose();

line('\n=== Q4: lazy + async streaming ===');
const lazy = lazyRollingMean(INPUT, 3);
const lazyCollected = [];
for (const v of lazy) lazyCollected.push(v); // pulled one at a time
line('sync generator   : ' + fmtArr(lazyCollected));

async function* asyncSource(xs) {
  for (const x of xs) {
    await Promise.resolve(); // simulate awaited I/O between events
    yield x;
  }
}
const asyncCollected = [];
for await (const v of lazyRollingMeanAsync(asyncSource(INPUT), 3)) {
  asyncCollected.push(v);
}
line('async generator  : ' + fmtArr(asyncCollected));

line('\n=== Q5: correctness vs Python screamer.RollingMean(3) ===');
line('WASM batch : ' + fmtArr(batchOut));
line('Python ref : ' + fmtArr(PY_REF));
const eq = (a, b) =>
  a.length === b.length &&
  a.every((x, i) => (Number.isNaN(x) && Number.isNaN(b[i])) || x === b[i]);
const ok = eq(Array.from(batchOut), PY_REF) &&
           eq(scalarOut, PY_REF) &&
           eq(lazyCollected, PY_REF) &&
           eq(asyncCollected, PY_REF);
line('match (batch, scalar, lazy, async all == Python): ' + ok);

line('\n=== Q6: second distinct kernel wired (RollingSum) ===');
const sumInPtr = mod.allocF64(INPUT.length);
mod.viewF64(sumInPtr, INPUT.length).set(Float64Array.from(INPUT));
const sumOutPtr = mod.rollingSumBatchInto(sumInPtr, INPUT.length, 3, 'strict');
const sumOut = Float64Array.from(mod.viewF64(sumOutPtr, INPUT.length));
mod.freeBuf(sumInPtr); mod.freeBuf(sumOutPtr);
line('RollingSum(3)    : ' + fmtArr(sumOut));

if (!ok) process.exitCode = 1;
