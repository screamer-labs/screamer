// Load operators: Python's screamer.load, loads and from_settings. The C++
// core reads the files and builds the operators (serial::make_op, bound in
// wasm/serial_bindings.cpp); the save methods of every operator are in
// runtime.ts. Python and JavaScript read and write the same files.
import type { RawOp, Screamer } from "./loader.js";
import { SaveError, StateMismatch, StateVersionError, wasmCall } from "./errors.js";
import { current } from "./index.js";
import { jsonText } from "./json_text.js";
import { aKind } from "./kinds.js";
import { SETTINGS, wrapOp, wrapReducerOp, wrapRowOp, type ScreamerOp } from "./runtime.js";

/** The options of `load` and `loads`. */
export interface LoadOptions {
  /**
   * What to do with a state that does not fit the file's settings (an edited
   * setting) or that another state version wrote: `"error"` (the default)
   * raises; `"fresh"` keeps the settings, starts the operator fresh and warns
   * with `console.warn`.
   */
  onMismatch?: "error" | "fresh";
}

/** An operator's type and settings: the `"object"` of a save file, or `{ type, settings: op.settings() }`. */
export interface SettingsNode {
  type: string;
  settings?: Record<string, unknown>;
}

// The factory of fresh raw ops of `type` built from `text`, holding the first
// one, already built, for the wrapper. Made outside the function that holds
// the wrapper, so the factory (kept by the wrapper's finalizer) never reaches it.
function rawSource(M: Screamer, type: string, text: string): { raw: RawOp; src: () => RawOp } {
  const make = (): RawOp => wasmCall(() => M.serialMakeOp(type, text));
  let first: RawOp | null = make();
  const raw = first;
  return { raw, src: () => { const r = first ?? make(); first = null; return r; } };
}

// The record a generated factory makes of the settings it was given, read by
// Pipeline.toJSON (pipeline_io.ts): every setting of the file is given, null
// is not. A nested operator ({type, settings}) is recorded as a factory-built
// op is.
function record(cls: string, settings: Record<string, unknown>): { cls: string; names: string[]; given: unknown[] } {
  const names = Object.keys(settings);
  const given = names.map((n) => {
    const v: any = settings[n];
    if (v !== null && typeof v === "object" && !Array.isArray(v) && typeof v.type === "string" && "settings" in v) {
      const nested = () => undefined;
      (nested as any)[SETTINGS] = record(v.type, v.settings);
      return nested;
    }
    return v === null ? undefined : v;
  });
  return { cls, names, given };
}

// The operator `type` built from its settings (JSON text), wrapped as its
// factory wraps it.
function build(M: Screamer, type: string, text: string): { op: ScreamerOp; raw: RawOp } {
  const { raw, src } = rawSource(M, type, text);
  const op = raw.mapRowInto ? wrapRowOp(M, src) : raw.reduceInto ? wrapReducerOp(M, src) : wrapOp(M, src);
  (op as any)[SETTINGS] = record(type, wasmCall(() => M.serialSettingsCompat(type, text)));
  return { op, raw };
}

/**
 * A fresh operator from its type and settings, `{ type, settings }`: the
 * `"object"` of a save file, or one built from an operator's `settings()`. A
 * `state` in it is ignored. Python's `screamer.from_settings`.
 */
export function fromSettings(node: SettingsNode): ScreamerOp {
  if (node === null || typeof node !== "object" || Array.isArray(node))
    throw new SaveError(`expected {type: ..., settings: {...}}, got ${aKind(node)}`);
  const type: unknown = node.type;
  if (typeof type !== "string") throw new SaveError(`object.type: unknown type ${String(type)}`);
  return build(current(), type, jsonText(node.settings === undefined ? {} : node.settings)).op;
}

/**
 * Loads an operator from the file `dumps` returned: the JSON text, or the CBOR
 * bytes (the encoding is read from the data). The operator continues where
 * the saved one stopped. Python's `screamer.loads`.
 */
export function loads(data: string | Uint8Array | ArrayBuffer, options: LoadOptions = {}): ScreamerOp {
  const onMismatch = options?.onMismatch ?? "error";
  if (onMismatch !== "error" && onMismatch !== "fresh")
    throw new RangeError(`loads: onMismatch must be 'error' or 'fresh', got ${typeof onMismatch === "string" ? `'${onMismatch}'` : aKind(onMismatch)}`);
  const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : data;
  if (typeof bytes !== "string" && !(bytes instanceof Uint8Array))
    throw new TypeError(`loads: takes the string or the Uint8Array that dumps returned, got ${aKind(data)}`);
  const M = current();
  const node = wasmCall(() => M.serialLoadNode(bytes));
  const { op, raw } = build(M, node.type, node.settings);
  if (node.state !== null) {
    try {
      wasmCall(() => M.serialSetState(raw, node.state, node.type));
    } catch (e) {
      if (onMismatch === "fresh" && (e instanceof StateMismatch || e instanceof StateVersionError)) {
        // A failed load leaves the operator as built: fresh.
        console.warn(`${e.message}; ${node.type} starts fresh`);
      } else {
        op.dispose();
        throw e;
      }
    }
  }
  return op;
}

/**
 * Loads an operator from a file that `op.save` wrote (Node). Python's
 * `screamer.load`. See `loads` for the options.
 */
export async function load(path: string, options: LoadOptions = {}): Promise<ScreamerOp> {
  if (typeof path !== "string") throw new TypeError(`load: path must be a string, got ${aKind(path)}`);
  // node:fs is loaded here, on the first load, so a browser never asks for it.
  const fs = await import(["node:fs/promises"][0]);
  return loads(new Uint8Array(await fs.readFile(path)), options);
}
