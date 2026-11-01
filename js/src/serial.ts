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
import { LOAD_STATE, Pipeline, type LiveDriver } from "./pipeline.js";

/** What a save file gives back: an operator, a Pipeline or a live session. */
export type Loaded = ScreamerOp | Pipeline | LiveDriver;

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
// `settings` has the renames applied (settingsCompat); a nested operator's
// are applied here.
function record(M: Screamer, cls: string, settings: Record<string, unknown>): { cls: string; names: string[]; given: unknown[] } {
  const names = Object.keys(settings);
  const given = names.map((n) => {
    const v: any = settings[n];
    if (v !== null && typeof v === "object" && !Array.isArray(v) && typeof v.type === "string" && "settings" in v) {
      const nested = () => undefined;
      const inner = wasmCall(() => M.serialSettingsCompat(v.type, jsonText(v.settings)));
      (nested as any)[SETTINGS] = record(M, v.type, inner);
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
  (op as any)[SETTINGS] = record(M, type, wasmCall(() => M.serialSettingsCompat(type, text)));
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

// A Pipeline from its file, or a live session that continues where the saved
// one stopped. A session's state fits only the description it was saved
// with: an edited graph raises StateMismatch, or with onMismatch "fresh"
// gives a fresh session of the edited Pipeline. The session owns the
// Pipeline it runs: its dispose() frees both.
function loadPipeline(node: { type: string; settings: string; state: string | null },
                      onMismatch: string): Pipeline | LiveDriver {
  let pipe: Pipeline;
  try {
    pipe = Pipeline.fromJSON(node.settings);
  } catch (e) {
    if (e instanceof SaveError || !(e instanceof TypeError || e instanceof RangeError)) throw e;
    throw new SaveError(`${node.type}.settings: ${e.message}`);
  }
  if (node.type === "Pipeline") {
    if (node.state !== null) {
      pipe.dispose();
      throw new SaveError('Pipeline.state: a Pipeline holds no state; a file with state is a live session\'s ("type": "LiveSession")');
    }
    return pipe;
  }
  const session = pipe.live();
  const end = session.dispose;
  const dispose = () => { end(); pipe.dispose(); };
  session.dispose = dispose;
  session[Symbol.dispose] = dispose;
  if (node.state !== null) {
    try {
      (session as any)[LOAD_STATE](node.state);
    } catch (e) {
      if (onMismatch === "fresh" && (e instanceof StateMismatch || e instanceof StateVersionError)) {
        // A failed load leaves the session as it was built: fresh.
        console.warn(`${e.message}; the LiveSession starts fresh`);
      } else {
        dispose();
        throw e;
      }
    }
  }
  return session;
}

/**
 * Loads an operator, a Pipeline or a live session from the file `dumps`
 * returned: the JSON text, or the CBOR bytes (the encoding is read from the
 * data). An operator or a session continues where the saved one stopped.
 * Python's `screamer.loads`.
 */
export function loads<T extends Loaded = ScreamerOp>(data: string | Uint8Array | ArrayBuffer, options: LoadOptions = {}): T {
  const onMismatch = options?.onMismatch ?? "error";
  if (onMismatch !== "error" && onMismatch !== "fresh")
    throw new RangeError(`loads: onMismatch must be 'error' or 'fresh', got ${typeof onMismatch === "string" ? `'${onMismatch}'` : aKind(onMismatch)}`);
  const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : data;
  if (typeof bytes !== "string" && !(bytes instanceof Uint8Array))
    throw new TypeError(`loads: takes the string or the Uint8Array that dumps returned, got ${aKind(data)}`);
  const M = current();
  const node = wasmCall(() => M.serialLoadNode(bytes));
  if (node.type === "Pipeline" || node.type === "LiveSession") return loadPipeline(node, onMismatch) as T;
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
  return op as T;
}

/**
 * Loads an operator, a Pipeline or a live session from a file that `save`
 * wrote (Node). Python's `screamer.load`. See `loads` for the options.
 */
export async function load<T extends Loaded = ScreamerOp>(path: string, options: LoadOptions = {}): Promise<T> {
  if (typeof path !== "string") throw new TypeError(`load: path must be a string, got ${aKind(path)}`);
  // node:fs is loaded here, on the first load, so a browser never asks for it.
  const fs = await import(["node:fs/promises"][0]);
  return loads<T>(new Uint8Array(await fs.readFile(path)), options);
}
