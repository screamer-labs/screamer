// Imports the Emscripten module. A literal dynamic import: bundlers (esbuild,
// Vite, webpack, Rollup) follow it and bundle the module, and it needs no
// eval, so a strict Content-Security-Policy allows it. The CommonJS build
// replaces this file (scripts/cjs-wasm-module.mjs): tsc would turn the
// import() into a require(), which cannot load an ES module.
export const loadModule = (): Promise<{ default: () => Promise<any> }> =>
  import("./generated/screamer.mjs");
