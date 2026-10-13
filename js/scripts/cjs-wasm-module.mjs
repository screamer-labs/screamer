// The CommonJS build's wasm_module.js. tsc's CommonJS output turns import()
// into require(), and require() cannot load the ES module screamer.mjs, so
// this file keeps a real import() out of tsc's sight. The CommonJS build runs
// in Node (a bundler takes the ES build), so it imports the ES build's copy
// of the module by its file URL: the package ships one copy.
//
//   node scripts/cjs-wasm-module.mjs dist/cjs/wasm_module.js
import { writeFileSync } from "node:fs";

const out = process.argv[2] ?? "dist/cjs/wasm_module.js";
writeFileSync(out, `"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.loadModule = void 0;
const { pathToFileURL } = require("node:url");
const { join } = require("node:path");
const dynamicImport = new Function("specifier", "return import(specifier)");
const loadModule = () => dynamicImport(pathToFileURL(join(__dirname, "..", "generated", "screamer.mjs")).href);
exports.loadModule = loadModule;
`);
console.log(`cjs-wasm-module: wrote ${out}`);
